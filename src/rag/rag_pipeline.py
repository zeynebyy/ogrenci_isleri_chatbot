import requests
import json
import logging
from src.rag.retriever import retriever_instance
from src.config import OLLAMA_URL, DEFAULT_MODEL, LLM_TIMEOUT, TOP_K_RESULTS, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

import re

def clean_context_text(text: str) -> str:
    """Metni LLM için temizler: Satır sonlarını ve fazla boşlukları siler."""
    if not text: return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_prompt(question: str, contexts: list[str]) -> str:
    # Dokümanları temizleyip belirgin ayraçlarla birleştiriyoruz
    context_text = "\n\n".join([clean_context_text(c) for c in contexts if c])

    prompt = f"""Sen Bitlis Eren Üniversitesi Öğrenci İşleri asistanısın. 
Görevin: Aşağıdaki kuralları kullanarak öğrenciye sade, anlaşılır ve profesyonel bir cevap vermektir.

ÖNEMLİ KURALLAR:
1. SADE DİL: Akademik terimlerden kaçın. Bir öğrencinin kolayca anlayacağı günlük ama saygılı bir Türkçe kullan.
2. TEKRAR YAPMA: Aynı bilgileri veya cümleleri asla üst üste yazma.
3. GİRİŞSİZ BAŞLA: "Öğrenciye cevap:", "Yönergeye göre:" gibi ifadeleri ASLA kullanma. Doğrudan bilgiye geç.
4. REFERANS VERME: Madde numarası veya doküman ismi paylaşma. Sadece kuralları anlat.
5. FORMAT: Markdown kullan. Başlıkları **Kalın** yap.

BAĞLAM:
{context_text}

ÖĞRENCİ SORUSU: {question}

ÖNEMLİ: 
- Eğer soru yukarıdaki BAĞLAM ile ilgili değilse veya üniversite/öğrenci işleri dışındaysa, sadece: "Üzgünüm, sadece Bitlis Eren Üniversitesi öğrenci işleri ile ilgili soruları cevaplayabilirim." yaz.
- Eğer bağlamda cevap yoksa uydurma, "Bu konuda yönergelerde bilgi bulunamadı." yaz.

(Şimdi cevabı sade Türkçe ile, girişsiz ve tekrarsız yaz):"""
    return prompt

def ask_llm(prompt: str, stream: bool = False):
    try:
        logger.info(f"Ollama'ya istek gönderiliyor. Model: {DEFAULT_MODEL}")
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": DEFAULT_MODEL, 
                "prompt": prompt, 
                "stream": stream,
                "options": {
                    "num_predict": 512,
                    "temperature": 0.1,
                    "repeat_penalty": 1.2
                }
            },
            stream=stream,
            timeout=LLM_TIMEOUT
        )
        response.raise_for_status()
        return response
    except Exception as e:
        logger.error(f"[LLM Error] {str(e)}")
        return None

def rag_answer(question: str) -> str:
    # CLI veya test için kullanılır. Server'da retriever_instance kullanılır.
    retriever_instance.load_resources()
    results = retriever_instance.search(question, top_k=TOP_K_RESULTS)
    
    # Skor kontrolü: En iyi skor eşik değerinin altındaysa "bilgim yok" de.
    if not results or results[0].get("score", 0) < SIMILARITY_THRESHOLD:
        return "Üzgünüm, bu konuyla ilgili üniversite yönergelerinde yeterli bilgi bulamadım veya sorunuz kapsam dışı."

    contexts = [r.get("icerik", "") for r in results]
    
    if not any(contexts):
        return "Üzgünüm, bu konuyla ilgili yönergelerde hiçbir kayıt bulunamadı."
        
    prompt = build_prompt(question, contexts)
    response = ask_llm(prompt, stream=False)
    
    if response:
        full_text = ""
        for line in response.iter_lines():
            if line:
                full_text += json.loads(line).get("response", "")
        return full_text.strip()
    
    return "Hata: LLM sunucusuna ulaşılamadı."

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Soru: ")
    print("\n[Cevap]\n", rag_answer(q))