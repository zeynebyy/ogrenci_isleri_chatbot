import logging
import time
import asyncio
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama

from src.rag.retriever import retriever_instance
from src.rag.rag_pipeline import build_prompt
from src.config import TOP_K_RESULTS, DEFAULT_MODEL, SIMILARITY_THRESHOLD

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Embedding modelini ve FAISS index'ini bir kez yükle
    logger.info("Uygulama başlatılıyor. Kaynaklar yükleniyor...")
    start_time = time.time()
    retriever_instance.load_resources()
    logger.info(f"Kaynaklar yüklendi. Süre: {time.time() - start_time:.2f} saniye.")
    yield
    logger.info("Uygulama kapatılıyor...")


app = FastAPI(
    title="BEÜ Chatbot API",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.get("/")
async def health():
    return {
        "status": "ok",
        "service": "beu-chatbot",
        "model": DEFAULT_MODEL,
        "resources_loaded": retriever_instance.model is not None
    }


@app.post("/chat/stream")
async def chat_stream(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    logger.info(f"[1] Soru alındı: {q.question}")

    # --- RAG: Semantik Arama ---
    search_start = time.time()
    results = retriever_instance.search(q.question, top_k=TOP_K_RESULTS)
    # --- Skor Kontrolü (Saçma Soruları Engelle) ---
    if not results or results[0].get("score", 0) < SIMILARITY_THRESHOLD:
        async def low_score():
            logger.warning(f"[!] Düşük skor ({results[0].get('score', 0) if results else 0}). Cevap reddedildi.")
            yield "Üzgünüm, sadece üniversite öğrenci işleri ile ilgili soruları cevaplayabilirim. Bu konu hakkında yönergelerde bilgi bulamadım."
        return StreamingResponse(low_score(), media_type="text/plain; charset=utf-8")

    contexts = [r.get("icerik", "") for r in results if r.get("icerik")]
    logger.info(f"[2] Arama tamamlandı ({time.time() - search_start:.2f}s). Bulunan bağlam: {len(contexts)}")

    if not contexts:
        async def no_context():
            msg = "Üzgünüm, sorunuzla ilgili üniversite yönergelerinde bilgi bulunamadı."
            logger.warning("[!] Bağlam bulunamadı.")
            yield msg
        return StreamingResponse(no_context(), media_type="text/plain; charset=utf-8")

    # --- Prompt Oluştur ---
    prompt = build_prompt(q.question, contexts)
    messages = [{"role": "user", "content": prompt}]

    # --- asyncio.Queue ile thread-safe streaming ---
    # Ollama'nın blocking iteratörü ayrı bir thread'de çalışır.
    # Async generator queue'dan okuyarak FastAPI'ya yield eder — event loop bloke olmaz.
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    _DONE = object()  # sentinel değer

    def ollama_worker():
        """Ollama blocking çağrısını ayrı thread'de çalıştır."""
        try:
            logger.info(f"[3] Ollama isteği gönderildi. Model: {DEFAULT_MODEL}")
            
            # Daha kararlı olması için .generate kullanıyoruz
            response = ollama.generate(
                model=DEFAULT_MODEL,
                prompt=prompt,
                stream=True,
                options={
                    "num_predict": 512,
                    "temperature": 0.1,  # Daha kesin cevaplar için
                    "top_p": 0.9,
                    "repeat_penalty": 1.2
                }
            )
            
            first_token = True
            for chunk in response:
                # generate() çıktısında parça 'response' anahtarındadır
                content = chunk.get('response', '')
                if content:
                    if first_token:
                        logger.info("[4] İlk token geldi, stream başlıyor...")
                        first_token = False
                    loop.call_soon_threadsafe(queue.put_nowait, content)
            
            logger.info("[5] Stream tamamlandı.")
        except ollama.ResponseError as e:
            logger.error(f"[!] Ollama hatası: {e.error}", exc_info=True)
            loop.call_soon_threadsafe(queue.put_nowait, f"Ollama hatası: {e.error}")
        except Exception as e:
            logger.error(f"[!] Beklenmeyen hata: {str(e)}", exc_info=True)
            loop.call_soon_threadsafe(queue.put_nowait, f"Sunucu hatası: {str(e)}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    # Ollama thread'ini başlat
    thread = threading.Thread(target=ollama_worker, daemon=True)
    thread.start()

    async def generate():
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            yield item

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)