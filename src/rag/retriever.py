import faiss
import json
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from src.config import INDEX_PATH, META_PATH, TOP_K_RESULTS

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self):
        self.model = None
        self.index = None
        self.meta = None

    def load_resources(self):
        """Kaynakları belleğe yükler (Sadece 1 kere çalışır)."""
        if self.model is not None:
            return

        if not INDEX_PATH.exists() or not META_PATH.exists():
            logger.error(f"[Retriever] Hata: İndeks dosyaları bulunamadı! ({INDEX_PATH})")
            return

        logger.info("[Retriever] Güçlü E5 Modeli ve FAISS İndeksi yükleniyor...")
        try:
            # Daha güçlü multilingual model
            self.model = SentenceTransformer("intfloat/multilingual-e5-base")
            
            # İndeks yükleme
            self.index = faiss.read_index(str(INDEX_PATH))
            
            # Meta veri yükleme
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            
            logger.info(f"[Retriever] Hazır. Toplam kayıt: {len(self.meta)}")
        except Exception as e:
            logger.error(f"[Retriever] Yükleme hatası: {str(e)}")
            self.model = None

    def _turkish_lower(self, text):
        return text.replace("İ", "i").replace("I", "ı").lower()

    def search(self, question: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        if self.model is None:
            logger.warning("[Retriever] Model yüklü değil.")
            return []

        # 1. Semantik Arama (E5 modeli için 'query: ' prefixi gerekli)
        q_with_prefix = f"query: {question}"
        q_vec = self.model.encode([q_with_prefix], normalize_embeddings=True).astype("float32")
        distances, indices = self.index.search(q_vec, top_k * 3)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            entry = dict(self.meta[idx])
            entry["score"] = float(dist)
            results.append(entry)

        # 2. Anahtar Kelime Boost (Hybrid Search)
        q_lower = self._turkish_lower(question)
        tokens = [t for t in q_lower.split() if len(t) > 2]
        
        for r in results:
            text_lower = self._turkish_lower(r.get("icerik", ""))
            hits = sum(1 for t in tokens if t in text_lower)
            r["score"] += hits * 0.1  # Keyword eşleşme bonusu

        results.sort(key=lambda x: x["score"], reverse=True)
        final_results = results[:top_k]

        # --- DEBUG LOGLARI ---
        print("\n" + "="*50)
        print(f"🔍 RETRIEVAL DEBUG (Soru: {question})")
        for i, res in enumerate(final_results):
            print(f"[{i+1}] Skor: {res['score']:.4f} | Metin: {res['icerik'][:150]}...")
        print("="*50 + "\n")
        # ---------------------

        return final_results

# Singleton instance
retriever_instance = Retriever()
