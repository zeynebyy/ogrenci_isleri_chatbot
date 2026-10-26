import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss
import json

from src.config import DATA_DIR

DB_PATH = DATA_DIR / "database/chatbot.db"
INDEX_PATH = DATA_DIR / "vector_index.faiss"
META_PATH = DATA_DIR / "vector_meta.json"


def load_fiches():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Yönerge adı ve Madde başlığını alarak bağlamı güçlendiriyoruz
    cursor.execute("SELECT fiche_id, yonerge_adi, madde_basligi, metin FROM fiches")

    rows = cursor.fetchall()

    conn.close()

    return rows


def create_embeddings():

    print("Güçlü E5 Embedding modeli yükleniyor...")

    model = SentenceTransformer("intfloat/multilingual-e5-base")

    rows = load_fiches()

    # E5 modeli için dokümanların başına "passage: " eklenmelidir
    texts = [f"passage: {(r[1] or '')} - {(r[3] or '')}" for r in rows]

    print(f"{len(texts)} adet metin için embedding üretiliyor...")

    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("FAISS index oluşturuluyor...")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    # 🔥 HYBRID İÇİN GELİŞTİRİLMİŞ METADATA
    meta = []

    for r in rows:
        fiche_id, yonerge_adi, madde_basligi, metin = r
        # Metadata içine tam bağlamı (context) gömüyoruz
        full_context = f"{yonerge_adi} - Madde {fiche_id.split('_m')[1].split('_')[0]} - {metin}"
        
        meta.append({
            "fiche_id": fiche_id,
            "icerik": full_context, # LLM'e gidecek metin budur
            "lower_text": full_context.lower(),
            "tokens": full_context.lower().split()
        })

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Embedding ve index oluşturuldu.")


if __name__ == "__main__":
    create_embeddings()