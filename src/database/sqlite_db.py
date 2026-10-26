import sqlite3
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data" / "database" / "chatbot.db"


def create_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


def create_table():

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fiches (
        fiche_id TEXT PRIMARY KEY,
        seviye TEXT,
        kurum TEXT,
        fakulte TEXT,
        bolum TEXT,
        belge_turu TEXT,
        yonerge_id TEXT,
        yonerge_adi TEXT,
        bolum_no INTEGER,
        bolum_basligi TEXT,
        madde_no INTEGER,
        madde_basligi TEXT,
        fikra_no INTEGER,
        bent TEXT,
        metin TEXT,
        embedding_text TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_fiches(json_path):

    conn = create_connection()
    cursor = conn.cursor()

    with open(json_path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    for fiche in data:
        # Veriyi temizle ve None kontrolü yap
        clean_fiche = {k: (v if v is not None else "") for k, v in fiche.items()}
        
        cursor.execute("""
        INSERT OR REPLACE INTO fiches VALUES (
            :fiche_id,
            :seviye,
            :kurum,
            :fakulte,
            :bolum,
            :belge_turu,
            :yonerge_id,
            :yonerge_adi,
            :bolum_no,
            :bolum_basligi,
            :madde_no,
            :madde_basligi,
            :fikra_no,
            :bent,
            :metin,
            :embedding_text
        )
        """, clean_fiche)

    conn.commit()
    conn.close()