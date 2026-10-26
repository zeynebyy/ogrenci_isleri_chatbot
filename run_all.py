import sys
import os
import json

# src klasörünü path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pipeline.pdf_to_text import convert_pdf_to_text
from pipeline.clean_text import clean_text
from pipeline.parse_legal_text import parse_articles
from fiche.builder import build_fiches
from database.sqlite_db import insert_fiches

RAW_PDF_DIR = "data/raw_pdf"
RAW_TEXT_DIR = "data/raw_text"

# klasör yoksa oluştur
os.makedirs(RAW_TEXT_DIR, exist_ok=True)

# 🔥 TÜM FICHELERİ BURADA TUTACAĞIZ
all_fiches = []

for file in os.listdir(RAW_PDF_DIR):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(RAW_PDF_DIR, file)
        txt_path = os.path.join(RAW_TEXT_DIR, file.replace(".pdf", ".txt"))

        print(f"İşleniyor: {file}")

        # 1. PDF → TXT
        convert_pdf_to_text(pdf_path, txt_path)

        # 2. TXT oku
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 3. Temizle
        cleaned = clean_text(text)

        # 4. Parse (MADDE, fıkra vs.)
        parsed = parse_articles(cleaned)

        # 5. YÖNERGE ID EKLE
        yonerge_id = file.replace(".pdf", "")
        for item in parsed:
            item["yonerge_id"] = yonerge_id

        # 6. Fiche oluştur
        fiches = build_fiches(parsed)

        # 🔥 hepsini biriktir
        for f in fiches:
              all_fiches.append(f.model_dump())

# 🔥 TEK JSON DOSYASI OLUŞTUR
json_path = os.path.join("data", "fiches.json")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_fiches, f, ensure_ascii=False, indent=2)

# 🔥 TEK SEFERDE DB'YE YÜKLE
insert_fiches(json_path)

print("TÜM PDFLER YÜKLENDİ 🚀")