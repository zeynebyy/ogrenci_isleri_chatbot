from src.config import RAW_PDF_DIR, RAW_TEXT_DIR, CLEANED_TEXT_DIR, FICHES_FILE
from src.pipeline.pdf_to_text import convert_pdf_to_text
from src.pipeline.clean_text import clean_text
from src.pipeline.parse_legal_text import parse_articles
from src.fiche.builder import build_fiches
from src.utils.file_utils import save_json
from src.utils.text_utils import read_text


def main():
    for pdf_file in RAW_PDF_DIR.glob("*.pdf"):
        txt_file = RAW_TEXT_DIR / f"{pdf_file.stem}.txt"
        convert_pdf_to_text(pdf_file, txt_file)

    parsed_all = []

    for txt_file in RAW_TEXT_DIR.glob("*.txt"):
        text = read_text(txt_file)
        cleaned = clean_text(text)

        clean_path = CLEANED_TEXT_DIR / txt_file.name
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        parsed = parse_articles(cleaned)
        yonerge_id = txt_file.stem

        for item in parsed:
            item["yonerge_id"] = yonerge_id
        
        parsed_all.extend(parsed)
        

    fiches = build_fiches(parsed_all)
    json_data = [f.model_dump() for f in fiches]
    save_json(FICHES_FILE, json_data)

    print(f"Fişleme tamamlandı. Toplam fiş: {len(json_data)}")
    print(f"Çıktı dosyası: {FICHES_FILE}")


if __name__ == "__main__":
    main()