import re
import pdfplumber
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        # Regex for matching "MADDE X-" or "MADDE X –"
        self.madde_pattern = re.compile(r"MADDE\s+(\d+)\s*[-–]")
        # Regex for matching "BÖLÜM" titles
        self.bolum_pattern = re.compile(r"([A-ZÇĞİÖŞÜ]+\s+BÖLÜM)")

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"PDF okuma hatası ({pdf_path.name}): {e}")
        return text

    def semantic_chunking(self, text: str, doc_name: str) -> list[dict]:
        """
        Metni maddelere göre böler ve metadata ekler.
        """
        chunks = []
        
        # Maddelere göre böl
        parts = self.madde_pattern.split(text)
        # parts[0] is text before first MADDE
        # parts[1] is first MADDE number
        # parts[2] is first MADDE content
        
        doc_title = text.split('\n')[0:3] # İlk birkaç satır genelde başlıktır
        doc_title = " ".join([t.strip() for t in doc_title if t.strip()])

        # Başlangıç kısmını ekle (Amaç, Kapsam vb. genelde ilk maddelerdir ama öncesi de olabilir)
        if parts[0].strip():
            chunks.append({
                "content": parts[0].strip(),
                "metadata": {
                    "source": doc_name,
                    "title": doc_title,
                    "madde": "Giriş",
                    "type": "regulation"
                }
            })

        for i in range(1, len(parts), 2):
            madde_no = parts[i]
            madde_content = parts[i+1] if i+1 < len(parts) else ""
            
            # İçeriği temizle
            clean_content = madde_content.strip()
            if clean_content:
                chunks.append({
                    "content": f"MADDE {madde_no}- {clean_content}",
                    "metadata": {
                        "source": doc_name,
                        "title": doc_title,
                        "madde": madde_no,
                        "type": "regulation"
                    }
                })
        
        return chunks

    def process_all_pdfs(self, pdf_dir: Path) -> list[dict]:
        all_chunks = []
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            logger.info(f"İşleniyor: {pdf_file.name}")
            text = self.extract_text_from_pdf(pdf_file)
            chunks = self.semantic_chunking(text, pdf_file.name)
            all_chunks.extend(chunks)
            
        return all_chunks

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = DocumentProcessor()
    # Test path
    base_dir = Path(__file__).resolve().parent.parent.parent
    pdf_dir = base_dir / "data" / "raw_pdf"
    chunks = processor.process_all_pdfs(pdf_dir)
    
    output_path = base_dir / "data" / "processed_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"Toplam {len(chunks)} parça oluşturuldu ve {output_path} dosyasına kaydedildi.")
