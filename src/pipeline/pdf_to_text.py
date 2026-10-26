from pdfminer.high_level import extract_text as pdf_extract
from pathlib import Path
import re

def convert_pdf_to_text(pdf_path, txt_path):
    """
    PDF dosyasını okur ve metne dönüştürür.
    pdfminer kullanarak daha sağlam bir extraction dener.
    """
    try:
        # pdfminer.high_level.extract_text daha başarılı olabiliyor
        text = pdf_extract(str(pdf_path))
        
        if not text or not text.strip():
            print(f"[Uyarı] {pdf_path.name} içeriği boş veya okunamadı.")
            return

        # Gereksiz boşlukları ve sayfa numarası gibi tekrarları temizleyebiliriz
        text = re.sub(r'\x0c', '\n', text) # Form feed character
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
            
    except Exception as e:
        print(f"[Hata] PDF dönüştürme başarısız ({pdf_path.name}): {str(e)}")