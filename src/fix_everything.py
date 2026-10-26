import os
import shutil
from pathlib import Path
import subprocess
import sys

# 1. Kök dizini bul
BASE_DIR = Path(__file__).resolve().parent

def run_step(name, cmd):
    print(f"\n>>> ADIM: {name}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"--- {name} TAMAMLANDI ---")
    except Exception as e:
        print(f"!!! {name} HATASI: {str(e)}")
        sys.exit(1)

def cleanup():
    print(">>> Temizlik yapılıyor...")
    # Gereksiz dosyaları sil
    junk_files = [
        "data/fiches.json",
        "data/database/chatbot.db",
        "data/vector_index.faiss",
        "data/vector_meta.json",
        "test_res.json",
        "scratch/test_out.txt",
        "scratch/verify_res.json",
        "ogrenci.db",
        "test_db.py"
    ]
    for f in junk_files:
        path = BASE_DIR / f
        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
    print("--- Temizlik tamamlandı ---")

def main():
    # 1. Temizlik
    cleanup()

    # 2. Pipeline çalıştır (PDF -> Fiches)
    run_step("Metin Çıkarma ve Fişleme (main.py)", "python -m src.main")

    # 3. Veritabanı Oluştur (create_database.py)
    run_step("Veritabanı Oluşturma", "python -m src.create_database")

    # 4. Embedding Üret (embed_fiches.py)
    run_step("Vektör İndeksi Oluşturma (FAISS)", "python -m src.embeddings.embed_fiches")

    print("\n" + "="*50)
    print("TÜM SİSTEM BAŞARIYLA YENİLENDİ!")
    print("Şimdi backend sunucusunu başlatabilirsiniz:")
    print("python -m uvicorn src.api.server:app --reload")
    print("="*50)

if __name__ == "__main__":
    main()
