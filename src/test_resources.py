import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    print("Retriever instance import ediliyor...")
    from src.rag.retriever import retriever_instance
    print("Retriever instance import edildi.")
    
    print("Kaynaklar yükleniyor...")
    retriever_instance.load_resources()
    print("Kaynaklar başarıyla yüklendi.")
    
    print("Test araması yapılıyor...")
    res = retriever_instance.search("yatay geçiş", top_k=1)
    print(f"Sonuç: {res}")
    
    print("Test başarılı!")
except Exception as e:
    print(f"HATA: {e}")
    import traceback
    traceback.print_exc()
