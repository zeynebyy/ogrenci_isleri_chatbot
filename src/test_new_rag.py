import json
from pathlib import Path
from src.services.rag_service import rag_service

def test_rag():
    base_dir = Path(__file__).resolve().parent.parent
    chunks_path = base_dir / "data" / "processed_chunks.json"
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print("RAG sistemi başlatılıyor...")
    rag_service.initialize(chunks)
    
    questions = [
        "Mazeret sınavına nasıl başvurabilirim?",
        "Staj dosyası ne zaman teslim edilir?",
        "Ders kaydı nasıl yapılır?",
        "Makarna nasıl yapılır?" # Bu alakasız olmalı
    ]
    
    for q in questions:
        print(f"\nSoru: {q}")
        print("Cevap: ", end="", flush=True)
        for chunk in rag_service.get_answer(q):
            print(chunk, end="", flush=True)
        print("\n" + "-"*50)

if __name__ == "__main__":
    test_rag()
