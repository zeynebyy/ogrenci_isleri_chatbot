import sys
import os
import json

# src klasörünü path'e ekle ki importlar çalışsın
sys.path.append(os.path.join(os.getcwd(), "src"))

from rag.retriever import search

def test_retrieval(query):
    print(f"\n{'='*50}")
    print(f"Soru: {query}")
    print(f"{'='*50}")
    
    results = search(query)
    
    print(f"\nBulunan En Alakalı 3 Fiş:")
    for i, r in enumerate(results[:3]):
        fiche_id = r.get('fiche_id', 'N/A')
        text = r.get('icerik') or r.get('text') or r.get('metin') or "İçerik yok"
        print(f"\n--- Sonuç {i+1} ({fiche_id}) ---")
        print(text[:500] + "..." if len(text) > 500 else text)
    
    return results

if __name__ == "__main__":
    test_queries = [
        "Mezuniyet için kaç AKTS gereklidir?",
        "Erasmus değişim programı nedir?",
        "Birim koordinatörünün görevleri nelerdir?"
    ]
    
    for q in test_queries:
        test_retrieval(q)
