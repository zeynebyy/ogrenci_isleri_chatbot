import json
import sqlite3

def verify():
    print("--- VERIFICATION ---")
    
    # 1. JSON karakter kontrolü
    with open("data/vector_meta.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        case = [d for d in data if "çift_ana_dal" in d['fiche_id']][0]
        print(f"JSON Character Fix Check: {case['fiche_id']}")
        print(f"Content Sample: {case['icerik'][:100]}")
        
    # 2. Bağlam kontrolü (Context Injection)
    # İçerik artık 'Yönerge Adı - Madde - Metin' şeklinde olmalı
    if "Yönergesi" in case['icerik']:
        print("Context Injection: SUCCESS (Yönerge adı içerikte mevcut)")
    else:
        print("Context Injection: FAILURE (Yönerge adı bulunamadı)")

if __name__ == "__main__":
    verify()
