import json
from collections import Counter
from config import FICHES_FILE


def main():
    with open(FICHES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Toplam fiş: {len(data)}")

    fiche_ids = [item["fiche_id"] for item in data]
    duplicates = [k for k, v in Counter(fiche_ids).items() if v > 1]

    empty_texts = [item["fiche_id"] for item in data if not item["metin"].strip()]

    bent_without_fikra = [
        item["fiche_id"]
        for item in data
        if item["bent"] is not None and item["fikra_no"] is None
    ]

    print(f"Tekrarlı fiche_id sayısı: {len(duplicates)}")
    print(f"Boş metin sayısı: {len(empty_texts)}")
    print(f"Bent var ama fıkra yok sayısı: {len(bent_without_fikra)}")

    if duplicates:
        print("Tekrarlı ID örnekleri:")
        print(duplicates[:10])

    if empty_texts:
        print("Boş metin örnekleri:")
        print(empty_texts[:10])

    if bent_without_fikra:
        print("Bent/fıkra problemi örnekleri:")
        print(bent_without_fikra[:10])


if __name__ == "__main__":
    main()