import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.services.rag_service import rag_service


def load_rag():
    chunks_path = Path(__file__).resolve().parent.parent / "data" / "processed_chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    rag_service.initialize(chunks)


def ask(question: str) -> dict:
    return rag_service.answer(question, history=[])


def ask_with_history(question: str, history: list[dict]) -> dict:
    result = rag_service.answer(question, history=history)
    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result["answer"]},
        ]
    )
    return result


def assert_contains(text: str | None, expected: str):
    assert text and expected.lower() in text.lower(), f"Beklenen metin yok: {expected}\nCevap:\n{text}"


def assert_not_contains(text: str, unexpected: str):
    assert unexpected.lower() not in text.lower(), f"Alakasız metin bulundu: {unexpected}\nCevap:\n{text}"


def run_guardrail_tests():
    load_rag()

    cift_definition = ask("Çift anadal nedir?")
    assert_contains(cift_definition["answer"], "Çift ana dal")
    assert_contains(cift_definition["answer"], "iki diploma")
    assert_not_contains(cift_definition["answer"], "2.75")
    assert_not_contains(cift_definition["answer"], "Başvuru şartları")

    cift_docs = ask("Çift anadal için gerekli belgeler nelerdir?")
    assert_contains(cift_docs["answer"], "Gerekli belgeler")
    assert_contains(cift_docs["answer"], "Not döküm belgesi")
    assert_contains(cift_docs["answer"], "Başarı şartı belgesi")
    assert_contains(cift_docs["answer"], "Merkezi yerleştirme sınav sonuç belgesi")
    assert_contains(cift_docs["answer"], "Yabancı dil yeterlik sınav sonuç belgesi")
    assert_contains(cift_docs["answer"], "Madde 9")
    assert_not_contains(cift_docs["answer"], "yedek öğrencilerin kayıtları")
    assert_not_contains(cift_docs["answer"], "danışman")
    assert_not_contains(cift_docs["answer"], "koordinatör")
    assert_not_contains(cift_docs["answer"], "2.75")

    same_department = ask("Aynı bölüm için çift anadal yapılabilir mi?")
    assert_contains(same_department["answer"], "Hayır")
    assert_contains(same_department["answer"], "Aynı isimli bölümler arasında çift anadal başvurusu yapılamaz")
    assert_contains(same_department["answer"], "Madde 7")
    assert_not_contains(same_department["answer"], "2.75")
    assert_not_contains(same_department["answer"], "not döküm")
    assert_not_contains(same_department["answer"], "başvuru şartları")

    erasmus_docs = ask("Erasmus için gerekli belgeler nelerdir?")
    assert_contains(erasmus_docs["answer"], "tam bir belge listesi bulamadım")
    assert_contains(erasmus_docs["answer"], "Öğrenim anlaşması")
    assert_contains(erasmus_docs["answer"], "Not döküm belgesi")
    assert_contains(erasmus_docs["answer"], "Erasmus Koordinatörlüğünün")
    assert_not_contains(erasmus_docs["answer"], "akademik birim yönetim kurulu kararı")

    erasmus_exam = ask("Erasmus için dil sınavı şartı var mı?")
    assert_contains(erasmus_exam["answer"], "Evet")
    assert_contains(erasmus_exam["answer"], "yabancı dil sınavına girmesi zorunludur")
    assert_not_contains(erasmus_exam["answer"], "kabul mektubu")

    erasmus_country = ask("Hangi ülkelere Erasmus yapılabilir?")
    assert_contains(erasmus_country["answer"], "ülkelerin veya anlaşmalı üniversitelerin doğrudan listesi bulunmamaktadır")
    assert_contains(erasmus_country["answer"], "Erasmus Koordinatörlüğü")
    assert erasmus_country["sources"] == []

    tezsiz_definition = ask("Tezsiz yüksek lisans nedir?")
    assert_contains(tezsiz_definition["answer"], "doğrudan bir tanım cümlesi bulamadım")
    assert_not_contains(tezsiz_definition["answer"], "BİTLİS EREN ÜNİVERSİTESİ TEZSİZ")

    leak_history = []
    ask_with_history("Çift anadal nasıl yapılır?", leak_history)
    cap_docs = ask_with_history("başvuru belgeleri nelerdir?", leak_history)
    assert_contains(cap_docs["conversation_state"]["expanded_query"], "çift anadal")
    assert_contains(cap_docs["answer"], "Not döküm belgesi")
    tezsiz_switch = ask_with_history("Tezsiz yüksek lisans nasıl yapılır?", leak_history)
    assert tezsiz_switch["conversation_state"]["expanded_query"] is None
    assert_contains(tezsiz_switch["answer"], "Tezsiz yüksek lisans")
    assert_not_contains(tezsiz_switch["answer"], "Çift anadal")
    rector = ask_with_history("rektör kimdir?", leak_history)
    assert rector["sources"] == []
    assert_contains(rector["answer"], "yeterli ve doğrudan bilgi bulamadım")

    special = ask("özel öğrenci kimdir")
    assert_contains(special["answer"], "farklı bir yükseköğretim")
    assert_contains(special["answer"], "Madde 3")
    assert_not_contains(special["answer"], "Komisyon")

    special_history = []
    ask_with_history("özel öğrenci kimdir", special_history)
    special_followup = ask_with_history("şartları nedir madde madde anlat", special_history)
    assert_contains(special_followup["conversation_state"]["expanded_query"], "özel öğrenci")
    assert_contains(special_followup["answer"], "2.50/4.00")

    club_docs = ask("Kulüp açmak için gerekli belgeler nelerdir?")
    assert_contains(club_docs["answer"], "Gerekli belgeler")
    assert_contains(club_docs["answer"], "Başvuru dilekçeleri")
    assert_contains(club_docs["answer"], "Öğrenci Kulübü Tüzüğü")
    assert_contains(club_docs["answer"], "disiplin cezası")
    assert_not_contains(club_docs["answer"], "en az 15 kurucu")

    weather = ask("hava nasıl")
    assert weather["sources"] == []
    assert_contains(weather["answer"], "yeterli ve doğrudan bilgi bulamadım")

    erasmus_definition = ask("Erasmus nedir?")
    assert_contains(erasmus_definition["answer"], "Erasmus")
    assert_not_contains(erasmus_definition["answer"], "2.20/4.00")

    print("RAG guardrail testleri başarılı.")


if __name__ == "__main__":
    run_guardrail_tests()
