from .models import Fiche


def build_fiche_id(yonerge_id, madde_no, fikra_no=None, bent=None):

    parts = [yonerge_id, f"m{madde_no}"]

    if fikra_no is not None:
        parts.append(f"f{fikra_no}")

    if bent is not None:
        parts.append(bent)

    return "_".join(parts)


def build_embedding_text(item):

    parts = [f"Madde {item['madde_no']}"]

    if item.get("madde_basligi"):
        parts.append(item["madde_basligi"])

    if item.get("fikra_no") is not None:
        parts.append(f"Fıkra {item['fikra_no']}")

    if item.get("bent") is not None:
        parts.append(f"Bent {item['bent']}")

    parts.append(item["text"])

    return " ".join(parts)


def build_tags(item):

    tags = []

    if item.get("madde_basligi"):
        tags.append(item["madde_basligi"].lower())

    text_lower = item["text"].lower()

    for keyword in [
        "akts",
        "erasmus",
        "farabi",
        "mevlana",
        "transkript",
        "diploma",
        "bologna",
    ]:
        if keyword in text_lower:
            tags.append(keyword)

    return sorted(set(tags))


def build_fiches(parsed_data):

    fiches = []

    for item in parsed_data:

        yonerge_id = item["yonerge_id"]

        # yonerge adı oluştur (dosya adından)
        yonerge_adi = yonerge_id.replace("_", " ").title()

        fiche = Fiche(

            fiche_id=build_fiche_id(
                yonerge_id=yonerge_id,
                madde_no=item["madde_no"],
                fikra_no=item.get("fikra_no"),
                bent=item.get("bent"),
            ),

            seviye="universite",
            kurum="Bitlis Eren Üniversitesi",

            fakulte=None,
            bolum=None,

            belge_turu="yonerge",

            yonerge_id=yonerge_id,
            yonerge_adi=yonerge_adi,

            bolum_no=item.get("bolum_no"),
            bolum_basligi=item.get("bolum_basligi"),

            madde_no=item["madde_no"],
            madde_basligi=item.get("madde_basligi"),

            fikra_no=item.get("fikra_no"),
            bent=item.get("bent"),

            metin=item["text"].strip(),

            embedding_text=build_embedding_text(item),

            tags=build_tags(item),

            version=1
        )

        fiches.append(fiche)

    return fiches