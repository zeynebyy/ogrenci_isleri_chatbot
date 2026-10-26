import unicodedata

def read_text(path):
    """
    Dosyayı UTF-8 formatında okur ve NFC normalizasyonu uygular.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Bazı Unicode karakterleri normalize et (Örn: i + nokta birleşimi)
    text = unicodedata.normalize("NFC", text)
    return text