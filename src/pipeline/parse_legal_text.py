import re


ARTICLE_PATTERN = re.compile(r"MADDE\s+(\d+)", re.IGNORECASE)
PARAGRAPH_PATTERN = re.compile(r"\((\d+)\)")
CLAUSE_PATTERN = re.compile(r"(?:^|\n)\s*([a-zçğıöşü])\)")


def clean_text(text):
    """Metni temizler, gereksiz satır sonlarını birleştirir."""
    text = text.replace("\r", "")
    
    # Satır sonundaki kelime bölünmelerini (tireli olanlar hariç) birleştir
    # Nokta veya noktalı virgül ile bitmeyen satırları bir sonraki ile birleştirmeyi deneriz
    # Ancak bu tehlikeli olabilir, o yüzden sadece tekli satır sonlarını boşluğa çevirelim
    # ama madde başları gibi yapıları koruyalım.
    
    lines = text.split("\n")
    cleaned_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        
        # Eğer satır bir madde başlığı veya fıkra başlığı değilse yanındakine yapıştırılabilir
        # Ama şimdilik en güvenli yol sadece çoklu boşlukları temizlemek.
        cleaned_lines.append(line)
    
    # Satırları birleştirirken: Eğer önceki satır bir cümle bitişi değilse boşlukla birleştir
    result = ""
    for i, line in enumerate(cleaned_lines):
        if not line:
            result += "\n\n"
            continue
        
        if result and not result.endswith("\n") and not result.endswith(". ") and not result.endswith(": ") and not result.endswith("; "):
            result += " " + line
        else:
            if result and not result.endswith("\n"):
                 result += "\n"
            result += line
            
    # Çoklu boşlukları temizle
    result = re.sub(r"[ \t]+", " ", result)
    # Çoklu satır sonlarını temizle
    result = re.sub(r"\n{3,}", "\n\n", result)
    
    return result.strip()


def split_paragraphs(text):

    parts = PARAGRAPH_PATTERN.split(text)

    if len(parts) < 3:
        return []

    paragraphs = []

    for i in range(1, len(parts), 2):

        fikra_no = int(parts[i])

        fikra_text = parts[i + 1].strip()

        paragraphs.append((fikra_no, fikra_text))

    return paragraphs


def split_clauses(text):

    matches = list(CLAUSE_PATTERN.finditer(text))

    if not matches:
        return []

    clauses = []

    for i, match in enumerate(matches):

        bent = match.group(1)

        start = match.end()

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        bent_text = text[start:end].strip()

        clauses.append((bent, bent_text))

    return clauses


def parse_articles(text):

    text = clean_text(text)

    articles = []

    article_matches = list(ARTICLE_PATTERN.finditer(text))

    for i, match in enumerate(article_matches):

        madde_no = int(match.group(1))

        start = match.end()

        end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(text)

        madde_text = text[start:end].strip()

        paragraphs = split_paragraphs(madde_text)

        if not paragraphs:

            articles.append({
                "madde_no": madde_no,
                "madde_basligi": None,
                "fikra_no": None,
                "bent": None,
                "text": madde_text
            })

            continue

        for fikra_no, fikra_text in paragraphs:

            clauses = split_clauses(fikra_text)

            if clauses:

                for bent, bent_text in clauses:

                    articles.append({
                        "madde_no": madde_no,
                        "madde_basligi": None,
                        "fikra_no": fikra_no,
                        "bent": bent,
                        "text": bent_text
                    })

            else:

                articles.append({
                    "madde_no": madde_no,
                    "madde_basligi": None,
                    "fikra_no": fikra_no,
                    "bent": None,
                    "text": fikra_text
                })

    return articles