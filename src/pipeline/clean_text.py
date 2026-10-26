import re

def clean_text(text):

    text = text.replace("\r", "\n")

    text = re.sub(r"\n{2,}", "\n\n", text)

    text = re.sub(r"[ \t]+", " ", text)

    text = text.strip()

    return text