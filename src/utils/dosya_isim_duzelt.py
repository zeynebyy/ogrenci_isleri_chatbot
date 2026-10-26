import os
import re
import unicodedata

FOLDER = "veri/temizlenmis_metin"

def normalize_filename(filename):
    # Türkçe karakterleri kaldır
    normalized = unicodedata.normalize("NFKD", filename).encode("ascii", "ignore").decode("ascii")
    
    # Fazla .txt uzantılarını temizle
    normalized = re.sub(r"\.txt.*$", ".txt", normalized)
    
    # Küçük harf yap
    normalized = normalized.lower()
    
    return normalized

for filename in os.listdir(FOLDER):
    old_path = os.path.join(FOLDER, filename)
    
    if os.path.isfile(old_path):
        new_name = normalize_filename(filename)
        new_path = os.path.join(FOLDER, new_name)
        
        os.rename(old_path, new_path)
        print(f"{filename} -> {new_name}")