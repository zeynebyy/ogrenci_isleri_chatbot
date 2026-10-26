from pathlib import Path

# Proje kök dizini
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Klasörler
RAW_PDF_DIR = DATA_DIR / "raw_pdf"
RAW_TEXT_DIR = DATA_DIR / "raw_text"
CLEANED_TEXT_DIR = DATA_DIR / "cleaned_text"
FICHES_DIR = DATA_DIR / "fiches"
DATABASE_DIR = DATA_DIR / "database"

# Dosya Yolları
FICHES_FILE = FICHES_DIR / "fiches.json"
DB_PATH = DATABASE_DIR / "chatbot.db"
INDEX_PATH = DATA_DIR / "vector_index.faiss"
META_PATH = DATA_DIR / "vector_meta.json"

# Klasörleri otomatik oluştur
for folder in [RAW_TEXT_DIR, CLEANED_TEXT_DIR, FICHES_DIR, DATABASE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# LLM Ayarları
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"
LLM_TIMEOUT = 300 # Saniyeler (yavaş sistemler için 5 dakika)
TOP_K_RESULTS = 10 # Arama derinliği (Kalite için 5'ten 10'a çıkarıldı)
SIMILARITY_THRESHOLD = 0.75 # Saçma soruları engellemek için eşik değeri