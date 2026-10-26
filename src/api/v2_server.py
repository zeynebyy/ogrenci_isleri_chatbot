import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import DEFAULT_MODEL
from src.services.rag_service import rag_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    base_dir = Path(__file__).resolve().parent.parent.parent
    chunks_path = base_dir / "data" / "processed_chunks.json"

    if not chunks_path.exists():
        logger.error("İşlenmiş fiş dosyası bulunamadı: %s", chunks_path)
    else:
        try:
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            rag_service.initialize(chunks)
            logger.info("RAG sistemi başarıyla başlatıldı.")
        except Exception:
            logger.exception("RAG sistemi başlatılamadı.")
    yield


app = FastAPI(title="Öğrenci İşleri Chatbot API V2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatHistoryItem] = []


@app.get("/health")
async def health():
    return {"status": "ok", "model": DEFAULT_MODEL}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Mesaj boş olamaz.")

    logger.info("Soru alındı: %s", request.message)

    try:
        history = [item.model_dump() for item in request.history[-8:]]
        result = rag_service.answer(request.message, history=history)
        return JSONResponse(result)
    except Exception:
        logger.exception("Chat endpoint beklenmeyen hata.")
        return JSONResponse(
            {
                "answer": (
                    "Bu konuda sistemde yeterli bilgi bulamadım. Güncel bilgi için ilgili "
                    "üniversite/fakülte sayfasını veya öğrenci işleri birimini kontrol edebilirsiniz."
                ),
                "sources": [],
                "error": "server_error",
            },
            status_code=200,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
