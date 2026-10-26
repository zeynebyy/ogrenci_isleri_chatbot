@echo off
echo ===================================================
echo   BEU OGRENCI ISLERI CHATBOT BASLATILIYOR 🎓
echo ===================================================

:: 1. Ollama kontrolu
echo [1/3] Ollama kontrol ediliyor...
tasklist /fi "imagename eq ollama.exe" | find ":" > nul
if errorlevel 1 (
    echo [OK] Ollama calisiyor.
) else (
    echo [UYARI] Ollama calismiyor! Baslatiliyor...
    start ollama serve
    timeout /t 5
)

:: 2. Backend Baslat
echo [2/3] Backend (FastAPI) baslatiliyor...
start "Chatbot Backend" cmd /k ".\venv\Scripts\python.exe -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000"

:: 3. Frontend Baslat
echo [3/3] Frontend (React) baslatiliyor...
cd frontend
start "Chatbot Frontend" cmd /k "npm start"

echo.
echo ===================================================
echo   Sistem baslatildi! 🚀
echo   Backend: http://localhost:8000
echo   Frontend: http://localhost:3000
echo ===================================================
pause
