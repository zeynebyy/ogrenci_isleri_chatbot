# Öğrenci İşleri Chatbot — Web Arayüzü

BEÜ yönergelerini sorgulayan sohbet uygulamasının Next.js 16, React 19 ve TypeScript ile hazırlanmış güncel arayüzüdür.

## Çalıştırma

```powershell
npm ci
npm run dev
```

Tarayıcı adresi: `http://localhost:3000`.

Python API'sini proje kökünden ayrı bir terminalde başlat:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.v2_server:app --host 127.0.0.1 --port 8002
```

API adresi `src/app/page.tsx` dosyasındaki `API_URL` sabitindedir. Varsayılan adres `http://127.0.0.1:8002/api/chat` şeklindedir; `.env` üzerinden okunmaz.

## Özellikler

- Açık/koyu tema ve tarayıcıda saklanan tema tercihi.
- Hazır soru önerileri ve konuşmayı temizleme.
- Markdown yanıtları ve kaynak/madde gösterimi.
- Son sekiz geçmiş mesajını API'ye göndererek takip sorularını destekleme.
- Framer Motion animasyonları ve mobil uyumlu görünüm.

## Kontroller

```powershell
npm run lint
npm run build
npm start
```

`npm start` başarılı üretim derlemesinden sonra kullanılır. Google Fonts üzerinden Inter yazı tipi alındığından derleme sırasında ağ erişimi gerekebilir.

Python kurulumu, veri hazırlama, model ve API ayrıntıları için [ana README](../README.md) dosyasına bak.
