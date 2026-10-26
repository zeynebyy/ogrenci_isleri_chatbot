# GitHub yayınlama notları

Hedef depo: https://github.com/zeynebyy/ogrenci_isleri_chatbot

## Depo açıklaması

GitHub'ın About / Description alanı için:

> Bitlis Eren Üniversitesi yönergelerinden kaynak göstererek Türkçe yanıt üreten öğrenci işleri asistanı. Next.js, FastAPI, E5, FAISS, BM25 ve Ollama ile RAG tabanlı sohbet uygulaması.

## Önerilen konu etiketleri

```text
rag chatbot turkish student-affairs nextjs fastapi ollama faiss bm25 sentence-transformers python typescript
```

## Yayın kapsamı

- Güncel Next.js arayüzü: `frontend_v2/`.
- Python API, RAG, PDF işleme ve alternatif fişleme kodu: `src/`.
- Kaynak PDF yönergeler ve hazır veri: `data/raw_pdf/`, `data/processed_chunks.json`.
- Kurulum ve sistem açıklaması: kök `README.md`.

`.gitignore`; Python ortamlarını, `node_modules`, `.next`, logları, `.env` dosyalarını, veritabanlarını, üretilebilir indeksleri, çıktı klasörlerini ve iç içe proje kopyasını hariç tutar. Ayrı Git geçmişine sahip eski `frontend/` klasörü de hariç tutulur. Hariç tutmak bilgisayardaki dosyaları silmez.

`REPOR_TURKCE.md` önceki modernizasyon notlarını içerir. Güncel kurulum ve davranış için kök README esas alınmalıdır.

## Yayından sonra

1. About alanına kısa açıklamayı ve konu etiketlerini ekle.
2. İstersen güncel uygulama ekran görüntülerini README'ye ekle.
3. Paylaşım koşullarına uygun bir yazılım lisansı seç. Bu hazırlıkta kullanıcı adına lisans seçilmemiştir.

GitHub'a kod yüklemek API ve Ollama hizmetini otomatik olarak internete açmaz. Canlı demo için arayüz, Python API'si ve model sunucusu ayrıca barındırılmalıdır.
