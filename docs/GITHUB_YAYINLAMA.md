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

## Yayın öncesi doğrulama

26 Ekim 2026 tarihinde mevcut yerel ortamda:

| Kontrol | Sonuç |
| --- | --- |
| `frontend_v2`: `npm run build` | Başarılı; TypeScript ve statik sayfa üretimi tamamlandı. |
| `python -m src.test_rag_guardrails` | Mevcut `venv` ve yerel E5 önbelleğiyle başarılı. |
| İşlenmiş veri şeması | 302 parçanın içerik ve metadata alanları doğrulandı. |
| Yayın dosyaları | Ortam, bağımlılık, özel anahtar ve yaygın token kalıpları kontrol edildi; eşleşme bulunmadı. Bu kontrol tam bir güvenlik denetimi değildir. |
| `frontend_v2`: `npm run lint` | Başarısız: `page.tsx` başlangıç tema efektindeki mevcut `react-hooks/set-state-in-effect` hatası. |

Bu yayın hazırlığında uygulama davranışı değiştirilmedi. ESLint hatası, `useEffect` içinde başlangıç temasını `setTheme` ile doğrudan atayan mevcut koddadır. Temanın ilk yükleme ve tarayıcı tercihiyle eşitlenmesi ayrıca düzenlenmelidir.

Yerel kontroller temiz bir makinede sıfırdan kurulum yapıldığı anlamına gelmez. Python sürümleri sabitlenmediği için yeni ortam kurulumu ayrıca doğrulanmalıdır.

## Yayından sonra

1. About alanına kısa açıklamayı ve konu etiketlerini ekle.
2. İstersen güncel uygulama ekran görüntülerini README'ye ekle.
3. Paylaşım koşullarına uygun bir yazılım lisansı seç. Bu hazırlıkta kullanıcı adına lisans seçilmemiştir.

GitHub'a kod yüklemek API ve Ollama hizmetini otomatik olarak internete açmaz. Canlı demo için arayüz, Python API'si ve model sunucusu ayrıca barındırılmalıdır.
