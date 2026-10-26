# Öğrenci İşleri Chatbot Modernizasyon Raporu

Bu proje, mevcut öğrenci işleri chatbotunu production seviyesine taşımak için modernize edilmiştir. Yapılan iyileştirmeler aşağıda detaylandırılmıştır.

## 1. RAG Sistemi İyileştirmeleri
- **Semantik Parçalama (Semantic Chunking):** Dokümanlar artık rastgele karakter sayılarına göre değil, resmi yönetmelik maddelerine (`MADDE X`) göre bölünmektedir. Bu, bağlam kaybını önler.
- **Hibrit Arama (Hybrid Search):** Hem semantik (E5 Multilingual Embedding) hem de anahtar kelime tabanlı (BM25) arama birleştirilmiştir. Bu sayede "Mazeret" gibi spesifik terimler daha doğru yakalanır.
- **Niyet Sınıflandırma (Intent Classification):** Kullanıcının sorusu önce bir kategoriye (staj, muafiyet, vb.) atanır. Eğer soru alakasızsa (siyaset, genel kültür) sistem otomatik olarak reddeder.
- **Kaynak Gösterme (Citations):** Her cevap, ilgili yönetmelik adı ve madde numarasını referans olarak içerir.

## 2. Backend Geliştirmeleri (FastAPI & LangChain)
- **LangChain Entegrasyonu:** Modüler ve ölçeklenebilir bir RAG pipeline'ı kuruldu.
- **Streaming (Akış):** Cevaplar kullanıcıya kelime kelime akar, bu da bekleme hissini azaltır.
- **Güvenlik Katmanı:** Prompt injection ve kapsam dışı sorular için filtreler eklendi.

## 3. Frontend Modernizasyonu (Next.js & TailwindCSS)
- **Modern Arayüz:** Glassmorphism efekti, karanlık mod (Dark Navy/Purple) ve yumuşak geçişler kullanıldı.
- **Responsive Tasarım:** Mobil cihazlarda tam uyumlu çalışma sağlandı.
- **Framer Motion:** Mesaj balonları ve yükleme animasyonları için akıcı efektler eklendi.
- **Markdown Desteği:** Yönetmelik maddeleri listeli ve kalın yazım desteğiyle daha okunabilir hale getirildi.

## Kurulum ve Çalıştırma

### Backend
1. Gereksinimleri yükleyin: `pip install -r requirements.txt`
2. Dokümanları işleyin: `python src/utils/processor.py`
3. API'yi başlatın: `python src/api/v2_server.py`

### Frontend
1. `frontend_v2` dizinine gidin.
2. Bağımlılıkları yükleyin: `npm install`
3. Uygulamayı başlatın: `npm run dev`

## Teknik Yığın (Tech Stack)
- **Backend:** FastAPI, LangChain, Ollama (Llama 3.2), FAISS, BM25.
- **Frontend:** Next.js 14, TailwindCSS, Framer Motion, Shadcn/ui.
- **Veri İşleme:** pdfplumber, Regex-based semantic chunking.
