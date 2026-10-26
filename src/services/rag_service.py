import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import faiss
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import DATA_DIR, DEFAULT_MODEL, LLM_TIMEOUT, OLLAMA_URL, TOP_K_RESULTS

logger = logging.getLogger(__name__)


STOP_WORDS = {
    "acaba",
    "ben",
    "bir",
    "bu",
    "da",
    "de",
    "diye",
    "icin",
    "için",
    "ile",
    "mi",
    "mu",
    "mı",
    "mü",
    "ne",
    "nedir",
    "neler",
    "nelerdir",
    "kadar",
    "kaç",
    "var",
    "ve",
}


DOCUMENT_RULES = [
    {
        "name": "ozel_ogrenci",
        "terms": {"ozel", "özel", "ogrenci", "ögrenci", "öğrenci"},
        "sources": ("ozel_ogrenci_yonergesi",),
        "boost": 1.4,
    },
    {
        "name": "staj",
        "terms": {"staj", "muhendislik", "mühendislik", "is", "iş", "gunu", "günü", "sure", "süre"},
        "sources": ("staj_yonergesi",),
        "boost": 1.5,
        "unless": {"ulusal", "kariyer", "kapisi", "kapısı"},
    },
    {
        "name": "ulusal_staj",
        "terms": {"ulusal", "staj", "kariyer", "kapisi", "kapısı"},
        "sources": ("ulusal_staj_programi_yonergesi",),
        "boost": 1.3,
    },
    {
        "name": "erasmus",
        "terms": {"erasmus", "basvuru", "şart", "sart", "kosul", "koşul", "on", "ön"},
        "sources": ("erasmus_degisim_programi_yonergesi",),
        "boost": 1.4,
    },
    {
        "name": "cift_anadal",
        "terms": {"cift", "çift", "anadal", "ana", "dal", "yandal", "yan"},
        "sources": ("çift_ana_dal_ve_yan_dal_programı_yonergesi",),
        "boost": 1.2,
    },
    {
        "name": "muafiyet",
        "terms": {"muafiyet", "intibak", "ders", "saydirma", "saydırma"},
        "sources": ("muafiyet_ve_intibak_yonergesi",),
        "boost": 1.2,
    },
    {
        "name": "yaz_okulu",
        "terms": {"yaz", "okulu"},
        "sources": ("yaz_okulu_yonergesi",),
        "boost": 1.2,
    },
    {
        "name": "diploma",
        "terms": {"diploma", "eki", "mezuniyet", "belge"},
        "sources": ("diploma_ve_diploma_eki_yonergesi",),
        "boost": 1.2,
    },
    {
        "name": "engelli_ogrenci",
        "terms": {"engelli", "ogrenci", "birim", "destek"},
        "sources": ("engelli_ogrenci_birimi_yonergesi",),
        "boost": 1.2,
    },
    {
        "name": "ogrenci_kulupleri",
        "terms": {"kulup", "kulüp", "topluluk", "danisman", "danışman"},
        "sources": ("ogrenci_kulupleri_yonergesi",),
        "boost": 1.2,
    },
    {
        "name": "ogrenci_senato",
        "terms": {"senato", "temsilci", "ogrenci", "öğrenci"},
        "sources": ("ogrenci_senato_yonergesi",),
        "boost": 1.1,
    },
    {
        "name": "uluslararasi_ogrenci",
        "terms": {"uluslararasi", "uluslararası", "yabanci", "yabancı", "uyruklu", "kabul"},
        "sources": (
            "yabancı_uyruklu_ogrenci_kabul_yonergesi",
            "lisanüstü_programlara_uluslararası_ogrenci_kabul_yonergesi",
        ),
        "boost": 1.2,
    },
    {
        "name": "not_sistemi",
        "terms": {"not", "ortalama", "bagil", "bağıl", "mutlak", "sinav", "sınav"},
        "sources": ("bagıl_degerlendirme_sistemi", "mutlak_not_sistemi"),
        "boost": 1.15,
    },
    {
        "name": "sosyal_transkript",
        "terms": {"sosyal", "transkript", "etkinlik"},
        "sources": ("sosyal_transkript_yonergesi",),
        "boost": 1.2,
    },
    {
        "name": "tezsiz_yuksek_lisans",
        "terms": {"tezsiz", "yuksek", "yüksek", "lisans", "donem", "dönem"},
        "sources": ("tezsiz_yuksek_lisans_yonergesi",),
        "boost": 1.15,
    },
]


class RAGService:
    def __init__(self):
        self.model = SentenceTransformer("intfloat/multilingual-e5-base")
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.bm25 = None
        self.search_texts: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.cache_index_path = DATA_DIR / "processed_chunks_index.faiss"
        self.cache_meta_path = DATA_DIR / "processed_chunks_index_meta.json"

    def initialize(self, chunks: List[Dict[str, Any]]):
        self.chunks = self._prepare_chunks(chunks)
        self.search_texts = [self._build_search_text(chunk) for chunk in chunks]
        self.search_texts = [self._build_search_text(chunk) for chunk in self.chunks]
        self.tokenized_corpus = [self._tokenize(text) for text in self.search_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        cache_key = self._chunks_hash(self.chunks)
        if self._load_cached_index(cache_key):
            logger.info("%s parca ile RAG sistemi cache'den baslatildi.", len(self.chunks))
            return

        logger.info("%s parca icin embedding indeksi olusturuluyor...", len(self.chunks))
        texts_with_prefix = [f"passage: {text}" for text in self.search_texts]
        embeddings = self.model.encode(
            texts_with_prefix,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self._save_cached_index(cache_key)
        logger.info("RAG kaynaklari hazir.")

    def _prepare_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for chunk in chunks:
            content = chunk.get("content", "")
            metadata = self._enrich_metadata(content, dict(chunk.get("metadata", {})))
            if len(self._tokenize(content)) <= 520:
                prepared.append({"content": content, "metadata": metadata})
                continue

            clauses = self._clause_candidates(content)
            if len(clauses) < 4:
                prepared.append({"content": content, "metadata": metadata})
                continue

            part: List[str] = []
            part_tokens = 0
            part_index = 1
            for clause in clauses:
                clause_tokens = len(self._tokenize(clause))
                if part and part_tokens + clause_tokens > 460:
                    part_text = "\n".join(part)
                    part_metadata = self._enrich_metadata(part_text, dict(metadata))
                    part_metadata["madde"] = f"{metadata.get('madde', 'Parça')}-{part_index}"
                    prepared.append({"content": part_text, "metadata": part_metadata})
                    overlap = part[-2:] if len(part) > 2 else part[-1:]
                    part = list(overlap)
                    part_tokens = sum(len(self._tokenize(item)) for item in part)
                    part_index += 1
                part.append(clause)
                part_tokens += clause_tokens

            if part:
                part_text = "\n".join(part)
                part_metadata = dict(metadata)
                part_metadata["madde"] = f"{metadata.get('madde', 'Parça')}-{part_index}"
                prepared.append({"content": part_text, "metadata": self._enrich_metadata(part_text, part_metadata)})
        return prepared

    def _enrich_metadata(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        metadata["article"] = str(metadata.get("madde", metadata.get("article", "-")))
        metadata["title"] = metadata.get("title") or metadata.get("source", "")
        metadata["category"] = metadata.get("category") or self._guess_category(content, metadata)
        metadata["topic"] = metadata.get("topic") or self._topic_from_source(metadata.get("source", ""))
        metadata["intent_tags"] = self._intent_tags_for_content(content, metadata)
        metadata["keywords"] = sorted(set(self._tokenize(" ".join([
            metadata.get("source", ""),
            metadata.get("title", ""),
            metadata.get("category", ""),
            content[:700],
        ]))) - STOP_WORDS)[:30]
        return metadata

    def _topic_from_source(self, source: str) -> str:
        source_norm = self._normalize(source)
        if "erasmus" in source_norm:
            return "erasmus"
        if "ozel_ogrenci" in source_norm:
            return "ozel_ogrenci"
        if "cift_ana_dal" in source_norm:
            return "cift_anadal_yandal"
        if "ogrenci_kulupleri" in source_norm:
            return "kulup"
        if "muafiyet_ve_intibak" in source_norm:
            return "muafiyet"
        if "staj_yonergesi" in source_norm:
            return "staj"
        if "ulusal_staj" in source_norm:
            return "ulusal_staj"
        if "yaz_okulu" in source_norm:
            return "yaz_okulu"
        if "disiplin" in source_norm:
            return "disiplin"
        if "kayit" in source_norm and "dondur" in source_norm:
            return "kayit_dondurma"
        return "general"

    def _intent_tags_for_content(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        normalized = self._normalize(" ".join([content, metadata.get("title", ""), str(metadata.get("madde", ""))]))
        tags = set()
        if any(term in normalized for term in ("ifade eder", "tanim", "tanım", "amac", "kapsam", "kapsar")):
            tags.add("definition")
        if any(term in normalized for term in ("sart", "kosul", "gerekir", "zorunlu", "en az", "olmalidir", "olmalıdır")):
            tags.add("sartlar")
        if any(term in normalized for term in ("belge", "belgesi", "evrak", "form", "dilekce", "transkript")):
            tags.add("belgeler")
        if any(term in normalized for term in ("basvuru", "kayit", "ilan", "akademik takvim", "koordinatorluge")):
            tags.add("basvuru")
        if any(term in normalized for term in ("surec", "degerlendiril", "onay", "karara baglan")):
            tags.add("surec")
        if any(term in normalized for term in ("puan", "baraj", "taban", "olcut", "degerlendirme")):
            tags.add("puan_baraj")
        if any(term in normalized for term in ("sinav", "yabanci dil", "ingilizce", "dil sinav")):
            tags.add("sinav")
        return sorted(tags) or ["general"]

    def _guess_category(self, content: str, metadata: Dict[str, Any]) -> str:
        normalized = self._normalize(" ".join([content, metadata.get("source", ""), metadata.get("title", "")]))
        if any(term in normalized for term in ("gerekli belge", "evrak", "belgesi", "diploma", "transkript")):
            return "documents"
        if any(term in normalized for term in ("basvuru kosul", "sart", "kosul", "gerekir", "zorunlu")):
            return "conditions"
        if any(term in normalized for term in ("basvuru", "kayit", "akademik takvim", "ilan")):
            return "application"
        if any(term in normalized for term in ("sure", "gun", "yariyil", "donem")):
            return "duration"
        return "general"

    def _load_cached_index(self, cache_key: str) -> bool:
        if not self.cache_index_path.exists() or not self.cache_meta_path.exists():
            return False

        try:
            meta = json.loads(self.cache_meta_path.read_text(encoding="utf-8"))
            if meta.get("cache_key") != cache_key or meta.get("count") != len(self.chunks):
                return False

            self.index = faiss.read_index(str(self.cache_index_path))
            return True
        except Exception as exc:
            logger.warning("Cache index okunamadi, yeniden olusturulacak: %s", exc)
            return False

    def _save_cached_index(self, cache_key: str) -> None:
        if self.index is None:
            return

        faiss.write_index(self.index, str(self.cache_index_path))
        self.cache_meta_path.write_text(
            json.dumps({"cache_key": cache_key, "count": len(self.chunks)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _chunks_hash(self, chunks: List[Dict[str, Any]]) -> str:
        compact = json.dumps(chunks, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(compact.encode("utf-8")).hexdigest()

    def _build_search_text(self, chunk: Dict[str, Any]) -> str:
        metadata = chunk.get("metadata", {})
        parts = [
            metadata.get("source", ""),
            metadata.get("title", ""),
            f"Madde {metadata.get('madde', '')}",
            f"Konu {metadata.get('topic', '')}",
            f"Kategori {metadata.get('category', '')}",
            " ".join(metadata.get("intent_tags", [])),
            " ".join(metadata.get("keywords", [])),
            chunk.get("content", ""),
        ]
        return "\n".join(part for part in parts if part)

    def _ascii_fold(self, text: str) -> str:
        text = text.replace("ı", "i").replace("İ", "i")
        text = text.replace("ı", "i").replace("İ", "i")
        text = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in text if not unicodedata.combining(ch))

    def _normalize(self, text: str) -> str:
        text = text.lower().replace("İ", "i").replace("I", "ı")
        text = text.lower().replace("İ", "i").replace("I", "ı")
        text = self._ascii_fold(text)
        text = text.replace("ögrenci", "ogrenci").replace("yönerge", "yonerge")
        text = text.replace("ozelogrenci", "ozel ogrenci")
        text = text.replace("ozelogrencilik", "ozel ogrencilik")
        text = text.replace("ciftanadal", "cift anadal")
        text = text.replace("cift ana dal", "cift anadal")
        text = re.sub(r"\bcap\b", "cift anadal", text)
        text = text.replace("sartalri", "sartlari").replace("sartalr", "sartlar")
        text = text.replace("basvuru evraklari", "gerekli belgeler")
        text = text.replace("istenen belgeler", "gerekli belgeler")
        text = text.replace("yandal", "yan dal")
        text = text.replace("kulubu", "kulup").replace("kulübü", "kulup")
        text = text.replace("belgelr", "belgeler").replace("nelrdir", "nelerdir").replace("neelrdir", "nelerdir")
        text = text.replace("geerkli", "gerekli")
        return text

    def _tokenize(self, text: str) -> List[str]:
        normalized = self._normalize(text)
        tokens = re.findall(r"[a-z0-9çğıöşü]+", normalized)
        return [token for token in tokens if len(token) > 1 and token not in STOP_WORDS]

    def normalize_query(self, question: str) -> str:
        return self._normalize(question)

    def _query_tokens(self, question: str) -> set[str]:
        tokens = set(self._tokenize(question))
        normalized = self._normalize(question)
        if {"özel", "ozel"} & tokens:
            tokens.update({"ozel", "ogrenci"})
        if {"öğrenci", "ogrenci"} & tokens:
            tokens.add("ogrenci")
        if "staj" in tokens:
            tokens.add("staj")
        if "erasmus" in tokens:
            tokens.add("erasmus")
        if any(token.startswith("sinav") for token in tokens):
            tokens.update({"sinav", "yabanci", "dil"})
        if "ingilizce" in tokens or "english" in tokens or "yabanci dil" in normalized:
            tokens.update({"ingilizce", "yabanci", "dil", "sinav"})
        if any(token.startswith("puan") for token in tokens) or any(token.startswith("olcut") for token in tokens):
            tokens.update({"puan", "olcut", "degerlendirme"})
        if any(token.startswith("sart") or token.startswith("art") for token in tokens):
            tokens.update({"sart", "kosul", "basvuru"})
        if any(token.startswith("kosul") or token.startswith("osul") for token in tokens):
            tokens.update({"kosul", "sart", "basvuru"})
        if any(token.startswith("basvur") for token in tokens):
            tokens.update({"basvuru", "sart", "kosul"})
        if any(token.startswith("belge") for token in tokens) or any(token.startswith("evrak") for token in tokens):
            tokens.update({"belge", "belgeler", "evrak"})
        if any(token.startswith("gerek") for token in tokens):
            tokens.add("gerekli")
        if "bagil" in tokens or "degerlendirme" in tokens:
            tokens.update({"bagil", "degerlendirme", "not", "final", "sinav"})
        if "final" in tokens or "butunleme" in tokens:
            tokens.update({"final", "sinav", "not"})
        if {"kulup", "kulubu", "kulübü", "kul"} & tokens:
            tokens.update({"kulup", "ogrenci"})
        if {"cift", "anadal"} & tokens:
            tokens.update({"cift", "anadal", "basvuru", "sart"})
        return tokens

    def _source_label(self, source: str) -> str:
        label = source.replace(".pdf", "").replace("_pdf", "")
        label = label.replace("_", " ")
        label = label.replace("yonergesi", "yönergesi")
        label = label.replace("ogrenci", "öğrenci")
        label = label.replace("cift", "çift")
        label = label.replace("ozel", "özel")
        label = label.replace("degisim", "değişim")
        label = label.replace("programi", "programı")
        label = label.replace("kulupleri", "kulüpleri")
        label = label.replace("yuksek", "yüksek")
        label = label.replace("uluslararasi", "uluslararası")
        label = label.replace("yabancı uyruklu", "yabancı uyruklu")
        label = label.replace("ana dal", "ana dal")
        return label.title()

    def _clean_snippet(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\b\d{1,2}\.\d{1,2}\.\d{4}\s+SENATO\s+[\w/.-]+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ)\s+BÖLÜM\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^MADDE\s+\d+\s*[-–]\s*(\(\d+\)\s*)?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^[a-zçğıöşü]\)\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\(\d+\)\s*", "", text)
        text = re.sub(r"\s+\d+\s+(zorundadırlar|zorundadirlar|zorundadır|zorundadir)\b", r" \1", text, flags=re.IGNORECASE)
        text = text.replace("Madde", "MADDE")
        return text.strip(" -;\n\t")

    def _source_text(self, doc: Dict[str, Any]) -> str:
        metadata = doc.get("metadata", {})
        return f"{self._source_label(metadata.get('source', 'Bilinmeyen belge'))}, Madde {metadata.get('madde', '-')}"

    def _is_allowed_source(self, source: str, facets: Dict[str, Any]) -> bool:
        source_norm = self._normalize(source)
        if {"cift_anadal", "yandal"} & facets.get("entities", set()):
            return "ana_dal" in source_norm and "yan_dal" in source_norm
        allowed_sources = facets.get("allowed_sources") or set()
        return not allowed_sources or any(self._normalize(src) in source_norm for src in allowed_sources)

    def _intent_title(self, intent: str) -> str:
        titles = {
            "condition": "Başvuru şartları:",
            "duration": "Süre bilgisi:",
            "documents": "Gerekli belgeler:",
            "application": "Başvuru/kayıt süreci:",
            "process": "Süreç özetle şöyledir:",
            "definition": "Kısa tanım:",
            "yes_no": "Kısa cevap:",
            "country_list": "Kısa cevap:",
            "list": "Liste bilgisi:",
            "person": "Kısa cevap:",
        }
        return titles.get(intent, "Kısa cevap:")

    def _is_document_sentence(self, sentence: str, topic: str | None = None) -> bool:
        normalized = self._normalize(sentence)
        document_terms = (
            "belge",
            "belgesi",
            "evrak",
            "form",
            "dilekce",
            "tuzuk",
            "tüzük",
            "transkript",
            "not dokum",
            "sonuc belgesi",
            "yeterlik",
            "basvuru formu",
        )
        if not any(term in normalized for term in document_terms):
            return False

        noise_terms = (
            "yedek ogrenci",
            "yedek ogrencilerin",
            "kesin kayit",
            "kayit hakki",
            "kayit yaptir",
            "ders kayit",
            "ogrenim suresi",
            "azami sure",
            "komisyon",
            "koordinator",
            "danisman atan",
            "degerlendiril",
            "ilan edilir",
            "basvurular",
        )
        if any(term in normalized for term in noise_terms):
            allow_document_phrase = any(
                term in normalized
                for term in (
                    "merkezi yerlestirme sinav sonuc belgesi",
                    "yabanci dil yeterlik sinav sonuc belgesi",
                    "basari sarti belgesi",
                    "not dokum belgesi",
                    "basvuru dilekce",
                    "disiplin cezasi almadiklarina dair belgeler",
                    "kulup tuzugu",
                    "ogrenci kulubu tuzugu",
                    "kulup basvuru formu",
                )
            )
            if not allow_document_phrase:
                return False

        if topic == "kulup" and "en az 15" in normalized:
            return False
        return True

    def _extract_document_items(self, docs: List[Dict[str, Any]], facets: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        topic = facets.get("primary_topic")
        items: List[Tuple[str, Dict[str, Any]]] = []
        seen = set()

        def add(label: str, doc: Dict[str, Any]) -> None:
            key = self._normalize(label)
            if key in seen:
                return
            seen.add(key)
            items.append((label, doc))

        for doc in docs:
            content = doc.get("content", "")
            normalized_content = self._normalize(content)
            if topic in {"cift_anadal", "yandal"}:
                patterns = [
                    ("Not döküm belgesi", "not dokum belgesi"),
                    ("Başarı şartı belgesi", "basari sarti belgesi"),
                    ("Merkezi yerleştirme sınav sonuç belgesi", "merkezi yerlestirme sinav sonuc belgesi"),
                    ("Yabancı dil yeterlik sınav sonuç belgesi (gerekli ise)", "yabanci dil yeterlik sinav sonuc belgesi"),
                ]
                if topic == "yandal":
                    patterns = [item for item in patterns if item[1] != "basari sarti belgesi"]
                for label, marker in patterns:
                    if marker in normalized_content:
                        add(label, doc)
                continue

            if topic == "kulup":
                patterns = [
                    ("Başvuru dilekçeleri ve ekleri", "basvuru dilekce"),
                    ("Öğrenci Kulübü Tüzüğü", "ogrenci kulup tuzugu"),
                    ("Yönetim kurulunda görev alacak öğrencilerin disiplin cezası almadıklarına dair belgeleri", "disiplin cezasi almadiklarina dair belgeler"),
                    ("Kulüp Başvuru Formu", "kulup basvuru formu"),
                ]
                for label, marker in patterns:
                    if marker in normalized_content:
                        add(label, doc)
                continue

            for sentence in self._clause_candidates(content):
                if self._is_document_sentence(sentence, topic):
                    add(self._clean_snippet(sentence), doc)
        return items

    def _document_answer(self, question: str, docs: List[Dict[str, Any]], facets: Dict[str, Any]) -> str | None:
        if facets.get("primary_topic") == "erasmus":
            erasmus_docs = [
                doc for doc in docs[:6]
                if "erasmus" in self._normalize(doc.get("metadata", {}).get("source", ""))
            ]
            mentioned = []
            seen = set()
            patterns = [
                ("Öğrenim anlaşması", ("ogrenim anlasmasi", "learning agreement")),
                ("Not döküm belgesi", ("not cizelgesi", "transkript", "not dokum")),
                ("Ders tanınırlık/denklik tablosu", ("ders taninirlik", "recognition sheet", "denklik tablosu")),
                ("Başvuru formu ve ekleri", ("basvuru formu",)),
            ]
            for doc in erasmus_docs:
                content = self._normalize(doc.get("content", ""))
                for label, markers in patterns:
                    if label in seen:
                        continue
                    if any(marker in content for marker in markers):
                        seen.add(label)
                        mentioned.append((label, doc))
            sources = []
            for _, doc in mentioned[:3] or erasmus_docs[:2]:
                source_text = self._source_text(doc)
                if source_text not in sources:
                    sources.append(source_text)
            source_suffix = "\n\nKaynak: " + "; ".join(sources) + "." if sources else ""
            if mentioned:
                bullets = "\n".join(f"- {label}" for label, _ in mentioned[:4])
                return (
                    "Yönergede Erasmus başvurusu için doğrudan ve tam bir belge listesi bulamadım.\n\n"
                    "Metinde geçen belge/form ifadeleri şunlardır:\n\n"
                    f"{bullets}\n\n"
                    "Güncel başvuru belgeleri için Erasmus Koordinatörlüğünün ilgili dönem ilanını kontrol ediniz."
                    + source_suffix
                )
            return (
                "Yönergede Erasmus başvurusu için doğrudan ve tam bir belge listesi bulamadım. "
                "Güncel başvuru belgeleri için Erasmus Koordinatörlüğünün ilgili dönem ilanını kontrol ediniz."
                + source_suffix
            )

        items = self._extract_document_items(docs[:6], facets)
        if not items:
            return None
        bullets = "\n".join(f"- {item}" for item, _ in items[:6])
        sources = []
        for _, doc in items:
            source_text = self._source_text(doc)
            if source_text not in sources:
                sources.append(source_text)
            if len(sources) >= 3:
                break
        return f"Gerekli belgeler:\n\n{bullets}\n\nKaynak: " + "; ".join(sources) + "."

    def _is_direct_definition_sentence(self, sentence: str, question: str) -> bool:
        normalized = self._normalize(sentence)
        qnorm = self._normalize(question)
        if any(term in normalized for term in ("bu yonergenin amaci", "bu yonerge", "kapsar", "dayanak", "amac")):
            return False
        if "ozel ogrenci" in qnorm:
            label = self._normalize(sentence.split(":", 1)[0]) if ":" in sentence[:120] else normalized[:80]
            return "ozel ogrenci" in label and "komisyon" not in label and ("ifade eder" in normalized or ":" in sentence[:120])
        if "tezsiz yuksek lisans" in qnorm:
            label = self._normalize(sentence.split(":", 1)[0]) if ":" in sentence[:120] else normalized[:80]
            return "tezsiz yuksek lisans" in label and ("ifade eder" in normalized or ":" in sentence[:120])
        if "cift anadal" in qnorm or "cift ana dal" in qnorm:
            label = self._normalize(sentence.split(":", 1)[0]) if ":" in sentence[:120] else normalized[:100]
            return "cift ana dal" in label or "cift anadal" in label
        if "erasmus" in qnorm:
            label = self._normalize(sentence.split(":", 1)[0]) if ":" in sentence[:120] else normalized[:100]
            return "erasmus programi" in label or "erasmus+ programi" in label
        if "ifade eder" not in normalized:
            return False
        return True

    def _definition_no_direct_answer(self, question: str, docs: List[Dict[str, Any]]) -> str | None:
        facets = self.extract_query_facets(question)
        if facets.get("primary_intent") != "definition":
            return None
        if facets.get("primary_topic") not in {"tezsiz_yuksek_lisans", "cift_anadal", "yandal"}:
            return None
        topic = facets.get("primary_topic")
        topic_docs = [
            doc for doc in docs[:4]
            if (
                topic == "tezsiz_yuksek_lisans"
                and "tezsiz_yuksek_lisans" in self._normalize(doc.get("metadata", {}).get("source", ""))
            )
            or (
                topic in {"cift_anadal", "yandal"}
                and "cift_ana_dal" in self._normalize(doc.get("metadata", {}).get("source", ""))
            )
        ]
        sources = [self._source_text(doc) for doc in topic_docs[:2]]
        source_suffix = "\n\nKaynak: " + "; ".join(dict.fromkeys(sources)) + "." if sources else ""
        if topic in {"cift_anadal", "yandal"}:
            return (
                "Yönergede bu kavram için doğrudan bir tanım bulunamadı. "
                "Ancak ilgili yönergede başvuru şartları ve uygulama esasları yer almaktadır."
                + source_suffix
            )
        return (
            "Yönergede tezsiz yüksek lisans için doğrudan bir tanım cümlesi bulamadım. "
            "İlgili yönergede programın amacı, kapsamı ve işleyişine ilişkin hükümler yer almaktadır."
            + source_suffix
        )

    def _yes_no_answer(self, question: str, docs: List[Dict[str, Any]], facets: Dict[str, Any]) -> str | None:
        qnorm = self._normalize(question)
        if {"cift_anadal", "yandal"} & facets["entities"] and any(term in qnorm for term in ("ayni bolum", "ayni isim", "aynı bölüm", "aynı isim")):
            candidate_docs = self._direct_candidate_chunks(question, facets) + docs[:6]
            for doc in candidate_docs:
                if "cift_ana_dal" not in self._normalize(doc.get("metadata", {}).get("source", "")):
                    continue
                for sentence in self._clause_candidates(doc.get("content", "")):
                    snorm = self._normalize(sentence)
                    if "ayni isimli bolumler arasinda" in snorm and "basvurusu yapilamaz" in snorm:
                        return (
                            "Hayır. Aynı isimli bölümler arasında çift anadal başvurusu yapılamaz.\n\n"
                            f"Kaynak: {self._source_text(doc)}."
                        )
        return None

    def _country_list_answer(self, question: str, docs: List[Dict[str, Any]], facets: Dict[str, Any]) -> str | None:
        if facets.get("primary_topic") != "erasmus" or facets.get("primary_intent") != "ulke_liste":
            return None
        erasmus_docs = [
            doc for doc in docs[:4]
            if "erasmus" in self._normalize(doc.get("metadata", {}).get("source", ""))
        ]
        sources = [self._source_text(doc) for doc in erasmus_docs[:2]]
        source_suffix = "\n\nKaynak: " + "; ".join(dict.fromkeys(sources)) + "." if sources else ""
        return (
            "Yönergede Erasmus yapılabilecek ülkelerin veya anlaşmalı üniversitelerin doğrudan listesi bulunmamaktadır. "
            "Erasmus yapılabilecek yerler güncel anlaşmalara ve Erasmus Koordinatörlüğünün ilanlarına göre değişir. "
            "Güncel liste için Erasmus Koordinatörlüğü sayfasını kontrol ediniz."
            + source_suffix
        )

    def _sentence_candidates(self, content: str) -> List[str]:
        content = content.replace("\r", "\n")
        parts = []
        for block in re.split(r"\n+|(?<=[.!?])\s+|(?=\(\d+\)\s)", content):
            block = self._clean_snippet(block)
            if len(block) < 18:
                continue
            first_word = block.split(" ", 1)[0].strip(" ,.;:").lower()
            if first_word in {"uygun", "yüksek", "yuksek", "öğretim", "ogretim", "bir", "enstitünün", "enstitunun"}:
                continue
            if re.fullmatch(r"\d+", block):
                continue
            if block.count("(") > block.count(")") or block.count(")") > block.count("("):
                continue
            if not re.search(r"[.!?]$|alamaz$|gerekir$|olmalıdır$|edilir$|yapılır$|başvurabilir$", block, re.IGNORECASE):
                continue
            parts.append(block)
        return parts

    def _clause_candidates(self, content: str) -> List[str]:
        text = content.replace("\r", "\n")
        text = re.sub(r"\n(?=\s*(MADDE|\(\d+\)|[a-zçğıöşü]\)))", " § ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*\n\s*", " ", text)
        text = re.sub(r"\s+", " ", text)

        parts = []
        for block in re.split(r"\s*§\s*|(?=\bmadde\s+\d+\s*[–-])|(?=\(\d+\)\s)|(?=\b[a-zçğıöşü]\)\s)|(?<=[.!?])\s+", text, flags=re.IGNORECASE):
            block = self._clean_snippet(block)
            if len(block) < 14:
                continue
            if self._normalize(block) in {
                "basvuru kosullari",
                "basvuru sureci",
                "basvuru sonuclarinin ilani ve kesin kayit",
                "yabanci uyruklu statusu",
            }:
                continue
            if re.search(r"^(amaç|kapsam|dayanak|tanımlar|yürürlük|yürütme)$", block, re.IGNORECASE):
                continue
            first_word = block.split(" ", 1)[0].strip(" ,.;:").lower()
            if first_word in {"uygun", "yüksek", "yuksek", "öğretim", "ogretim", "bir", "enstitünün", "enstitunun"}:
                continue
            parts.append(block)
        return parts

    def _sentence_score(self, sentence: str, query_tokens: set[str]) -> float:
        normalized = self._normalize(sentence)
        sentence_tokens = set(self._tokenize(sentence))
        score = len(sentence_tokens.intersection(query_tokens)) * 2.0
        if any(token in normalized for token in ("basvuru", "kosul", "sart", "gerekir", "zorunlu", "belge")):
            score += 1.2
        if any(token in normalized for token in ("sure", "gun", "ay", "yariyil", "donem", "akts", "ortalama")):
            score += 0.8
        if normalized.startswith(("madde", "a)", "b)", "c)", "ç)", "(1)", "(2)", "(3)")):
            score += 0.4
        return score

    def _question_intent(self, question: str) -> str:
        normalized = self._normalize(question)
        tokens = set(self._tokenize(question))

        if any(term in normalized for term in ("rektor kim", "rektör kim", "kimdir")) and not any(
            term in normalized for term in ("ozel ogrenci", "tezsiz", "erasmus", "cift anadal", "kulup")
        ):
            return "person"
        if any(term in normalized for term in ("hangi ulke", "hangi ülke", "ulkelere", "ülkelere", "hangi universite", "hangi üniversite", "anlasmali universite", "anlaşmalı üniversite")):
            return "country_list"
        if any(term in normalized for term in ("yapilabilir mi", "yapılır mı", "olur mu", "mümkün mü", "mumkun mu", "var mi", "var mı")):
            return "yes_no"
        if any(term in normalized for term in ("nasil yapilir", "nasıl yapılır", "nasil basvur", "nasıl başvur", "surec nasil", "süreç nasıl")):
            return "process"
        if tokens.intersection({"belge", "belgeler", "belgeleri", "evrak", "dokuman", "dokumanlar", "form"}):
            return "documents"
        if any(term in normalized for term in ("liste", "listesi", "nelerdir")) and tokens.intersection({"ulke", "ülke", "universite", "üniversite"}):
            return "list"
        if "ne zaman" in normalized or "hangi tari" in normalized:
            return "application"
        if any(term in normalized for term in ("nasil basvuru", "nasil kayit", "basvuru nasil", "kayit nasil")):
            return "application"
        condition_terms = {"sart", "sartlari", "şart", "kosul", "kosullari", "koşul", "ortalama", "gno", "puan", "sinav"}
        if tokens.intersection({"basvuru", "basvurular", "kayit", "kayıt"}) and not tokens.intersection(condition_terms):
            return "application"
        if any(term in normalized for term in ("kac gun", "kaç gun", "kac is gunu", "sure", "süre", "ne kadar")):
            return "duration"
        if tokens.intersection(condition_terms | {"sarti", "şartı"}):
            return "condition"
        if any(term in normalized for term in ("nedir", "ne demek", "kimdir", "tanimi", "tanımı")):
            return "definition"
        return "answer"

    def extract_query_facets(self, question: str) -> Dict[str, Any]:
        normalized = self._normalize(question)
        tokens = self._query_tokens(question)
        raw_tokens = set(self._tokenize(question))
        entities = set()
        intents = set()

        if "erasmus" in tokens:
            entities.add("erasmus")
        if "tezsiz" in tokens and {"yuksek", "lisans"}.issubset(tokens):
            entities.add("tezsiz_yuksek_lisans")
        if {"ozel", "ogrenci"}.issubset(tokens):
            entities.add("ozel_ogrenci")
        if {"cift", "anadal"}.issubset(tokens) or "cap" in tokens or "çap" in question.lower():
            entities.add("cift_anadal")
        if "yandal" in tokens or "yan dal" in normalized:
            entities.add("yandal")
        if "kulup" in tokens or "topluluk" in tokens:
            entities.add("kulup")
        if "muafiyet" in tokens or "intibak" in tokens:
            entities.add("muafiyet")
        if "kayit dondurma" in normalized or "kayit dondur" in normalized:
            entities.add("kayit_dondurma")
        if "staj" in tokens:
            entities.add("staj")
        if "disiplin" in tokens:
            entities.add("disiplin")

        if any(term in normalized for term in ("nedir", "ne demek", "kimdir", "tanimi", "tanım")):
            intents.add("definition")
        if any(term in normalized for term in ("rektor kim", "rektör kim")):
            intents.add("kisi_bilgisi")
        if any(term in normalized for term in ("yapilabilir mi", "yapılır mı", "olur mu", "mumkun mu", "mümkün mü", "var mi", "var mı")):
            intents.add("evet_hayir")
        if any(term in normalized for term in ("hangi ulke", "hangi ülke", "ulkelere", "ülkelere", "hangi universite", "hangi üniversite", "anlasmali universite", "anlaşmalı üniversite")):
            intents.add("ulke_liste")
        if "liste" in raw_tokens or "listesi" in raw_tokens:
            intents.add("liste")
        if raw_tokens.intersection({"sart", "sarti", "sartlari", "artlar", "kosul", "kosullari", "zorunlu"}):
            intents.add("sart")
        if raw_tokens.intersection({"belge", "belgeler", "belgeleri", "evrak", "form", "dokuman"}):
            intents.add("belge")
        if raw_tokens.intersection({"basvuru", "basvurular", "kayit", "ilan"}):
            intents.add("basvuru")
        if raw_tokens.intersection({"surec", "nasil", "nasıl", "yapilir", "yapılır", "yapilır", "yap"}):
            intents.add("surec")
        if raw_tokens.intersection({"sinav", "ingilizce", "yabanci", "dil"}):
            intents.add("sinav")
        if {"dil", "sinav"}.issubset(raw_tokens) or {"ingilizce", "sinav"}.issubset(raw_tokens):
            intents.add("dil_sinavi")
        if raw_tokens.intersection({"puan", "baraj", "taban", "olcut", "ölçüt", "degerlendirme"}):
            intents.add("puan_baraj")
        if any(term in normalized for term in ("her bolum", "her bölüm", "tum bolum", "tüm bölüm", "uygun mu", "yapilir mi", "yapılır mı")):
            intents.add("uygunluk")

        allowed_sources: set[str] = set()
        if "erasmus" in entities:
            allowed_sources.add("erasmus_degisim_programi_yonergesi")
        if "ozel_ogrenci" in entities:
            allowed_sources.add("ozel_ogrenci_yonergesi")
        if "kulup" in entities:
            allowed_sources.add("ogrenci_kulupleri_yonergesi")
        if {"cift_anadal", "yandal"} & entities:
            allowed_sources.add("cift_ana_dal_ve_yan_dal_programi_yonergesi")
            allowed_sources.add("çift_ana_dal_ve_yan_dal_programı_yonergesi")
        if "muafiyet" in entities:
            allowed_sources.add("muafiyet_ve_intibak_yonergesi")
        if "staj" in entities and "ulusal" not in tokens:
            allowed_sources.add("staj_yonergesi")
        if "tezsiz_yuksek_lisans" in entities:
            allowed_sources.add("tezsiz_yuksek_lisans_yonergesi")

        primary_intent = self._primary_intent(intents)
        primary_topic = self._primary_topic(entities)
        return {
            "entities": entities,
            "intents": intents,
            "allowed_sources": allowed_sources,
            "primary_intent": primary_intent,
            "primary_topic": primary_topic,
        }

    def _primary_intent(self, intents: set[str]) -> str:
        for intent in ("kisi_bilgisi", "ulke_liste", "evet_hayir", "surec", "belge", "puan_baraj", "dil_sinavi", "sinav", "sart", "definition", "basvuru", "uygunluk", "liste"):
            if intent in intents:
                return intent
        return "answer"

    def _primary_topic(self, entities: set[str]) -> str | None:
        for topic in ("erasmus", "ozel_ogrenci", "cift_anadal", "yandal", "kulup", "muafiyet", "kayit_dondurma", "staj", "tezsiz_yuksek_lisans", "disiplin"):
            if topic in entities:
                return topic
        return None

    def _topic_phrase(self, topic: str | None) -> str:
        phrases = {
            "kulup": "kulüp açmak için",
            "erasmus": "erasmus için",
            "ozel_ogrenci": "özel öğrenci",
            "cift_anadal": "çift anadal",
            "yandal": "yan dal",
            "muafiyet": "ders muafiyeti",
            "kayit_dondurma": "kayıt dondurma",
            "staj": "staj",
            "tezsiz_yuksek_lisans": "tezsiz yüksek lisans",
            "disiplin": "disiplin",
        }
        return phrases.get(topic or "", topic or "")

    def _intent_label(self, facets: Dict[str, Any]) -> str:
        intent = facets.get("primary_intent") or "answer"
        aliases = {
            "belge": "belgeler",
            "sart": "şartlar",
            "basvuru": "başvuru",
            "surec": "süreç",
            "dil_sinavi": "sınav",
            "sinav": "sınav",
            "puan_baraj": "puan/baraj",
            "definition": "definition",
            "evet_hayir": "evet/hayır",
            "ulke_liste": "ülke/anlaşmalı kurum listesi",
            "liste": "liste",
            "kisi_bilgisi": "kişi bilgisi",
        }
        return aliases.get(intent, intent)

    def _conversation_state(self, history: List[Dict[str, str]] | None) -> Dict[str, Any]:
        state = {
            "last_topic": None,
            "last_intent": None,
            "last_source": None,
            "recent_topics": [],
        }
        if not history:
            return state

        recent_topics = []
        for item in history[-10:]:
            content = item.get("content", "")
            role = item.get("role")
            if role == "user":
                facets = self.extract_query_facets(content)
                topic = facets.get("primary_topic")
                if topic:
                    state["last_topic"] = topic
                    state["last_intent"] = self._intent_label(facets)
                    if topic not in recent_topics:
                        recent_topics.append(topic)
            elif role == "assistant":
                source_match = re.search(r"Kaynak:\s*([^.\n]+)", content)
                if source_match:
                    state["last_source"] = source_match.group(1).strip()

        state["recent_topics"] = recent_topics[-5:]
        return state

    def _is_generic_followup(self, question: str, facets: Dict[str, Any]) -> bool:
        if facets.get("primary_topic"):
            return False
        normalized = self._normalize(question)
        tokens = self._query_tokens(question)
        base_tokens = set(self._tokenize(question))
        if len(base_tokens) > 7:
            return False
        generic_patterns = (
            "belge",
            "belgeler",
            "evrak",
            "sart",
            "şart",
            "kosul",
            "basvuru",
            "başvuru",
            "surec",
            "süreç",
            "nasil",
            "nasıl",
            "kac puan",
            "kaç puan",
            "baraj",
            "puan",
            "sinav",
            "sınav",
            "danisman",
            "danışman",
        )
        return any(pattern in normalized for pattern in generic_patterns) or bool(
            facets["intents"].intersection({"belge", "sart", "basvuru", "surec", "sinav", "dil_sinavi", "puan_baraj"})
        )

    def _expand_followup_query(self, question: str, state: Dict[str, Any]) -> str:
        topic = state.get("last_topic")
        if not topic:
            recent = state.get("recent_topics") or []
            topic = recent[-1] if recent else None
        topic_phrase = self._topic_phrase(topic)
        if not topic_phrase:
            return question
        normalized = self._normalize(question)
        if self._normalize(topic_phrase) in normalized:
            return question
        return f"{topic_phrase} {question}"

    def _intent_profile(self, intent: str) -> Dict[str, Any]:
        profiles = {
            "documents": {
                "required": {
                    "belge",
                    "belgeler",
                    "evrak",
                    "form",
                    "dilekce",
                    "dilekçe",
                    "nufus",
                    "kimlik",
                    "fotokopi",
                    "fotokopisi",
                    "nusha",
                    "nüsha",
                    "transkript",
                    "transkripti",
                    "diploma",
                    "diplomasi",
                    "diploması",
                    "denklik",
                    "pasaport",
                    "vize",
                    "foto",
                    "fotograf",
                    "fotoğraf",
                    "dokum",
                    "döküm",
                    "belgesi",
                    "sonuc",
                    "sonuç",
                    "yeterlik",
                },
                "preferred": {"basvuru", "kayit", "kayıt", "gerekli", "istenir"},
                "max_items": 7,
            },
            "application": {
                "required": {
                    "basvuru",
                    "basvurular",
                    "basvurusu",
                    "kayit",
                    "kayıt",
                    "kariyer",
                    "kapisi",
                    "kapısı",
                    "edevlet",
                    "e-devlet",
                    "akademik",
                    "takvim",
                    "ilan",
                    "ucret",
                    "ücret",
                },
                "preferred": {"yapilir", "yapılır", "alınır", "alinir", "odeme", "ödeme", "onay"},
                "max_items": 4,
            },
            "process": {
                "required": {
                    "basvuru",
                    "basvurular",
                    "kayit",
                    "degerlendirme",
                    "komisyon",
                    "belge",
                    "sart",
                    "kosul",
                    "akademik",
                    "takvim",
                    "ilan",
                },
                "preferred": {"yapilir", "yapılır", "basvurur", "kayit", "degerlendirilir", "onay"},
                "max_items": 6,
            },
            "duration": {
                "required": {"sure", "süre", "gun", "gün", "is", "iş", "saat", "hafta", "ay", "yariyil", "yarıyıl"},
                "preferred": {"en", "az", "40", "30", "20", "zorunludur", "sayilir", "sayılır"},
                "max_items": 3,
            },
            "condition": {
                "required": {
                    "aday",
                    "adaylar",
                    "basvuru",
                    "başvuru",
                    "basvurular",
                    "basvurabilmeleri",
                    "basvurularinin",
                    "mezun",
                    "sinif",
                    "sınıf",
                    "uyruklu",
                    "sart",
                    "şart",
                    "sartlari",
                    "kosul",
                    "koşul",
                    "kosullari",
                    "gerekir",
                    "gerekmektedir",
                    "zorunlu",
                    "olmalidir",
                    "olmalıdır",
                    "en",
                    "az",
                    "ortalama",
                    "ortalamasi",
                    "ortalaması",
                    "ortalamasina",
                    "ortalamasına",
                    "puan",
                    "sinav",
                    "sınav",
                    "basvurabilmesi",
                    "kayitli",
                    "kayıtlı",
                },
                "preferred": {"basvuru", "aday", "ogrenci", "öğrenci", "lisans", "onlisans"},
                "max_items": 5,
            },
            "definition": {
                "required": {"ifade", "tanim", "tanım"},
                "preferred": {"nedir", "ne", "demek"},
                "max_items": 3,
            },
            "yes_no": {
                "required": {"yapilamaz", "yapılamaz", "yapilabilir", "yapılabilir", "zorunlu", "gerekir", "degildir", "değildir", "olmaz"},
                "preferred": {"ayni", "aynı", "isimli", "bolum", "bölüm"},
                "max_items": 2,
            },
            "country_list": {
                "required": {"ulke", "ülke", "universite", "üniversite", "anlasma", "anlaşma", "liste"},
                "preferred": {"erasmus", "ikili", "kurum"},
                "max_items": 4,
            },
            "list": {
                "required": {"liste", "listesi", "ulke", "ülke", "universite", "üniversite"},
                "preferred": set(),
                "max_items": 4,
            },
            "person": {
                "required": {"rektor", "rektör", "kim"},
                "preferred": set(),
                "max_items": 1,
            },
            "answer": {
                "required": set(),
                "preferred": set(),
                "max_items": 4,
            },
        }
        return profiles.get(intent, profiles["answer"])

    def _intent_match_score(self, sentence: str, intent: str) -> float:
        profile = self._intent_profile(intent)
        normalized = self._normalize(sentence)
        tokens = set(self._tokenize(sentence))
        required_hits = len(tokens.intersection(profile["required"]))
        preferred_hits = len(tokens.intersection(profile["preferred"]))

        if intent == "documents" and re.search(r"\b[a-zçğıöşü]\)\s", sentence, re.IGNORECASE):
            preferred_hits += 1
        if intent == "duration" and re.search(r"\b\d+\s*(is\s+gunu|gun|gün|saat|hafta|ay|yariyil|yarıyıl)\b", normalized):
            required_hits += 2
        if intent == "definition" and re.search(r"\b(ifade eder|tan[ıi]m)\b", normalized):
            required_hits += 1

        return required_hits * 2.0 + preferred_hits * 0.8

    def _matches_intent(self, sentence: str, intent: str, question: str) -> bool:
        if intent == "answer":
            return True

        normalized = self._normalize(sentence)
        qnorm = self._normalize(question)
        tokens = set(self._tokenize(sentence))
        profile = self._intent_profile(intent)

        if not tokens.intersection(profile["required"]):
            return False

        if intent == "documents":
            facets = self.extract_query_facets(question)
            topic = facets.get("primary_topic")
            if not self._is_document_sentence(sentence, topic):
                return False
            if "gerekli belgeler" in normalized or "basvuru icin gerekli belgeler" in normalized:
                return False
            if re.match(r"^[a-zçğıöşü]\)\s", sentence, re.IGNORECASE):
                return True
            if "vatandasligindan cikma" in normalized or "vatandaşlığından çıkma" in normalized:
                return False
            document_terms = tokens.intersection(profile["required"])
            return bool(document_terms.intersection({"belge", "belgeler", "belgesi", "evrak", "form", "dilekce", "dilekçe", "transkript", "transkripti", "diploma", "diplomasi", "diploması", "denklik", "pasaport", "vize", "foto", "fotograf", "fotoğraf", "dokum", "döküm", "sonuc", "sonuç", "yeterlik"}))

        if intent == "duration":
            return bool(re.search(r"\b\d+\s*(is\s+gunu|gun|gün|saat|hafta|ay|yariyil|yarıyıl)\b", normalized)) or "is gunu" in normalized

        if intent == "definition":
            if any(term in normalized for term in ("basvuru", "kayit", "kariyer kapisi", "e-devlet")):
                return False
            return self._is_direct_definition_sentence(sentence, question)

        if intent == "yes_no":
            if any(term in qnorm for term in ("ayni bolum", "ayni isim", "aynı bölüm", "aynı isim")):
                return "ayni isimli" in normalized and "yapilamaz" in normalized
            return any(term in normalized for term in ("yapilamaz", "yapilabilir", "zorunlu", "gerekir", "degildir", "olmaz"))

        if intent in {"country_list", "list"}:
            if "erasmus" in qnorm and any(term in qnorm for term in ("ulke", "ülke", "universite", "üniversite")):
                return False
            return any(term in normalized for term in ("listesi", "asagida", "aşağıda", "ulke", "ülke", "universite", "üniversite"))

        if intent == "application":
            return any(term in normalized for term in ("basvuru", "kayit", "kariyer kapisi", "e-devlet", "akademik takvim", "ilan", "ucret"))

        if intent == "process":
            return any(term in normalized for term in ("basvuru", "kayit", "degerlendiril", "komisyon", "belge", "sart", "kosul", "akademik takvim", "ilan", "onay"))

        if intent == "condition":
            if any(term in normalized for term in ("ulusal sinavlar", "taban puanlari", "gecerlilik sureleri", "ülke sınav")):
                return False
            if "kapsar" in normalized and not any(term in normalized for term in ("basvuru kosul", "kabul", "aday")):
                return False
            return any(term in normalized for term in ("gerekir", "gerekmektedir", "zorunlu", "olmalidir", "olmalıdır", "sart", "kosul", "basvuru", "aday", "kabul", "en az", "ortalama", "puan", "sinav"))

        return True

    def _preferred_source(self, question: str, docs: List[Dict[str, Any]]) -> str | None:
        rules = self._matching_rules(question)
        if not rules:
            return docs[0].get("metadata", {}).get("source") if docs else None

        for rule in rules:
            for doc in docs:
                source = self._normalize(doc.get("metadata", {}).get("source", ""))
                if any(self._normalize(src) in source for src in rule["sources"]):
                    return doc.get("metadata", {}).get("source")
        return docs[0].get("metadata", {}).get("source") if docs else None

    def _no_direct_answer(self, question: str, docs: List[Dict[str, Any]], intent: str) -> str | None:
        if intent != "documents" or not docs:
            return None

        primary_source = self._preferred_source(question, docs)
        relevant_docs = [
            doc
            for doc in docs[:4]
            if not primary_source or doc.get("metadata", {}).get("source") == primary_source
        ]
        if "erasmus" in self._normalize(question):
            document_sentences = []
            for doc in relevant_docs:
                for sentence in self._clause_candidates(doc.get("content", "")):
                    normalized_sentence = self._normalize(sentence)
                    if "basvuru formu" in normalized_sentence and ("ekleri" in normalized_sentence or "gerekli belge" in normalized_sentence):
                        document_sentences.append((sentence, doc))
            if document_sentences:
                sentence, doc = document_sentences[0]
                metadata = doc.get("metadata", {})
                source_text = f"{self._source_label(metadata.get('source', 'Bilinmeyen belge'))}, Madde {metadata.get('madde', '-')}"
                return (
                    "İlgili yönergeye göre kısa cevap:\n\n"
                    f"- Erasmus başvurusu, Koordinatörlüğün web sitesinde yayımlanan başvuru formu ve ekleriyle yapılır.\n"
                    "- Yönergede başvuru belgeleri tek tek sabit bir liste olarak sayılmıyor; ilgili dönem ilanındaki belge listesi esas alınmalıdır.\n\n"
                    f"Kaynak: {source_text}."
                )

        has_document_clause = any(
            self._matches_intent(sentence, "documents", question)
            for doc in relevant_docs
            for sentence in self._clause_candidates(doc.get("content", ""))
        )
        if has_document_clause:
            return None

        sources = []
        for doc in relevant_docs[:2]:
            metadata = doc.get("metadata", {})
            source_text = f"{self._source_label(metadata.get('source', 'Bilinmeyen belge'))}, Madde {metadata.get('madde', '-')}"
            if source_text not in sources:
                sources.append(source_text)

        if not sources:
            return None

        return (
            "Bu konuda yönergede gerekli belgeler tek tek liste halinde açıkça sayılmamış.\n\n"
            "RAG aramasında ilgili fişler başvuru/kayıt işlemlerini gösteriyor; ancak belge adı olarak açık bir liste vermiyor. "
            "Bu yüzden belge listesi için ilgili dönem ilanı veya birim duyurusu esas alınmalıdır.\n\n"
            "Kaynak: " + "; ".join(sources) + "."
        )

    def _generic_fast_answer(self, question: str, docs: List[Dict[str, Any]]) -> str | None:
        if not docs:
            return None

        query_tokens = self._query_tokens(question)
        intent = self._question_intent(question)
        facets = self.extract_query_facets(question)
        if facets["primary_intent"] == "ulke_liste":
            country_answer = self._country_list_answer(question, docs, facets)
            if country_answer:
                return country_answer
        if intent == "yes_no" or facets["primary_intent"] == "evet_hayir":
            yes_no_answer = self._yes_no_answer(question, docs, facets)
            if yes_no_answer:
                return yes_no_answer
        if intent == "documents" or facets["primary_intent"] == "belge":
            document_answer = self._document_answer(question, docs, facets)
            if document_answer:
                return document_answer

        no_direct = self._no_direct_answer(question, docs, intent)
        if no_direct:
            return no_direct
        asks_definition = intent == "definition"
        primary_source = self._preferred_source(question, docs)
        qnorm = self._normalize(question)

        if facets["primary_intent"] == "surec" and {"cift_anadal", "yandal"} & facets["entities"]:
            process_docs = self._direct_candidate_chunks(question, facets)
            by_article = {str(doc.get("metadata", {}).get("madde", "")): doc for doc in process_docs}
            selected_sources = []
            for key in ("7", "8", "9", "10"):
                doc = by_article.get(key)
                if not doc:
                    continue
                metadata = doc.get("metadata", {})
                source_text = f"{self._source_label(metadata.get('source', 'Bilinmeyen belge'))}, Madde {metadata.get('madde', '-')}"
                if source_text not in selected_sources:
                    selected_sources.append(source_text)
            if selected_sources:
                return (
                    "Çift anadal yapmak için süreç özetle şöyledir:\n\n"
                    "- Önce başvuru şartlarını sağlamanız gerekir; başvurular ilan edilen tarihler arasında ilgili Fakülte/Yüksekokula yapılır.\n"
                    "- Başvurular ilgili bölüm başkanlığınca oluşturulan komisyon tarafından değerlendirilir.\n"
                    "- Kayıt hakkı kazanan öğrenciler akademik takvime göre ilgili Fakülte/Yüksekokula başvurur ve kesin kayıtları ilgili programa yapılır.\n"
                    "- Başvuru/kayıt aşamasında not döküm belgesi, başarı şartı belgesi, merkezi yerleştirme sınav sonuç belgesi ve gerekiyorsa yabancı dil yeterlik sınav sonuç belgesi istenir.\n"
                    "- Sürecin yürütülmesi için çift anadal yapılan bölümde bir program koordinatörü görevlendirilir.\n\n"
                    "Kaynak: " + "; ".join(selected_sources) + "."
                )

        if facets["primary_intent"] == "surec" and "tezsiz_yuksek_lisans" in facets["entities"]:
            process_docs = self._direct_candidate_chunks(question, facets)
            sources = []
            for doc in process_docs:
                source_text = self._source_text(doc)
                if source_text not in sources:
                    sources.append(source_text)
                if len(sources) >= 4:
                    break
            if sources:
                return (
                    "Tezsiz yüksek lisans süreci özetle şöyledir:\n\n"
                    "- Programa başvuru koşulları ve kontenjanlar Enstitü Yönetim Kurulu kararıyla belirlenir.\n"
                    "- Adayın lisans diplomasına sahip olması gerekir.\n"
                    "- Öğrenci alımı lisans mezuniyet not ortalamasına göre yapılır; adaylar bu ortalamaya göre sıralanır.\n"
                    "- Program en az 10 ders ve 1 dönem projesinden oluşur; toplam en az 30 kredi ve 60 AKTS olmalıdır.\n"
                    "- Programı başarıyla tamamlayan ve dönem projesini usulüne uygun hazırlayan öğrenciye tezsiz yüksek lisans diploması verilir.\n\n"
                    "Kaynak: " + "; ".join(sources) + "."
                )

        if {"cift_anadal", "yandal"} & facets["entities"] and "sart" in facets["intents"]:
            cift_docs = [
                doc for doc in self._direct_candidate_chunks(question, facets)
                if str(doc.get("metadata", {}).get("madde", "")).startswith("7")
            ]
            cift_sentences = []
            for doc in cift_docs:
                for sentence in self._clause_candidates(doc.get("content", "")):
                    normalized_sentence = self._normalize(sentence)
                    if any(term in normalized_sentence for term in ("2.75", "en ust %20", "tum dersleri", "basvuru", "basari siralamasi", "taban puani", "ayni isimli")):
                        cift_sentences.append((self._sentence_score(sentence, query_tokens), sentence, doc))
            if cift_sentences:
                cift_sentences.sort(key=lambda item: item[0], reverse=True)
                selected_cift = []
                seen_cift = set()
                for _, sentence, doc in cift_sentences:
                    key = self._normalize(sentence[:120])
                    if key in seen_cift:
                        continue
                    seen_cift.add(key)
                    selected_cift.append((sentence, doc))
                    if len(selected_cift) >= 5:
                        break
                metadata = selected_cift[0][1].get("metadata", {})
                source_text = f"{self._source_label(metadata.get('source', 'Bilinmeyen belge'))}, Madde {metadata.get('madde', '-')}"
                bullets = "\n".join(f"- {self._clean_snippet(sentence)}" for sentence, _ in selected_cift)
                return "İlgili yönergeye göre çift anadal şartları:\n\n" + bullets + "\n\nKaynak: " + source_text + "."

        if "ozel_ogrenci" in facets["entities"] and "sart" in facets["intents"]:
            special_docs = [
                doc for doc in (self._direct_candidate_chunks(question, facets) + docs)
                if "ozel_ogrenci_yonergesi" in self._normalize(doc.get("metadata", {}).get("source", ""))
                and str(doc.get("metadata", {}).get("madde", "")).startswith(("4", "5", "6"))
            ]
            if special_docs:
                bullets = []
                sources = []
                for doc in special_docs[:3]:
                    metadata = doc.get("metadata", {})
                    doc_norm = self._normalize(doc.get("content", ""))
                    if "2.50/4.00" in doc_norm or "2,50/4,00" in doc_norm:
                        bullets.append("- Öğrencinin kayıtlı olduğu üniversitede genel akademik başarı not ortalamasının 4.00 üzerinden en az 2.50 olması gerekir.")
                    for sentence in self._clause_candidates(doc.get("content", "")):
                        normalized_sentence = self._normalize(sentence)
                        if any(term in normalized_sentence for term in ("2.50/4.00", "en az", "ayni duzey", "eşdeğer", "esdeger", "disiplin", "basvurabilmesi")):
                            bullets.append(f"- {self._clean_snippet(sentence)}")
                            break
                    source_text = f"{self._source_label(metadata.get('source', 'Bilinmeyen belge'))}, Madde {metadata.get('madde', '-')}"
                    if source_text not in sources:
                        sources.append(source_text)
                if bullets:
                    return "İlgili yönergeye göre özel öğrenci şartları:\n\n" + "\n".join(bullets[:5]) + "\n\nKaynak: " + "; ".join(sources) + "."

        if intent == "definition":
            definition_sentences = []
            for doc in docs[:5]:
                metadata = doc.get("metadata", {})
                if primary_source and metadata.get("source") != primary_source:
                    continue
                if "ozel ogrenci" in qnorm:
                    match = re.search(
                        r"Özel\s+Öğrenci\s*:\s*(.*?öğrenciyi)",
                        doc.get("content", ""),
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    if match:
                        sentence = "Özel öğrenci: " + self._clean_snippet(match.group(1))
                        score = self._sentence_score(sentence, query_tokens) + 8.0
                        definition_sentences.append((score, sentence, doc))
                for sentence in self._clause_candidates(doc.get("content", "")):
                    normalized_sentence = self._normalize(sentence)
                    if "ozel ogrenci" in qnorm:
                        label = self._normalize(sentence.split(":", 1)[0]) if ":" in sentence[:120] else ""
                        is_exact_special_student = "ozel ogrenci" in label and "komisyon" not in label
                        if not is_exact_special_student or "ifade eder" not in normalized_sentence:
                            continue
                    elif not self._is_direct_definition_sentence(sentence, question):
                        continue
                    if any(term in normalized_sentence for term in ("komisyon", "kurul", "basvuru", "ceza")) and "ifade eder" not in normalized_sentence:
                        continue
                    score = self._sentence_score(sentence, query_tokens) + self._intent_match_score(sentence, intent)
                    if str(metadata.get("madde")).split("-")[0] in {"1", "3", "4", "Giriş"}:
                        score += 2.0
                    if "ifade eder" in normalized_sentence:
                        score += 3.0
                    definition_sentences.append((score, sentence, doc))
            if definition_sentences:
                definition_sentences.sort(key=lambda item: item[0], reverse=True)
                sentence, doc = definition_sentences[0][1], definition_sentences[0][2]
                metadata = doc.get("metadata", {})
                source_text = f"{self._source_label(metadata.get('source', 'Bilinmeyen belge'))}, Madde {metadata.get('madde', '-')}"
                return (
                    "İlgili yönergeye göre kısa tanım:\n\n"
                    f"- {self._clean_snippet(sentence)}\n\n"
                    f"Kaynak: {source_text}."
                )
            no_definition = self._definition_no_direct_answer(question, docs)
            if no_definition:
                return no_definition

        if "erasmus" in qnorm and ("sinav" in qnorm or "sınav" in question.lower()) and not any(term in qnorm for term in ("baraj", "puan", "olcut", "ölçüt")):
            selected_language = []
            for doc in docs[:5]:
                if primary_source and doc.get("metadata", {}).get("source") != primary_source:
                    continue
                for sentence in self._clause_candidates(doc.get("content", "")):
                    normalized_sentence = self._normalize(sentence)
                    has_language_exam = (
                        any(term in normalized_sentence for term in ("dil", "ingilizce", "yabanci"))
                        and any(term in normalized_sentence for term in ("sinav", "sınav"))
                    )
                    if not has_language_exam:
                        continue
                    score = self._sentence_score(sentence, query_tokens)
                    if any(term in normalized_sentence for term in ("zorunlu", "girmesi", "girmek")):
                        score += 5.0
                    if "ogrenci" in normalized_sentence or "ögrenci" in normalized_sentence:
                        score += 1.5
                    selected_language.append((score, sentence, doc))
            if selected_language:
                selected_language.sort(key=lambda item: item[0], reverse=True)
                sentence, doc = selected_language[0][1], selected_language[0][2]
                metadata = doc.get("metadata", {})
                source_text = f"{self._source_label(metadata.get('source', 'Bilinmeyen belge'))}, Madde {metadata.get('madde', '-')}"
                clean_sentence = self._clean_snippet(sentence)
                return (
                    "Evet. İlgili yönergeye göre öğrencinin Üniversite tarafından yapılacak yabancı dil sınavına girmesi zorunludur.\n\n"
                    f"Kaynak: {source_text}."
                )

        if "erasmus" in qnorm and any(term in qnorm for term in ("baraj", "puan", "olcut", "ölçüt")):
            score_sentences = []
            for doc in docs[:5]:
                metadata = doc.get("metadata", {})
                if primary_source and metadata.get("source") != primary_source:
                    continue
                content_norm = self._normalize(doc.get("content", ""))
                if "personel" in content_norm and "ogrenci" not in content_norm:
                    continue
                for sentence in self._clause_candidates(doc.get("content", "")):
                    normalized_sentence = self._normalize(sentence)
                    if not any(term in normalized_sentence for term in ("puan", "baraj", "olcut", "degerlendirme", "not ortalamasi", "2.20", "2.50", "yabanci dil")):
                        continue
                    if any(term in normalized_sentence for term in ("personel", "katilim sertifikasi", "hizmet sureleri", "not donusum", "donusum tablosu", "akts not")):
                        continue
                    score = self._sentence_score(sentence, query_tokens)
                    if any(term in normalized_sentence for term in ("2.20", "2.50", "yabanci dil", "not ortalamasi")):
                        score += 3.0
                    score_sentences.append((score, sentence, doc))
            if score_sentences:
                score_sentences.sort(key=lambda item: item[0], reverse=True)
                selected_score = []
                seen_score = set()
                for _, sentence, doc in score_sentences:
                    key = self._normalize(sentence[:120])
                    if key in seen_score:
                        continue
                    seen_score.add(key)
                    selected_score.append((sentence, doc))
                    if len(selected_score) >= 4:
                        break
                sources = []
                bullets = []
                for sentence, doc in selected_score:
                    bullets.append(f"- {self._clean_snippet(sentence)}")
                    metadata = doc.get("metadata", {})
                    source_text = f"{self._source_label(metadata.get('source', 'Bilinmeyen belge'))}, Madde {metadata.get('madde', '-')}"
                    if source_text not in sources:
                        sources.append(source_text)
                return "İlgili yönergeye göre puan/sınav bilgisi:\n\n" + "\n".join(bullets) + "\n\nKaynak: " + "; ".join(sources) + "."

        if "erasmus" in qnorm and any(term in qnorm for term in ("her bolum", "her bölüm", "tum bolum", "tüm bölüm", "bolum icin", "bölüm için")):
            eligibility_sentences = []
            for doc in docs[:5]:
                metadata = doc.get("metadata", {})
                if primary_source and metadata.get("source") != primary_source:
                    continue
                for sentence in self._clause_candidates(doc.get("content", "")):
                    normalized_sentence = self._normalize(sentence)
                    if any(term in normalized_sentence for term in ("personel", "taban puan", "katilim sertifikasi", "not donusum", "donusum tablosu", "akts not")):
                        continue
                    if not any(term in normalized_sentence for term in ("ikili anlasma", "tercih", "kontenjan", "akademik birim", "basvuru yontemi")):
                        continue
                    score = self._sentence_score(sentence, query_tokens)
                    if any(term in normalized_sentence for term in ("kontenjan", "akademik birim", "tercih", "basvuru yontemi")):
                        score += 2.0
                    eligibility_sentences.append((score, sentence, doc))
            if eligibility_sentences:
                eligibility_sentences.sort(key=lambda item: item[0], reverse=True)
                sentence, doc = eligibility_sentences[0][1], eligibility_sentences[0][2]
                metadata = doc.get("metadata", {})
                source_text = f"{self._source_label(metadata.get('source', 'Bilinmeyen belge'))}, Madde {metadata.get('madde', '-')}"
                return (
                    "İlgili yönergeye göre kısa cevap:\n\n"
                    "- Erasmus her bölüm için otomatik bir hak gibi uygulanmaz; başvuruda kontenjanlar, varsa akademik birimlerin özel seçim şartları ve gidilecek üniversite/program tercihleri ilanda belirtilen koşullara göre değerlendirilir.\n"
                    f"Kaynak: {source_text}."
                )

        if "kulup" in query_tokens and any(term in qnorm for term in ("acma", "kurma", "kurulus", "kurul", "açma")):
            club_sentences = []
            for doc in docs[:5]:
                metadata = doc.get("metadata", {})
                source = self._normalize(metadata.get("source", ""))
                if "ogrenci_kulupleri" not in source:
                    continue
                if str(metadata.get("madde")) not in {"9"}:
                    continue
                for sentence in self._clause_candidates(doc.get("content", "")):
                    normalized_sentence = self._normalize(sentence)
                    if "belge" in facets["intents"] and not self._is_document_sentence(sentence, "kulup"):
                        continue
                    if not any(term in normalized_sentence for term in ("kurulabilmesi", "kurulur", "kurucu", "basvuru dilekce", "kulup tuzugu", "senato")):
                        continue
                    score = self._sentence_score(sentence, query_tokens)
                    if str(metadata.get("madde")) == "9":
                        score += 4.0
                    if "en az 15" in normalized_sentence or "15 kurucu" in normalized_sentence:
                        score += 2.0
                    if "basvuru dilekce" in normalized_sentence or "kulup tuzugu" in normalized_sentence:
                        score += 1.5
                    club_sentences.append((score, sentence, doc))
            if club_sentences:
                club_sentences.sort(key=lambda item: item[0], reverse=True)
                selected_club = []
                seen_club = set()
                for _, sentence, doc in club_sentences:
                    key = self._normalize(sentence[:120])
                    if key in seen_club:
                        continue
                    seen_club.add(key)
                    selected_club.append((sentence, doc))
                    if len(selected_club) >= 4:
                        break
                bullets = []
                sources = []
                for sentence, doc in selected_club:
                    bullets.append(f"- {self._clean_snippet(sentence)}")
                    metadata = doc.get("metadata", {})
                    source_text = f"{self._source_label(metadata.get('source', 'Bilinmeyen belge'))}, Madde {metadata.get('madde', '-')}"
                    if source_text not in sources:
                        sources.append(source_text)
                title = (
                    "Gerekli belgeler:"
                    if "belge" in facets["intents"]
                    else "İlgili yönergeye göre kulüp açma şartları:"
                )
                return (
                    title + "\n\n"
                    + "\n".join(bullets)
                    + "\n\nKaynak: "
                    + "; ".join(sources)
                    + "."
                )

        scored_sentences: List[Tuple[float, str, Dict[str, Any]]] = []
        for doc_rank, doc in enumerate(docs[:4]):
            metadata = doc.get("metadata", {})
            if primary_source and metadata.get("source") != primary_source:
                continue
            if {"cift", "anadal"}.issubset(query_tokens):
                if intent == "documents" and str(metadata.get("madde")) != "9":
                    continue
                if intent == "condition" and str(metadata.get("madde")) not in {"7"}:
                    continue
            if (
                metadata.get("madde") == "Giriş"
                and len(docs) > 1
                and "yabanci_uyruklu_ogrenci_kabul" not in self._normalize(metadata.get("source", ""))
            ):
                continue
            for sentence in self._clause_candidates(doc.get("content", "")):
                normalized_sentence = self._normalize(sentence)
                if not asks_definition and ("):" in sentence or " tanımlar" in normalized_sentence):
                    continue
                if ":" in sentence[:80]:
                    label = self._normalize(sentence.split(":", 1)[0])
                    if asks_definition and any(term in label for term in ("ozel ogrenci", "erasmus", "kulup", "cift", "yan dal")):
                        pass
                    elif not label or label not in self._normalize(question):
                        continue
                if not asks_definition and re.match(r"^[A-ZÇĞİÖŞÜa-zçğıöşü\s/()]+:\s", sentence):
                    continue
                normalized_sentence = self._normalize(sentence)
                if "tezsiz yuksek lisans" in qnorm and "tezli yuksek lisans" in normalized_sentence:
                    if not any(term in qnorm for term in ("tezli", "gecis", "geçiş", "aktarma")):
                        continue
                if "cift anadal" in qnorm and "yan dal" in normalized_sentence:
                    continue
                if intent == "condition" and any(term in normalized_sentence for term in ("dahil edilmeksizin", "transkript", "diploma ek")):
                    continue
                if intent == "application" and "ne zaman" in qnorm:
                    if not any(term in normalized_sentence for term in ("tarih", "yariyil", "yarıyıl", "donem", "dönem", "ilan")):
                        continue
                    if any(term in normalized_sentence for term in ("belgesi", "belge", "not dokum", "sonuc belgesi")):
                        continue
                if intent != "documents" and any(term in normalized_sentence for term in ("sonuc belgesi", "sonuç belgesi", "yeterlik belgesi", "not dokum belgesi", "not döküm belgesi")):
                    continue
                if not self._matches_intent(sentence, intent, question):
                    continue
                score = self._sentence_score(sentence, query_tokens)
                score += self._intent_match_score(sentence, intent)
                score -= doc_rank * 0.25
                if metadata.get("madde") in {"Amaç", "Kapsam", "Dayanak"}:
                    score -= 0.8
                if score > 0:
                    scored_sentences.append((score, sentence, doc))

        if not scored_sentences and intent == "answer":
            top_doc = docs[0]
            scored_sentences = [
                (1.0, sentence, top_doc)
                for sentence in self._sentence_candidates(top_doc.get("content", ""))[:5]
            ]
        elif not scored_sentences:
            for doc_rank, doc in enumerate(docs[:4]):
                metadata = doc.get("metadata", {})
                if primary_source and metadata.get("source") != primary_source:
                    continue
                if {"cift", "anadal"}.issubset(query_tokens):
                    if intent == "documents" and str(metadata.get("madde")) != "9":
                        continue
                    if intent == "condition" and str(metadata.get("madde")) not in {"7"}:
                        continue
                if (
                    metadata.get("madde") == "Giriş"
                    and len(docs) > 1
                    and "yabanci_uyruklu_ogrenci_kabul" not in self._normalize(metadata.get("source", ""))
                ):
                    continue
                for sentence in self._clause_candidates(doc.get("content", "")):
                    normalized_sentence = self._normalize(sentence)
                    if "tezsiz yuksek lisans" in qnorm and "tezli yuksek lisans" in normalized_sentence:
                        if not any(term in qnorm for term in ("tezli", "gecis", "geçiş", "aktarma")):
                            continue
                    if "cift anadal" in qnorm and "yan dal" in normalized_sentence:
                        continue
                    if intent == "condition" and any(term in normalized_sentence for term in ("dahil edilmeksizin", "transkript", "diploma ek")):
                        continue
                    if intent == "application" and "ne zaman" in qnorm:
                        if not any(term in normalized_sentence for term in ("tarih", "yariyil", "yarıyıl", "donem", "dönem", "ilan")):
                            continue
                        if any(term in normalized_sentence for term in ("belgesi", "belge", "not dokum", "sonuc belgesi")):
                            continue
                    if intent != "documents" and any(term in normalized_sentence for term in ("sonuc belgesi", "sonuç belgesi", "yeterlik belgesi", "not dokum belgesi", "not döküm belgesi")):
                        continue
                    if intent == "documents" and not self._matches_intent(sentence, intent, question):
                        continue
                    score = self._sentence_score(sentence, query_tokens) - doc_rank * 0.35
                    if intent == "condition" and any(term in normalized_sentence for term in ("ortalama", "en az", "gerekir", "olmalidir", "olmalıdır")):
                        score += 1.5
                    if intent == "application" and any(term in normalized_sentence for term in ("basvuru", "kayit", "kariyer kapisi", "akademik takvim")):
                        score += 1.5
                    if intent == "definition" and any(term in normalized_sentence for term in ("ifade eder", "kapsar", "amac", "program")):
                        score += 1.5
                    if score > 0:
                        scored_sentences.append((score, sentence, doc))

            if not scored_sentences:
                return None

        if intent == "condition" and "ortalama" in qnorm:
            average_sentences = [
                item for item in scored_sentences if "ortalama" in self._normalize(item[1])
            ]
            if average_sentences:
                scored_sentences = average_sentences

        if "erasmus" in qnorm and any(term in qnorm for term in ("dil", "sinav", "sınav")):
            language_sentences = [
                item
                for item in scored_sentences
                if any(
                    term in self._normalize(item[1])
                    for term in (
                        "yabanci dil",
                        "dil sinav",
                        "dil sinavi",
                        "dil sınavı",
                        "ols",
                        "language",
                    )
                )
            ]
            if language_sentences:
                boosted = []
                for score, sentence, doc in language_sentences:
                    normalized_sentence = self._normalize(sentence)
                    if any(term in normalized_sentence for term in ("zorunlu", "girmesi", "sinavina", "sınavına")):
                        score += 4.0
                    boosted.append((score, sentence, doc))
                scored_sentences = boosted

        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        selected = []
        seen = set()
        max_items = self._intent_profile(intent)["max_items"]
        for _, sentence, doc in scored_sentences:
            key = self._normalize(sentence[:120])
            if key in seen:
                continue
            seen.add(key)
            selected.append((sentence, doc))
            if len(selected) >= max_items:
                break

        if not selected:
            return None

        title = self._intent_title(intent)

        bullets = []
        for sentence, _ in selected:
            sentence = self._clean_snippet(sentence)
            if len(sentence) > 260:
                sentence = sentence[:257].rsplit(" ", 1)[0] + "..."
            bullets.append(f"- {sentence}")

        sources = []
        for _, doc in selected:
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Bilinmeyen belge")
            madde = metadata.get("madde", "-")
            source_text = f"{self._source_label(source)}, Madde {madde}"
            if source_text not in sources:
                sources.append(source_text)
            if len(sources) >= 3:
                break

        return f"{title}\n\n" + "\n".join(bullets) + "\n\nKaynak: " + "; ".join(sources) + "."

    def _matching_rules(self, question: str) -> List[Dict[str, Any]]:
        tokens = self._query_tokens(question)

        rules = []
        for rule in DOCUMENT_RULES:
            name = rule.get("name")
            if name == "erasmus" and "erasmus" not in tokens:
                continue
            if name == "cift_anadal" and not tokens.intersection({"cift", "anadal", "yandal", "yan"}):
                continue
            if name == "tezsiz_yuksek_lisans" and not (
                "tezsiz" in tokens or {"yuksek", "yüksek", "lisans"}.issubset(tokens)
            ):
                continue
            if name == "yaz_okulu" and not {"yaz", "okulu"}.issubset(tokens):
                continue
            if name == "ozel_ogrenci" and "ozel" not in tokens:
                continue
            if name == "uluslararasi_ogrenci" and not tokens.intersection({"uluslararasi", "yabanci", "uyruklu"}):
                continue
            if rule.get("unless") and tokens.intersection(rule["unless"]):
                continue
            if len(tokens.intersection(rule["terms"])) >= min(2, len(rule["terms"])):
                rules.append(rule)
        return rules

    def _document_boost(self, chunk: Dict[str, Any], question: str) -> float:
        source = self._normalize(chunk.get("metadata", {}).get("source", ""))
        content = self._normalize(chunk.get("content", ""))
        madde = str(chunk.get("metadata", {}).get("madde", ""))
        tokens = self._query_tokens(question)
        boost = 0.0

        for rule in self._matching_rules(question):
            if any(self._normalize(src) in source for src in rule["sources"]):
                boost += rule["boost"]

        if {"ulusal", "staj"}.issubset(tokens):
            if "ulusal_staj_programi" in source:
                boost += 4.0
            elif "erasmus" in source:
                boost -= 1.5

        if {"yaz", "okulu"}.issubset(tokens):
            if "yaz_okulu" in source:
                boost += 3.0
            else:
                boost -= 0.5

        if {"sart", "şart", "kosul", "koşul", "basvuru", "başvuru"}.intersection(tokens):
            if any(word in content for word in ("basvuru", "kosul", "sart", "genel ilkeler")):
                boost += 0.35
            if any(phrase in content for phrase in ("staj basvurusu", "basvuru formu", "basvurular", "basvuru kosullari")):
                boost += 1.25
            if any(phrase in content for phrase in ("derslere basvuru", "kayit islemleri", "kayit yaptirmak", "kayit yaptiramaz")):
                boost += 1.6
            if any(phrase in content for phrase in ("devam zorunludur", "ucretini odemeleri", "dersin acilabilmesi")):
                boost += 0.8

        asks_definition = "nedir" in self._normalize(question) or "ne demek" in self._normalize(question)
        has_other_intent = {"sart", "kosul", "basvuru", "belge", "evrak", "sure", "süre", "kac", "kaç"}.intersection(tokens)
        if asks_definition and not has_other_intent:
            if any(word in content for word in ("amac", "kapsam", "kapsar", "ifade eder")):
                boost += 0.9
            if madde in {"1", "2", "4"}:
                boost += 0.35
            if any(word in content for word in ("sure", "tarih", "ucret", "devam")):
                boost -= 0.25

        if {"sure", "süre", "gunu", "günü", "kaç", "kac"}.intersection(tokens):
            if any(word in content for word in ("sure", "is gunu", "kirk", "40", "otuz", "30")):
                boost += 0.45

        asks_ulusal_staj = {"ulusal", "kariyer", "kapisi", "kapısı"}.intersection(tokens)
        if "staj" in tokens and {"sure", "süre", "gunu", "günü", "kac"}.intersection(tokens) and not asks_ulusal_staj:
            if "staj_yonergesi" in source and madde == "5":
                boost += 2.4
            if "staj_yonergesi" in source and "en az 40" in content:
                boost += 1.8
            if "ulusal_staj_programi" in source:
                boost -= 0.8

        if {"ozel", "ogrenci"}.issubset(tokens) and {"sart", "kosul", "basvuru"}.intersection(tokens):
            if "ozel_ogrenci_yonergesi" in source and madde in {"5", "6"}:
                boost += 2.0
            if "ozel_ogrenci_yonergesi" in source and madde == "4":
                boost += 1.2
            if "ozel_ogrenci_yonergesi" in source and any(
                phrase in content
                for phrase in (
                    "basvurabilmesi icin",
                    "kosullari saglamasi",
                    "genel ilkeler",
                    "en fazla iki donem",
                )
            ):
                boost += 0.8

        if "tezsiz" in tokens and {"not", "ortalama", "sart", "kosul", "basvuru"}.intersection(tokens):
            if "tezsiz_yuksek_lisans_yonergesi" in source and madde in {"7", "8"}:
                boost += 2.4
            if "tezsiz_yuksek_lisans_yonergesi" in source and madde == "15":
                if not {"tezli", "gecis"}.intersection(tokens):
                    boost -= 1.5

        if tokens.intersection({"yabanci", "uyruklu", "uluslararasi"}):
            asks_lisansustu = bool(tokens.intersection({"lisansustu", "lisansüstü", "yuksek", "yüksek", "doktora"}))
            if "yabanci_uyruklu_ogrenci_kabul" in source and not asks_lisansustu:
                boost += 3.0
            if "lisanustu_programlara_uluslararasi_ogrenci_kabul" in source and asks_lisansustu:
                boost += 3.0
            if "lisanustu_programlara_uluslararasi_ogrenci_kabul" in source and not asks_lisansustu:
                boost -= 0.8
            if any(word in content for word in ("basvuru kosullari", "kayit islemleri", "yabanci uyruklu statusu")):
                boost += 1.4

        if {"cift", "anadal"}.issubset(tokens) and {"sart", "kosul", "basvuru"}.intersection(tokens):
            if "cift_ana_dal_ve_yan_dal" in source and madde == "7":
                boost += 2.6
            if "cift_ana_dal_ve_yan_dal" in source and madde in {"19", "23"} and not {"belge", "belgeler", "belgeleri", "evrak"}.intersection(tokens):
                boost -= 1.6

        return boost

    def _chunk_intent_score(self, question: str, chunk: Dict[str, Any]) -> float:
        facets = self.extract_query_facets(question)
        normalized = self._normalize(chunk.get("content", ""))
        metadata = chunk.get("metadata", {})
        tags = set(metadata.get("intent_tags", []))
        madde = str(metadata.get("madde", ""))
        score = 0.0

        if facets["primary_intent"] in tags:
            score += 0.45
        if "definition" in facets["intents"]:
            if "definition" in tags:
                score += 0.55
            if madde.split("-")[0] in {"1", "2", "3", "4", "Amaç", "Kapsam", "Tanımlar", "Giriş"}:
                score += 0.25
            if any(term in normalized for term in ("komisyon", "kurul", "basvuru", "kayit", "disiplin")) and "ifade eder" not in normalized:
                score -= 0.55
        if "belge" in facets["intents"] and any(term in normalized for term in ("belge", "belgesi", "evrak", "form", "dilekce")):
            score += 0.45
        if "sart" in facets["intents"] and any(term in normalized for term in ("sart", "kosul", "gerekir", "zorunlu", "en az", "olmalidir")):
            score += 0.45
        if "dil_sinavi" in facets["intents"] and any(term in normalized for term in ("yabanci dil", "ingilizce", "dil sinav", "dil sinavi")):
            score += 0.75
        if "puan_baraj" in facets["intents"] and any(term in normalized for term in ("baraj", "puan", "taban", "olcut", "degerlendirme", "basari")):
            score += 0.55
        if "uygunluk" in facets["intents"] and any(term in normalized for term in ("ikili anlasma", "anlasma", "uygun", "program", "bolum", "kontenjan")):
            score += 0.55
        if "kulup" in facets["entities"] and any(term in normalized for term in ("kulup", "kulubu", "kurulabilmesi", "kurucu uye", "basvuru dilekcesi")):
            score += 0.65
        if "erasmus" in facets["entities"] and "erasmus" in normalized:
            score += 0.45
        if {"cift_anadal", "yandal"} & facets["entities"] and any(term in normalized for term in ("cift ana dal", "yan dal", "anadal")):
            score += 0.45

        if "dil_sinavi" in facets["intents"] and any(term in normalized for term in ("kabul mektubu", "vize", "pasaport")):
            score -= 0.8
        return max(0.0, min(score, 1.0))

    def _title_match_score(self, question: str, chunk: Dict[str, Any]) -> float:
        qtokens = self._query_tokens(question)
        metadata = chunk.get("metadata", {})
        title_tokens = set(self._tokenize(" ".join([
            metadata.get("source", ""),
            metadata.get("title", ""),
            metadata.get("topic", ""),
        ])))
        if not qtokens or not title_tokens:
            return 0.0
        return min(1.0, len(qtokens.intersection(title_tokens)) / max(2, len(qtokens)))

    def _topic_match_score(self, facets: Dict[str, Any], chunk: Dict[str, Any]) -> float:
        topic = facets.get("primary_topic")
        if not topic:
            return 0.0
        chunk_topic = chunk.get("metadata", {}).get("topic", "")
        if topic == chunk_topic:
            return 1.0
        if topic in {"cift_anadal", "yandal"} and chunk_topic == "cift_anadal_yandal":
            return 1.0
        return 0.0

    def _is_in_scope(self, question: str) -> bool:
        normalized = self._normalize(question)
        out_of_scope_terms = {
            "hava",
            "yemek",
            "yemekhane",
            "telefon",
            "numara",
            "rektör",
            "rektor",
            "kimdir",
            "kapanıyor",
            "kapaniyor",
            "saat kac",
            "saat kaç",
        }
        if any(term in normalized for term in out_of_scope_terms) and not any(
            term in normalized for term in ("ozel ogrenci", "erasmus", "cift anadal", "kulup", "muafiyet", "staj", "yonerge")
        ):
            return False
        tokens = self._query_tokens(question)
        if tokens.intersection({"merhaba", "selam", "teşekkür", "tesekkur"}):
            return True
        education_terms = {
            "akts",
            "anadal",
            "basvuru",
            "başvuru",
            "belge",
            "ders",
            "diploma",
            "erasmus",
            "harc",
            "intibak",
            "kayit",
            "kayıt",
            "lisans",
            "mezuniyet",
            "muafiyet",
            "not",
            "puan",
            "baraj",
            "bagil",
            "degerlendirme",
            "final",
            "ogrenci",
            "öğrenci",
            "sinav",
            "sınav",
            "staj",
            "sart",
            "kosul",
            "danisman",
            "tezsiz",
            "transkript",
            "yatay",
            "yaz",
            "yonerge",
            "yönerge",
            "yuksek",
            "yüksek",
        }
        return bool(tokens.intersection(education_terms))

    def search(self, question: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        if self.index is None or self.bm25 is None:
            logger.warning("RAG kaynaklari yuklu degil.")
            return []

        top_k = max(10, min(top_k, 15))
        facets = self.extract_query_facets(question)
        normalized_question = self.normalize_query(question)
        q_vec = self.model.encode([f"query: {normalized_question}"], normalize_embeddings=True).astype("float32")
        v_distances, v_indices = self.index.search(q_vec, min(len(self.chunks), top_k * 5))

        tokenized_query = self._tokenize(normalized_question)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        max_bm25 = float(np.max(bm25_scores)) if len(bm25_scores) and np.max(bm25_scores) > 0 else 1.0

        semantic_scores: Dict[int, float] = {}
        for dist, idx in zip(v_distances[0], v_indices[0]):
            if idx != -1:
                semantic_scores[int(idx)] = max(0.0, float(dist))

        keyword_scores: Dict[int, float] = {}
        for idx, score in enumerate(bm25_scores):
            if score > 0:
                keyword_scores[idx] = float(score / max_bm25)

        candidate_indices = set(semantic_scores) | set(keyword_scores)
        scored: Dict[int, float] = {}
        for idx in candidate_indices:
            chunk = self.chunks[idx]
            source = self._normalize(chunk.get("metadata", {}).get("source", ""))
            if not self._is_allowed_source(source, facets):
                continue
            semantic_score = semantic_scores.get(idx, 0.0)
            keyword_score = keyword_scores.get(idx, 0.0)
            title_score = self._title_match_score(question, chunk)
            topic_score = self._topic_match_score(facets, chunk)
            intent_score = self._chunk_intent_score(question, chunk)
            answerability_score = self._answerability_score(question, chunk)
            final_score = (
                semantic_score * 0.4
                + keyword_score * 0.2
                + intent_score * 0.2
                + topic_score * 0.15
                + answerability_score * 0.1
            )
            scored[idx] = final_score

        sorted_indices = sorted(scored.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results = []
        seen = set()
        for idx, score in sorted_indices:
            chunk = dict(self.chunks[idx])
            chunk["metadata"] = dict(chunk.get("metadata", {}))
            dedupe_key = (chunk["metadata"].get("source"), chunk["metadata"].get("madde"))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            chunk["retrieval_score"] = round(score, 4)
            chunk["semantic_score"] = round(semantic_scores.get(idx, 0.0), 4)
            chunk["keyword_score"] = round(keyword_scores.get(idx, 0.0), 4)
            results.append(chunk)
        return results

    def retrieve_relevant_chunks(self, question: str, top_k: int = 6) -> List[Dict[str, Any]]:
        facets = self.extract_query_facets(question)
        if "dil_sinavi" in facets["intents"] and not facets["entities"]:
            logger.info("Belirsiz dil sinavi sorusu, program/yönerge entity yok: %s", question)
            return []

        docs = self.search(question, top_k=15)
        docs = self._merge_direct_candidates(question, docs, facets)
        if not docs:
            return []

        tokens = self._query_tokens(question)
        normalized = self._normalize(question)
        if "kayit dondurma" in normalized or "kayıt dondurma" in normalized:
            direct = [
                doc for doc in docs
                if "kayit dondur" in self._normalize(doc.get("content", ""))
                or "kayıt dondur" in doc.get("content", "").lower()
            ]
            if not direct:
                return []
            docs = direct

        if "erasmus" in tokens:
            erasmus_docs = [
                doc for doc in docs
                if "erasmus" in self._normalize(doc.get("metadata", {}).get("source", ""))
            ]
            if erasmus_docs:
                docs = erasmus_docs

        if "kulup" in facets["entities"]:
            club_docs = [
                doc for doc in docs
                if "ogrenci_kulupleri" in self._normalize(doc.get("metadata", {}).get("source", ""))
            ]
            if club_docs:
                docs = club_docs

        if "muafiyet" in facets["entities"]:
            muafiyet_docs = [
                doc for doc in docs
                if "muafiyet_ve_intibak" in self._normalize(doc.get("metadata", {}).get("source", ""))
            ]
            if muafiyet_docs:
                docs = muafiyet_docs

        if {"cift", "anadal"}.issubset(tokens):
            cift_docs = [
                doc for doc in docs
                if "cift_ana_dal" in self._normalize(doc.get("metadata", {}).get("source", ""))
            ]
            if cift_docs:
                docs = cift_docs

        best_score = float(docs[0].get("retrieval_score", 0.0))
        if best_score < 0.22:
            logger.info("Dusuk retrieval skoru: %.3f | soru=%s", best_score, question)
            return []
        reranked = self.rerank_chunks(question, docs)
        if not reranked:
            return []
        if float(reranked[0].get("final_score", reranked[0].get("rerank_score", 0.0))) < 0.28:
            logger.info("Dusuk final skor: %.3f | soru=%s", float(reranked[0].get("final_score", 0.0)), question)
            return []
        return reranked

    def _merge_direct_candidates(self, question: str, docs: List[Dict[str, Any]], facets: Dict[str, Any]) -> List[Dict[str, Any]]:
        direct = self._direct_candidate_chunks(question, facets)
        merged = []
        seen = set()
        for doc in direct + docs:
            metadata = doc.get("metadata", {})
            key = (metadata.get("source"), metadata.get("madde"), doc.get("content", "")[:80])
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
        return merged

    def _direct_candidate_chunks(self, question: str, facets: Dict[str, Any]) -> List[Dict[str, Any]]:
        qnorm = self._normalize(question)
        candidates = []
        for chunk in self.chunks:
            metadata = chunk.get("metadata", {})
            source = self._normalize(metadata.get("source", ""))
            content = self._normalize(chunk.get("content", ""))
            madde = str(metadata.get("madde", ""))

            selected = False
            if "surec" in facets["intents"] and self._process_articles_for_topic(facets):
                selected = madde.split("-")[0] in self._process_articles_for_topic(facets) or madde in self._process_articles_for_topic(facets)
                if selected and facets["allowed_sources"]:
                    selected = any(self._normalize(src) in source for src in facets["allowed_sources"])
            elif {"cift_anadal", "yandal"} & facets["entities"] and "definition" in facets["intents"]:
                selected = "ana_dal" in source and "yan_dal" in source and madde.split("-")[0] in {"4"}
            elif "ozel_ogrenci" in facets["entities"] and "sart" in facets["intents"]:
                selected = "ozel_ogrenci_yonergesi" in source and madde.split("-")[0] in {"4", "5", "6"}
            elif "ozel_ogrenci" in facets["entities"] and "definition" in facets["intents"]:
                selected = "ozel_ogrenci_yonergesi" in source and madde.startswith("3")
            elif "tezsiz_yuksek_lisans" in facets["entities"] and "definition" in facets["intents"]:
                selected = "tezsiz_yuksek_lisans" in source and madde.split("-")[0] in {"1", "2", "3", "4"}
            elif "erasmus" in facets["entities"] and "definition" in facets["intents"]:
                selected = "erasmus" in source and madde.startswith("Giriş")
            elif {"cift_anadal", "yandal"} & facets["entities"] and "belge" in facets["intents"]:
                selected = "ana_dal" in source and "yan_dal" in source and madde.startswith("9")
            elif {"cift_anadal", "yandal"} & facets["entities"] and "evet_hayir" in facets["intents"]:
                selected = "ana_dal" in source and "yan_dal" in source and madde.startswith("7")
            elif {"cift_anadal", "yandal"} & facets["entities"] and "sart" in facets["intents"]:
                selected = "ana_dal" in source and "yan_dal" in source and madde.startswith("7")
            elif "kulup" in facets["entities"] and ("belge" in facets["intents"] or "sart" in facets["intents"] or "basvuru" in facets["intents"]):
                selected = "ogrenci_kulupleri" in source and madde.startswith("9")
            elif "erasmus" in facets["entities"] and "dil_sinavi" in facets["intents"]:
                selected = "erasmus" in source and "yabanci dil sinav" in content and ("girmesi" in content or "zorunlu" in content)
            elif "erasmus" in facets["entities"] and "puan_baraj" in facets["intents"]:
                selected = (
                    "erasmus" in source
                    and any(term in content for term in ("2.20/4.00", "2.50/4.00", "yabanci dil sinav", "not ortalamasi"))
                    and not any(term in content for term in ("not donusum", "donusum tablosu", "personel hizmet", "katilim sertifikasi"))
                )
            elif "erasmus" in facets["entities"] and "uygunluk" in facets["intents"]:
                selected = (
                    "erasmus" in source
                    and any(term in content for term in ("kontenjan", "akademik birim", "universite tercih", "basvuru yontemi"))
                    and not any(term in content for term in ("not donusum", "personel hizmet", "katilim sertifikasi"))
                )

            if not selected:
                continue
            item = dict(chunk)
            item["metadata"] = dict(metadata)
            item["retrieval_score"] = max(float(item.get("retrieval_score", 0.0)), 0.82)
            item["semantic_score"] = item.get("semantic_score", 0.0)
            item["keyword_score"] = item.get("keyword_score", 0.0)
            candidates.append(item)
            if len(candidates) >= 6:
                break
        return candidates

    def _process_articles_for_topic(self, facets: Dict[str, Any]) -> set[str]:
        if {"cift_anadal", "yandal"} & facets["entities"]:
            return {"7", "8", "9", "10"}
        if "kulup" in facets["entities"]:
            return {"7", "9", "10"}
        if "erasmus" in facets["entities"]:
            return {"Giriş-7", "Giriş-9"}
        if "ozel_ogrenci" in facets["entities"]:
            return {"4", "5", "6"}
        if "tezsiz_yuksek_lisans" in facets["entities"]:
            return {"6", "7", "8", "10", "11", "14"}
        return set()

    def rerank_chunks(self, question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        intent = self._question_intent(question)
        qtokens = self._query_tokens(question)
        facets = self.extract_query_facets(question)
        reranked = []
        for doc in docs:
            content = doc.get("content", "")
            normalized_content = self._normalize(content)
            metadata = doc.get("metadata", {})
            answerability = self._answerability_score(question, doc)
            if answerability < 0.2:
                continue
            score = float(doc.get("retrieval_score", 0.0))
            category = metadata.get("category") or self._guess_category(content, metadata)
            if "dil_sinavi" in facets["intents"]:
                if not any(term in normalized_content for term in ("yabanci dil", "ingilizce", "dil sinav", "dil sinavi")):
                    continue
                if any(term in normalized_content for term in ("kabul mektubu", "vize islemleri", "pasaport")):
                    score -= 0.35
            if intent == "documents" and category == "documents":
                score += 0.25
            if intent == "condition" and category == "conditions":
                score += 0.2
            if intent == "application" and category == "application":
                score += 0.18
            if intent == "duration" and category == "duration":
                score += 0.18
            if "erasmus" in qtokens and any(term in normalized_content for term in ("yabanci dil", "dil sinav", "dil sinavi", "dil sınavı")):
                score += 0.3
            score += self._chunk_intent_score(question, doc) * 0.35
            score += answerability * 0.45
            item = dict(doc)
            item["rerank_score"] = round(score, 4)
            item["final_score"] = round(score, 4)
            item["answerability_score"] = round(answerability, 4)
            reranked.append(item)
        reranked.sort(key=lambda item: item.get("final_score", item.get("retrieval_score", 0)), reverse=True)
        return reranked[:5]

    def _answerability_score(self, question: str, doc: Dict[str, Any]) -> float:
        facets = self.extract_query_facets(question)
        content = self._normalize(doc.get("content", ""))
        metadata = doc.get("metadata", {})
        tags = set(metadata.get("intent_tags", []))
        score = 0.0

        if facets["allowed_sources"]:
            source = self._normalize(metadata.get("source", ""))
            if not self._is_allowed_source(source, facets):
                return 0.0
            score += 0.25

        tag_aliases = {
            "belge": "belgeler",
            "sart": "sartlar",
            "surec": "surec",
            "basvuru": "basvuru",
            "sinav": "sinav",
            "dil_sinavi": "sinav",
            "puan_baraj": "puan_baraj",
            "definition": "definition",
            "evet_hayir": "sartlar",
            "ulke_liste": "general",
            "liste": "general",
        }
        if facets["primary_intent"] in tags or tag_aliases.get(facets["primary_intent"]) in tags:
            score += 0.25
        if "definition" in facets["intents"]:
            if "ifade eder" in content or "tanim" in content or "tanım" in content:
                score += 0.65
            elif any(term in content for term in ("amac", "kapsam", "kapsar")):
                score -= 0.25
            if any(term in content for term in ("komisyon", "kurul", "ceza", "sona erer")) and "ifade eder" not in content:
                score -= 0.5
            if facets.get("primary_topic") == "tezsiz_yuksek_lisans" and "ifade eder" not in content:
                score -= 0.45
        if "evet_hayir" in facets["intents"]:
            if any(term in content for term in ("yapilamaz", "yapilabilir", "zorunlu", "gerekir", "degildir", "olmaz")):
                score += 0.45
            if "ayni isimli bolumler" in content:
                score += 0.4
        if "ulke_liste" in facets["intents"]:
            if any(term in content for term in ("ulke listesi", "anlasmali universite listesi", "anlasmali kurum", "ulkeler")):
                score += 0.35
            else:
                score -= 0.4
        if "dil_sinavi" in facets["intents"]:
            if any(term in content for term in ("yabanci dil sinav", "dil sinav", "ingilizce")):
                score += 0.65
            if any(term in content for term in ("kabul mektubu", "vize", "pasaport", "katilim sertifikasi")):
                score -= 0.7
        if "puan_baraj" in facets["intents"]:
            if any(term in content for term in ("puan", "baraj", "taban", "olcut", "degerlendirme", "basari notu")):
                score += 0.45
            if any(term in content for term in ("kabul mektubu", "vize", "pasaport")):
                score -= 0.5
        if "uygunluk" in facets["intents"]:
            if any(term in content for term in ("ikili anlasma", "anlasma", "kontenjan", "program", "bolum", "tercih")):
                score += 0.45
            if any(term in content for term in ("puan", "personel", "katilim sertifikasi")):
                score -= 0.45
        if "sart" in facets["intents"] and any(term in content for term in ("gerekir", "zorunlu", "sart", "kosul", "en az", "olmalidir")):
            score += 0.3
        if "belge" in facets["intents"] and any(term in content for term in ("belge", "belgesi", "evrak", "form", "dilekce")):
            score += 0.3
            if any(
                term in content
                for term in (
                    "yedek ogrenci",
                    "kesin kayit",
                    "ders kayit",
                    "ogrenim suresi",
                    "komisyon",
                    "koordinator",
                    "danisman atan",
                )
            ) and not any(
                term in content
                for term in (
                    "not dokum belgesi",
                    "basari sarti belgesi",
                    "merkezi yerlestirme sinav sonuc belgesi",
                    "yabanci dil yeterlik sinav sonuc belgesi",
                    "basvuru dilekce",
                    "kulup tuzugu",
                    "kulup basvuru formu",
                )
            ):
                score -= 0.35

        qtokens = self._query_tokens(question)
        ctokens = set(self._tokenize(content))
        score += min(0.25, len(qtokens.intersection(ctokens)) * 0.04)
        return max(0.0, min(score, 1.0))

    def _format_context(self, docs: Iterable[Dict[str, Any]]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.get("metadata", {})
            parts.append(
                "\n".join(
                    [
                        f"KAYNAK {i}",
                        f"Belge: {metadata.get('source', 'Bilinmeyen belge')}",
                        f"Madde: {metadata.get('madde', '-')}",
                        f"Metin: {doc.get('content', '').strip()}",
                    ]
                )
            )
        return "\n\n".join(parts)

    def _source_entry(self, doc: Dict[str, Any]) -> Dict[str, str]:
        metadata = doc.get("metadata", {})
        return {
            "source": self._source_label(metadata.get("source", "Bilinmeyen belge")),
            "madde": str(metadata.get("madde", "-")),
            "title": metadata.get("title", ""),
        }

    def _unique_sources(self, docs: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
        sources: List[Dict[str, str]] = []
        seen = set()
        for doc in docs:
            source = self._source_entry(doc)
            key = (source["source"], source["madde"])
            if key in seen:
                continue
            seen.add(key)
            sources.append(source)
            if len(sources) >= 4:
                break
        return sources

    def _sources_for_answer(self, answer: str, docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        source_docs = docs
        madde_matches = set(re.findall(r"Madde\s+([A-Za-zÇĞİÖŞÜçğıöşü0-9-]+)", answer))
        if madde_matches:
            narrowed = [
                doc for doc in docs
                if str(doc.get("metadata", {}).get("madde", "")) in madde_matches
            ]
            if narrowed:
                sources = self._unique_sources(narrowed)
                present = {source["madde"] for source in sources}
                template = self._source_entry(narrowed[0])
                for madde in sorted(madde_matches):
                    if madde not in present:
                        sources.append({**template, "madde": madde})
                    if len(sources) >= 4:
                        break
                return sources
        return self._unique_sources(source_docs)

    def _guidance_message(self, question: str) -> str:
        normalized = self._normalize(question)
        if "erasmus" in normalized:
            unit = "Erasmus koordinatörlüğü"
        elif "staj" in normalized:
            unit = "bölüm staj komisyonu veya fakülte öğrenci işleri"
        elif "yabanci" in normalized or "uluslararasi" in normalized:
            unit = "Uluslararası İlişkiler Koordinatörlüğü veya öğrenci işleri"
        else:
            unit = "ilgili fakülte/bölüm sekreterliği veya öğrenci işleri birimi"
        return (
            "Bu konuda yönergelerde yeterli ve doğrudan bilgi bulamadım. "
            f"Güncel bilgi için üniversite/fakülte sayfasını ve {unit} ile görüşmenizi öneririm."
        )

    def build_prompt(self, question: str, docs: List[Dict[str, Any]]) -> str:
        context = self._format_context(docs)
        return f"""Sen Bitlis Eren Üniversitesi Öğrenci İşleri asistanısın.
Yalnızca verilen KAYNAKLAR içindeki fişlere göre cevap ver.

Kurallar:
- Önce kullanıcı intentini anla: tanım, şart, belge, başvuru, süreç, puan/baraj, sınav, süre veya danışman.
- Kısa/generic takip sorularında önceki mesajlardan gelen topic korunmuş olabilir; verilen soru bu genişletilmiş bağlamla cevaplanır.
- Yeni veya alakasız sorularda eski topic'i kullanma.
- "Nasıl yapılır / süreç nasıl / nasıl başvurulur" sorularında yalnızca şartları değil; başvuru, belge, değerlendirme ve kayıt adımlarını birlikte özetle.
- Kaynakta olmayan bilgiyi üretme.
- Kaynaklar soruyla doğrudan ilgili değilse cevap üretme; "Bu konuda yönergelerde yeterli/doğrudan bilgi bulamadım." de.
- Verilen chunklar soruyu doğrudan cevaplamıyorsa cevap yazma; fallback cümlesiyle yönlendir.
- Sorunun niyetine odaklan; belge sorusunda belge, şart sorusunda şart, süreç sorusunda süreç anlat.
- Belge sorusunda yedek kayıt, kesin kayıt, öğrenim süresi, komisyon, koordinatör veya danışman bilgisini yazma; yalnızca açık belge/evrak/form adlarını yaz.
- Şart sorusunda belge listesini veya kayıt sürecini karıştırma.
- Evet/hayır sorularında önce net "Evet" veya "Hayır" de ve sadece doğrudan ilgili cümleyi kullan.
- Tanım sorularında başlık, Amaç, Kapsam veya Dayanak metinlerini tanım gibi kullanma.
- Liste/ülke sorularında yalnızca kaynakta gerçek liste varsa liste ver; liste yoksa ilgili birime/koordinatörlüğe yönlendir.
- Alakasız maddeyi, başka yönergeyi veya sadece aynı kelime geçtiği için bulunan parçayı kullanma.
- Dil sınavı sorusunda kabul mektubu, vize, pasaport gibi parçaları kullanma.
- Puan/baraj sorusunda personel puan tablosunu öğrenci başvurusu gibi yorumlama.
- Tanım sorusunda komisyon, kurul, başvuru veya ceza maddelerini tanım gibi kullanma.
- Cevap kısa, açık ve öğrenci dilinde olsun.
- Kaynaklarda doğrudan bilgi yoksa bunu söyle ve ilgili birime yönlendir.
- Cevabın sonunda kaynakları "Kaynak: Belge adı, Madde X." biçiminde yaz.

KAYNAKLAR:
{context}

SORU: {question}

Cevap:"""

    def fallback_answer(self, question: str, docs: List[Dict[str, Any]]) -> str:
        generic = self._generic_fast_answer(question, docs)
        if generic:
            return generic
        intent = self._question_intent(question)
        facets = self.extract_query_facets(question)
        if intent == "documents" or facets["primary_intent"] == "belge":
            document_answer = self._document_answer(question, docs, facets)
            if document_answer:
                return document_answer
            return self._guidance_message(question)
        if docs:
            snippets = []
            for doc in docs[:3]:
                for sentence in self._clause_candidates(doc.get("content", "")):
                    if not self._matches_intent(sentence, intent, question):
                        continue
                    if len(sentence) > 40:
                        snippets.append(self._clean_snippet(sentence))
                        break
            if snippets:
                bullets = "\n".join(f"- {snippet[:240].rsplit(' ', 1)[0]}." for snippet in snippets[:3])
                source_text = "; ".join(
                    f"{source['source']}, Madde {source['madde']}" for source in self._unique_sources(docs)
                )
                return f"İlgili fişlere göre kısa bilgi:\n\n{bullets}\n\nKaynak: {source_text}."
        return self._guidance_message(question)

    def generate_answer(self, question: str, docs: List[Dict[str, Any]]) -> str:
        prompt = self.build_prompt(question, docs)
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": DEFAULT_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                        "num_predict": 280,
                        "temperature": 0.0,
                        "top_p": 0.8,
                        "repeat_penalty": 1.15,
                    },
                },
                timeout=LLM_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("response", "").strip()
            if answer:
                return answer
            logger.warning("LLM bos cevap dondu.")
        except requests.exceptions.Timeout:
            logger.exception("LLM zaman asimi.")
        except requests.exceptions.RequestException:
            logger.exception("LLM HTTP/connection hatasi.")
        except json.JSONDecodeError:
            logger.exception("LLM JSON parse hatasi.")
        except Exception:
            logger.exception("LLM beklenmeyen hata.")
        return self.fallback_answer(question, docs)

    def _infer_history_topic(self, history: List[Dict[str, str]]) -> str | None:
        state = self._conversation_state(history)
        return state.get("last_topic")

    def _contextualize_question(self, question: str, history: List[Dict[str, str]] | None = None) -> str:
        if not history:
            return question

        history = history[-5:]
        current_facets = self.extract_query_facets(question)
        if current_facets.get("primary_topic"):
            return question
        state = self._conversation_state(history)
        if self._is_generic_followup(question, current_facets):
            expanded = self._expand_followup_query(question, state)
            if expanded != question:
                logger.info("Generic takip sorusu genisletildi: '%s' -> '%s'", question, expanded)
                return expanded

        tokens = self._query_tokens(question)
        qnorm = self._normalize(question)
        known_topic_terms = {
            "cift",
            "anadal",
            "ozel",
            "ogrenci",
            "erasmus",
            "staj",
            "diploma",
            "yaz",
            "okulu",
            "bagil",
            "degerlendirme",
            "kulup",
            "tezsiz",
            "yuksek",
            "lisans",
            "yabanci",
            "uyruklu",
            "uluslararasi",
        }
        has_topic = bool(tokens.intersection(known_topic_terms)) or any(term in qnorm for term in known_topic_terms)
        if has_topic:
            return question

        topic = self._infer_history_topic(history)

        if topic == "erasmus" and any(term in qnorm for term in ("sinav", "puan", "olcut", "dil")):
            if not any(term in qnorm for term in ("bagil", "mutlak", "final", "butunleme")):
                return f"erasmus {question}"
        if topic and any(term in qnorm for term in ("belge", "evrak", "gerekli")):
            return f"{self._topic_phrase(topic)} {question}"
        if topic == "ulusal_staj" and any(term in qnorm for term in ("basvuru", "başvuru", "nasil", "nasıl")):
            return f"ulusal staj {question}"

        followup_terms = {
            "ayni",
            "bolum",
            "program",
            "ortalama",
            "gno",
            "sart",
            "kosul",
            "belge",
            "basvuru",
            "sure",
            "kac",
            "var",
            "mi",
        }

        looks_like_followup = len(tokens) <= 5 and bool(tokens.intersection(followup_terms))
        if topic and looks_like_followup and not self.extract_query_facets(question).get("primary_topic"):
            return f"{self._topic_phrase(topic)} {question}"
        return question

    def answer(self, question: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
        conversation_state = self._conversation_state((history or [])[-5:])
        if not self._is_in_scope(question):
            return {
                "answer": self._guidance_message(question),
                "sources": [],
                "error": None,
                "conversation_state": {
                    **conversation_state,
                    "expanded_query": None,
                    "current_topic": None,
                    "current_intent": self._intent_label(self.extract_query_facets(question)),
                },
            }

        contextual_question = self._contextualize_question(question, history)
        if not self._is_in_scope(contextual_question):
            return {
                "answer": self._guidance_message(question),
                "sources": [],
                "error": None,
                "conversation_state": conversation_state,
            }

        contextual_facets = self.extract_query_facets(contextual_question)
        if contextual_facets.get("primary_intent") == "ulke_liste" and contextual_facets.get("primary_topic") == "erasmus":
            return {
                "answer": (
                    "Yönergede Erasmus yapılabilecek ülkelerin veya anlaşmalı üniversitelerin doğrudan listesi bulunmamaktadır. "
                    "Erasmus yapılabilecek yerler güncel anlaşmalara ve Erasmus Koordinatörlüğünün ilanlarına göre değişir. "
                    "Güncel liste için Erasmus Koordinatörlüğü sayfasını kontrol ediniz."
                ),
                "sources": [],
                "error": None,
                "conversation_state": {
                    **conversation_state,
                    "expanded_query": contextual_question if contextual_question != question else None,
                    "current_topic": contextual_facets.get("primary_topic"),
                    "current_intent": self._intent_label(contextual_facets),
                },
            }

        try:
            docs = self.retrieve_relevant_chunks(contextual_question, top_k=6)
        except Exception:
            logger.exception("Retrieval hatasi.")
            return {
                "answer": self._guidance_message(question),
                "sources": [],
                "error": "retrieval_error",
                "conversation_state": conversation_state,
            }

        if not docs:
            return {
                "answer": self._guidance_message(question),
                "sources": [],
                "error": None,
                "conversation_state": conversation_state,
            }

        # Fişten çıkarılan cevap önceliklidir; LLM yalnızca ihtiyaç olursa devreye girer.
        answer = self._generic_fast_answer(contextual_question, docs)
        if not answer:
            answer = self.generate_answer(contextual_question, docs)

        return {
            "answer": answer,
            "sources": self._sources_for_answer(answer, docs),
            "error": None,
            "conversation_state": {
                **conversation_state,
                "expanded_query": contextual_question if contextual_question != question else None,
                "current_topic": self.extract_query_facets(contextual_question).get("primary_topic"),
                "current_intent": self._intent_label(self.extract_query_facets(contextual_question)),
            },
        }

    def get_answer(self, question: str, history: List[Dict[str, str]] | None = None):
        yield self.answer(question, history).get("answer", "")


rag_service = RAGService()


