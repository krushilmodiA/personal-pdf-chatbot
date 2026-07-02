"""The retrieval half of the RAG pipeline (v3).

Flow:  PDF file(s) -> page texts (+tables, +OCR fallback)
       -> parent/child chunks with contextual headers
       -> index (TF-IDF, embeddings, or hybrid RRF fusion)
       -> top-k retrieval (+ optional cross-encoder reranking)

Key v3 ideas, each explained at its class/function:
  - small-to-big:        search SMALL child chunks, hand the model BIG parents
  - contextual headers:  each child is indexed with a "where am I from" prefix
  - hybrid + RRF:        fuse word-ranking and meaning-ranking into one list
  - reranker:            a cross-encoder re-scores question+chunk TOGETHER

Everything here runs locally. Only answer generation (chatbot.py) calls an API.
"""

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RRF_K = 60  # standard constant in Reciprocal Rank Fusion


@dataclass(frozen=True)
class Chunk:
    """A small CHILD piece of a document, plus the bigger PARENT it lives in.

    text         - the child text (what gets scored against your question)
    context_text - header + child text (what actually gets indexed)
    parent_text  - the surrounding section (what the model gets to read)
    """

    text: str
    context_text: str
    parent_text: str
    page: int
    source: str = "document"


# ---------------------------------------------------------------------------
# Stage 1 — Extract (now with tables and an OCR fallback for scanned pages)
# ---------------------------------------------------------------------------

def _read_bytes(pdf_file) -> bytes:
    if isinstance(pdf_file, (bytes, bytearray)):
        return bytes(pdf_file)
    if hasattr(pdf_file, "read"):
        data = pdf_file.read()
        try:
            pdf_file.seek(0)
        except Exception:
            pass
        return data
    with open(pdf_file, "rb") as f:
        return f.read()


def _extract_tables(data: bytes) -> dict[int, str]:
    """Pull tables out of each page as readable 'cell | cell | cell' rows.

    pypdf reads a table as scrambled words; pdfplumber sees the grid lines
    and reconstructs rows. Failure is fine — we just skip tables.
    """
    tables_by_page: dict[int, str] = {}
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for number, page in enumerate(pdf.pages, start=1):
                rows = []
                for table in page.extract_tables() or []:
                    for row in table:
                        cells = [(cell or "").strip() for cell in row]
                        if any(cells):
                            rows.append(" | ".join(cells))
                if rows:
                    tables_by_page[number] = "Table:\n" + "\n".join(rows)
    except Exception:
        pass
    return tables_by_page


def _ocr_page(data: bytes, page_index: int) -> str:
    """OCR fallback: render the page to an image and read the pixels.

    Used only when a page has no extractable text (scanned documents).
    Needs pypdfium2 + rapidocr-onnxruntime; returns "" if unavailable.
    """
    try:
        import numpy as np
        import pypdfium2 as pdfium
        from rapidocr_onnxruntime import RapidOCR

        page = pdfium.PdfDocument(data)[page_index]
        image = page.render(scale=2.0).to_pil()
        result, _ = RapidOCR()(np.asarray(image))
        if result:
            return " ".join(line[1] for line in result)
    except Exception:
        pass
    return ""


def extract_pages(pdf_file) -> list[tuple[int, str]]:
    """Stage 1 — Extract: text per page, tables appended, OCR as a last resort."""
    data = _read_bytes(pdf_file)
    reader = PdfReader(io.BytesIO(data))
    tables = _extract_tables(data)

    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < 20:  # practically empty -> probably a scanned image
            ocr_text = _ocr_page(data, number - 1).strip()
            if ocr_text:
                text = ocr_text
        if number in tables:
            text = (text + "\n\n" + tables[number]).strip()
        if text:
            pages.append((number, text))
    return pages


# ---------------------------------------------------------------------------
# Stage 2 — Chunk (small-to-big + contextual headers)
# ---------------------------------------------------------------------------

# Good-enough sentence splitter: breaks after . ! ? followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _sentences_as_word_lists(text: str, max_words: int) -> list[list[str]]:
    """Sentences as word lists; giant unpunctuated blocks get hard-split."""
    pieces = []
    for sentence in _SENTENCE_END.split(text):
        words = sentence.split()
        for start in range(0, len(words), max_words):
            piece = words[start : start + max_words]
            if piece:
                pieces.append(piece)
    return pieces


def _pack_sentences(text: str, budget: int, overlap: int) -> list[str]:
    """Pack whole sentences into windows of ~budget words with overlap."""
    windows = []
    current: list[list[str]] = []
    count = 0
    for sentence in _sentences_as_word_lists(text, max_words=budget):
        if current and count + len(sentence) > budget:
            windows.append(" ".join(w for s in current for w in s))
            carried, carried_count = [], 0
            for prev in reversed(current):
                carried.insert(0, prev)
                carried_count += len(prev)
                if carried_count >= overlap:
                    break
            current, count = carried, carried_count
        current.append(sentence)
        count += len(sentence)
    if current:
        windows.append(" ".join(w for s in current for w in s))
    return windows


def chunk_pages(
    pages: list[tuple[int, str]],
    source: str = "document",
    chunk_words: int = 120,
    overlap_words: int = 30,
) -> list[Chunk]:
    """Stage 2 — small-to-big chunking with contextual headers.

    Two sizes per page:
      PARENTS (~3x child size): rich sections the model will actually read.
      CHILDREN (chunk_words):   sharp little pieces we search with.

    Each child also gets a CONTEXTUAL HEADER for indexing — "[manual.pdf,
    page 7] The warranty section..." — so a child that just says "It lasts
    twelve months" still matches warranty questions. (This is a free, local
    version of Anthropic's "contextual retrieval"; the full version has an
    LLM write each header.)
    """
    overlap_words = min(overlap_words, chunk_words // 2)
    parent_budget = min(3 * chunk_words, 600)
    chunks = []
    for page_number, text in pages:
        for parent in _pack_sentences(text, parent_budget, overlap_words):
            topic = " ".join(parent.split()[:12])  # parent's opening words
            for child in _pack_sentences(parent, chunk_words, overlap_words):
                header = f"[{source}, page {page_number}] {topic}..."
                chunks.append(
                    Chunk(
                        text=child,
                        context_text=f"{header}\n{child}",
                        parent_text=parent,
                        page=page_number,
                        source=source,
                    )
                )
    return chunks


# ---------------------------------------------------------------------------
# Stages 3 & 4 — Index and Retrieve (three interchangeable indexes)
# ---------------------------------------------------------------------------

class TfidfIndex:
    """Word-matching index. Fast, transparent, blind to synonyms."""

    label = "TF-IDF (word matching)"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.context_text for c in chunks])

    def retrieve(self, question: str, k: int = 4) -> list[tuple[Chunk, float]]:
        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in best if scores[i] > 0]


class EmbeddingIndex:
    """Meaning-matching index, persisted in ChromaDB (.chroma folder)."""

    label = "Embeddings (meaning matching)"

    def __init__(self, chunks: list[Chunk], model, chroma_client, collection_key: str):
        self.chunks = chunks
        self.model = model
        self.collection = chroma_client.get_or_create_collection(
            name=f"pdf_{collection_key[:32]}",
            metadata={"hnsw:space": "cosine"},
        )
        if self.collection.count() != len(chunks):
            existing = self.collection.get(include=[])["ids"]
            if existing:
                self.collection.delete(ids=existing)
            vectors = model.encode(
                [c.context_text for c in chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.collection.add(
                ids=[str(i) for i in range(len(chunks))],
                embeddings=vectors.tolist(),
                metadatas=[{"page": c.page, "source": c.source} for c in chunks],
                documents=[c.context_text for c in chunks],
            )

    def retrieve(self, question: str, k: int = 4) -> list[tuple[Chunk, float]]:
        question_vector = self.model.encode([question], normalize_embeddings=True)[0]
        result = self.collection.query(
            query_embeddings=[question_vector.tolist()],
            n_results=min(k, len(self.chunks)),
        )
        hits = []
        for chunk_id, distance in zip(result["ids"][0], result["distances"][0]):
            score = 1.0 - float(distance)
            if score > 0:
                hits.append((self.chunks[int(chunk_id)], score))
        return hits


def rrf_fuse(
    ranked_lists: list[list[tuple[Chunk, float]]], k: int
) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion: merge several rankings into one.

    Each list votes: an item at rank r earns 1/(60+r) points. Items ranked
    well by SEVERAL lists float to the top. Only positions matter, so it
    fuses scores that live on totally different scales (TF-IDF vs cosine).
    """
    points: dict[int, float] = {}
    by_id: dict[int, Chunk] = {}
    for ranked in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked):
            key = id(chunk)
            by_id[key] = chunk
            points[key] = points.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
    if not points:
        return []
    top = max(points.values())
    ordered = sorted(points, key=points.get, reverse=True)[:k]
    return [(by_id[key], points[key] / top) for key in ordered]


class HybridIndex:
    """Hybrid retrieval: run BOTH indexes, fuse their rankings with RRF.

    Word matching catches exact terms ("X100", "0.6 liters") that embeddings
    sometimes fumble; embeddings catch synonyms that words can't see. Fused,
    each covers the other's blind spot.
    """

    label = "Hybrid: words + meaning (RRF)"

    def __init__(self, tfidf: TfidfIndex, embeddings: EmbeddingIndex):
        self.tfidf = tfidf
        self.embeddings = embeddings
        self.chunks = tfidf.chunks

    def retrieve(self, question: str, k: int = 4) -> list[tuple[Chunk, float]]:
        pool = max(k * 3, 12)  # give RRF enough candidates to fuse
        return rrf_fuse(
            [self.tfidf.retrieve(question, pool), self.embeddings.retrieve(question, pool)],
            k=k,
        )


class Reranker:
    """Cross-encoder reranking: the accuracy stage.

    Regular retrieval embeds the question and each chunk SEPARATELY and
    compares the vectors. A cross-encoder reads question + chunk TOGETHER
    and outputs one relevance score — slower per pair, but far more precise.
    Modern pattern: retrieve ~16 candidates cheaply, rerank, keep the best k.
    """

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(self.model_name)

    def rerank(
        self, question: str, candidates: list[tuple[Chunk, float]], k: int = 4
    ) -> list[tuple[Chunk, float]]:
        import math

        if not candidates:
            return []
        scores = self.model.predict([(question, c.text) for c, _ in candidates])
        rescored = [
            (chunk, 1.0 / (1.0 + math.exp(-float(s))))  # squash logits to 0..1
            for (chunk, _), s in zip(candidates, scores)
        ]
        rescored.sort(key=lambda pair: pair[1], reverse=True)
        return rescored[:k]
