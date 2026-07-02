# 1. Personal PDF Chatbot

A local RAG (Retrieval-Augmented Generation) app: upload **one or more PDFs**,
ask questions in a chat, and an LLM (via **Groq's free tier**) answers
**strictly from the text inside those files** — quoting the evidence and citing
the file and page each fact came from.

> New to the project? Open `study-guide.html` for a from-zero explanation of
> the core concepts, with a live retrieval demo. (The guide covers v1–v2;
> the v3 features below are summarized in this README.)

## The pipeline (v3)

```
 PDFs ─▶ 1 Extract ─▶ 2 Chunk ─▶ 3 Index ─▶ 4½ Rewrite ─▶ 4 Retrieve ─▶ Rerank ─▶ Grade ─▶ 5 Augment ─▶ 6 Generate ─▶ Self-check
        text+tables   parent/     TF-IDF /   follow-ups    hybrid RRF    cross-    CRAG      parents      Groq LLM      groundedness
        +OCR          child       embeddings  standalone    fusion        encoder   retry     into prompt  (fallback)    verification
```

| Stage | What happens | Where |
|---|---|---|
| 1. Extract | Page text + tables (pdfplumber) + OCR fallback for scanned pages | `rag_pipeline.py` → `extract_pages()` |
| 2. Chunk | Small-to-big: small CHILD chunks for search, big PARENT sections for reading; each child indexed with a contextual header | `rag_pipeline.py` → `chunk_pages()` |
| 3. Index | TF-IDF, semantic embeddings (ChromaDB-persisted), or hybrid | `TfidfIndex` / `EmbeddingIndex` / `HybridIndex` |
| 4½. Rewrite | Follow-ups become standalone search queries | `chatbot.py` → `standalone_question()` |
| 4. Retrieve | Top candidates via RRF fusion of both rankings | `HybridIndex.retrieve()` + `rrf_fuse()` |
| +. Rerank | Cross-encoder re-scores question+chunk together, keeps the best | `rag_pipeline.py` → `Reranker` |
| +. Grade & retry | "Is this evidence enough?" — if not, synonym queries and a second pass (CRAG) | `advanced_rag.py` → `smart_retrieve()` |
| 5. Augment | Parent sections + rules injected into the prompt | `chatbot.py` → `ask()` |
| 6. Generate | Streaming answer with exact supporting quotes; automatic model fallback | `chatbot.py` |
| +. Self-check | Every claim verified against the excerpts; unsupported ones flagged | `chatbot.py` → `check_groundedness()` |

Every question shows an **"Agentic trace + retrieved excerpts"** expander — the
pipeline's decisions, the rewritten query, reranker scores, and the exact
sections the model saw.

## Measured on the included sample manual (`sample_questions.json`, top-4)

| Configuration | Hit rate | Correct page ranked #1 |
|---|---|---|
| TF-IDF (words) | 6/8 | 4/8 |
| Embeddings (meaning) | 8/8 | 8/8 |
| Hybrid RRF | 8/8 | 6/8 |
| **Hybrid + reranker** | **8/8** | **8/8** |

LLM-judge answer quality (`eval_answers.py`, 3 questions): faithfulness 5.0/5,
relevance 4.7/5. On this small clean document embeddings alone already do well —
hybrid + reranking earns its keep on bigger, messier documents where exact terms
(model numbers, codes) matter.

## Setup (one time)

```powershell
cd Personal-PDF-Chatbot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Get a **free** API key from https://console.groq.com/keys, then set it:

```powershell
setx GROQ_API_KEY "gsk_..."   # takes effect in NEW terminals
```

## Run

Double-click **`run-chatbot.bat`**, or:

```powershell
.venv\Scripts\activate
streamlit run app.py
```

First run downloads two small local models (~170 MB total): MiniLM embeddings
and the ms-marco cross-encoder. Both run on your machine, free.

## Sidebar controls

| Control | Default | What it does |
|---|---|---|
| Retrieval method | Hybrid | Hybrid fuses word + meaning rankings (RRF); or pick either alone |
| k | 4 | How many parent sections the model reads |
| Child chunk size | 120 | Search granularity; the model always reads the bigger parent |
| Rerank results | on | Cross-encoder precision pass over 16 candidates |
| Agentic retry loop | on | Grade evidence; retry with synonym queries if weak (CRAG) |
| Self-check answers | on | Verify every claim against the excerpts after answering |
| HyDE search probe | off | Search with a model-written hypothetical answer instead of the question |

More toggles on = better answers but more API calls per question (rewrite +
grade + answer + self-check ≈ 4 calls). If you hit free-tier rate limits,
turn some off or wait a minute.

## Evaluation tools

```powershell
# retrieval hit rate across all 4 configurations:
python eval_retrieval.py --pdf sample-manual.pdf --questions sample_questions.json

# LLM-as-judge answer scoring (faithfulness + relevance, RAGAS-style):
python eval_answers.py --pdf sample-manual.pdf --questions sample_questions.json --n 3
```

Write your own questions file (copy `eval_questions.example.json`) for any PDF
you know well — change one setting, re-run, and see whether the number moved.

## Test data

`sample-manual.pdf` is a generated 8-page vacuum-cleaner manual with built-in
traps: page 5 says "rinse" (never "wash"), page 8 says "Weight:" (never
"weigh"), and colors are never mentioned (honesty test). `sample_questions.json`
holds 8 eval questions for it.

## Troubleshooting

- **"ModuleNotFoundError: No module named 'torchvision'" tracebacks in the
  console** — harmless noise from Streamlit's file watcher poking optional
  `transformers` vision modules we don't use. Fixed by
  `.streamlit/config.toml` (`fileWatcherType = "none"`). Side effect: after
  editing the code, restart the app instead of relying on auto-reload — or,
  if you want auto-reload back, set `fileWatcherType = "auto"` and
  `pip install torchvision`.

## Known limitations

- **OCR is basic.** Scanned pages go through RapidOCR — fine for clean scans,
  struggles with handwriting or poor quality. First OCR use downloads its
  models (~15 MB).
- **Tables are flattened to text rows** — good enough for lookups, not for
  complex merged-cell tables.
- **Free tier rate limits.** Advanced toggles multiply API calls per question.
  Only the last 6 Q&A pairs are re-sent, so old turns fall out of model memory.
- **`.chroma` grows** per unique files+settings combination. Delete it anytime.
- **Model retirement** is handled by the `MODELS` fallback list in `chatbot.py`.

## Where to go next (project 4 ideas)

1. **GraphRAG** — extract entities/relations into a knowledge graph for
   "summarize everything" questions that top-k retrieval can't answer.
2. **True contextual retrieval** — have an LLM write each chunk's context
   header at index time (we use a mechanical header now).
3. **Multimodal RAG** — embed figures and diagrams with a vision model, not
   just text.
4. **A proper web deployment** — Streamlit Community Cloud or a small VPS,
   with per-user sessions.
