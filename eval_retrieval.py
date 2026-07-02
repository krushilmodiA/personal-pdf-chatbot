"""Measure retrieval quality: how often does the right page land in the top k?

Write a questions file for a PDF you know well (see eval_questions.example.json),
then run:

    python eval_retrieval.py --pdf manual.pdf --questions my_questions.json
    python eval_retrieval.py --pdf manual.pdf --questions my_questions.json --mode rerank

Or check the script works with the built-in synthetic demo (no PDF needed):

    python eval_retrieval.py --demo

Modes: tfidf | embed | hybrid | rerank (hybrid + cross-encoder) | all
"""

import argparse
import hashlib
import json

from rag_pipeline import TfidfIndex, chunk_pages, extract_pages

DEMO_PAGES = [
    (1, "The X100 vacuum comes with a twelve month limited warranty. "
        "The warranty covers manufacturing defects and workmanship issues. " * 10),
    (2, "To clean the filter, remove the dust container. "
        "Rinse the filter under cold water and let it dry for 24 hours. " * 10),
    (3, "The battery provides 45 minutes of runtime. "
        "It recharges fully in about four hours using the included dock. " * 10),
]

DEMO_QUESTIONS = [
    {"question": "How long is the warranty?", "page": 1},
    {"question": "How do I clean the filter?", "page": 2},
    {"question": "How do I wash it?", "page": 2},  # synonym test: word-matching fails
    {"question": "What is the battery runtime?", "page": 3},
]


def build_embedding_index(chunks):
    from rag_pipeline import EmbeddingIndex  # heavy imports only when needed
    import chromadb
    from sentence_transformers import SentenceTransformer

    key = hashlib.md5(("|".join(c.context_text for c in chunks)).encode()).hexdigest()
    return EmbeddingIndex(
        chunks,
        SentenceTransformer("all-MiniLM-L6-v2"),
        chromadb.PersistentClient(path=".chroma"),
        collection_key=key,
    )


def build_retriever(chunks, mode: str):
    """Returns (label, retrieve_fn) for the requested configuration."""
    if mode == "tfidf":
        index = TfidfIndex(chunks)
        return index.label, index.retrieve
    if mode == "embed":
        index = build_embedding_index(chunks)
        return index.label, index.retrieve
    from rag_pipeline import HybridIndex, Reranker

    hybrid = HybridIndex(TfidfIndex(chunks), build_embedding_index(chunks))
    if mode == "hybrid":
        return hybrid.label, hybrid.retrieve
    reranker = Reranker()  # mode == "rerank"

    def retrieve(question, k=4):
        return reranker.rerank(question, hybrid.retrieve(question, k=16), k=k)

    return "Hybrid + cross-encoder rerank", retrieve


def run_eval(chunks, questions, mode: str, k: int) -> None:
    label, retrieve = build_retriever(chunks, mode)
    hits = 0
    print(f"\n=== {label} - top-{k} ===")
    for item in questions:
        found = retrieve(item["question"], k=k)
        pages_found = [c.page for c, _ in found]
        ok = item["page"] in pages_found
        hits += ok
        print(f"  {'HIT ' if ok else 'MISS'}  {item['question']!r}"
              f"  (want page {item['page']}, got {pages_found})")
    print(f"  Hit rate: {hits}/{len(questions)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", help="Path to the PDF to test against")
    parser.add_argument("--questions", help="JSON file: [{question, page}, ...]")
    parser.add_argument(
        "--mode", choices=["tfidf", "embed", "hybrid", "rerank", "all"], default="all"
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--demo", action="store_true",
                        help="Run against built-in synthetic pages (no PDF needed)")
    args = parser.parse_args()

    if args.demo:
        pages, questions = DEMO_PAGES, DEMO_QUESTIONS
    else:
        if not (args.pdf and args.questions):
            parser.error("--pdf and --questions are required (or use --demo)")
        pages = extract_pages(args.pdf)
        with open(args.questions, encoding="utf-8") as f:
            questions = json.load(f)

    chunks = chunk_pages(pages, source="eval")
    modes = ["tfidf", "embed", "hybrid", "rerank"] if args.mode == "all" else [args.mode]
    for mode in modes:
        run_eval(chunks, questions, mode, args.k)


if __name__ == "__main__":
    main()
