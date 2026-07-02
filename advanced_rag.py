"""Agentic retrieval (v3): a pipeline that makes decisions.

Classic RAG is a fixed conveyor belt: retrieve -> answer, no matter what.
smart_retrieve() turns it into a small loop with judgment (a "CRAG-lite"):

    rewrite the question -> (optionally HyDE) -> retrieve -> rerank
        -> GRADE: "is this enough to answer?"
        -> if NO: expand the query into synonym variants, retrieve again, fuse
        -> answer with the best evidence found

Every decision is recorded in a human-readable `trace` so the UI can show
exactly what the "agent" did and why.
"""

from chatbot import PdfChatbot
from rag_pipeline import Chunk, Reranker, rrf_fuse

CANDIDATE_POOL = 16  # retrieve this many children before reranking/fusing
MAX_ATTEMPTS = 2  # initial attempt + one corrective retry


def _dedup_parents(
    children: list[tuple[Chunk, float]], k: int
) -> list[tuple[Chunk, float]]:
    """Small-to-big hand-off: children found -> unique PARENT sections to read.

    Several children often share one parent; the model should read that
    parent once, not three overlapping copies.
    """
    seen = set()
    parents = []
    for child, score in children:
        key = (child.source, child.page, child.parent_text)
        if key not in seen:
            seen.add(key)
            parents.append((child, score))
        if len(parents) == k:
            break
    return parents


def smart_retrieve(
    bot: PdfChatbot,
    index,
    reranker: Reranker | None,
    question: str,
    k: int = 4,
    agentic: bool = True,
    use_hyde: bool = False,
):
    """Run the full v3 retrieval flow. Returns (excerpts, trace, search_query).

    excerpts     - list of (chunk, score) whose parent_text feeds the model
    trace        - list of strings describing every step and decision
    search_query - the final standalone query that was searched
    """
    trace: list[str] = []

    # Stage 4½ — make follow-ups standalone
    search_query = bot.standalone_question(question)
    if search_query != question:
        trace.append(f'Rewrote follow-up for search: "{search_query}"')

    # Optional HyDE: search with a fake ideal answer instead of the question
    probe = search_query
    if use_hyde:
        fake = bot.hyde_answer(search_query)
        if fake:
            probe = fake
            trace.append(f'HyDE probe: "{fake[:120]}..."')
        else:
            trace.append("HyDE skipped (API error) — searching with the question")

    def retrieve_and_rank(queries: list[str]) -> list[tuple[Chunk, float]]:
        ranked_lists = [index.retrieve(q, k=CANDIDATE_POOL) for q in queries]
        pool = ranked_lists[0] if len(ranked_lists) == 1 else rrf_fuse(
            ranked_lists, k=CANDIDATE_POOL
        )
        if reranker is not None and pool:
            top = reranker.rerank(search_query, pool, k=max(k * 2, 8))
            trace.append(
                f"Reranked {len(pool)} candidates with the cross-encoder "
                f"(best score {top[0][1]:.2f})"
            )
            return top
        return pool

    children = retrieve_and_rank([probe])
    trace.append(f"Attempt 1: retrieved {len(children)} candidate chunks")

    if agentic and children:
        preview = "\n\n".join(c.parent_text[:600] for c, _ in children[:k])
        if bot.grade_context(search_query, preview):
            trace.append("Grade: excerpts look sufficient — answering")
        else:
            trace.append("Grade: NOT sufficient — trying synonym queries (CRAG retry)")
            variants = bot.expand_queries(search_query)
            if variants:
                trace.append("Query variants: " + " / ".join(f'"{v}"' for v in variants))
                retry = retrieve_and_rank([search_query, *variants])
                merged = rrf_fuse([children, retry], k=CANDIDATE_POOL)
                children = (
                    reranker.rerank(search_query, merged, k=max(k * 2, 8))
                    if reranker is not None
                    else merged
                )
                trace.append(f"Attempt 2: fused pool now {len(children)} candidates")
            else:
                trace.append("Query expansion failed — keeping attempt 1 results")
    elif agentic and not children:
        trace.append("Nothing retrieved — trying synonym queries (CRAG retry)")
        variants = bot.expand_queries(search_query)
        if variants:
            trace.append("Query variants: " + " / ".join(f'"{v}"' for v in variants))
            children = retrieve_and_rank([search_query, *variants])
            trace.append(f"Attempt 2: retrieved {len(children)} candidate chunks")

    excerpts = _dedup_parents(children, k)
    pages = ", ".join(f"{c.source} p.{c.page}" for c, _ in excerpts) or "none"
    trace.append(f"Final evidence: {len(excerpts)} parent section(s) — {pages}")
    return excerpts, trace, search_query
