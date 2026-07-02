"""Streamlit UI (v3): upload PDFs, chat, and watch the agentic pipeline think.

Run with:  streamlit run app.py
"""

import hashlib
import io
from pathlib import Path

import groq
import streamlit as st

from advanced_rag import smart_retrieve
from chatbot import PdfChatbot
from rag_pipeline import (
    EmbeddingIndex,
    HybridIndex,
    Reranker,
    TfidfIndex,
    chunk_pages,
    extract_pages,
)

st.set_page_config(page_title="Personal PDF Chatbot", page_icon="📄")
st.title("📄 Personal PDF Chatbot")
st.caption("Upload PDFs and ask questions — answers come strictly from the documents.")


# Heavy resources are created once per app run, not once per rerun.
@st.cache_resource(show_spinner="Loading the embedding model (first time only)...")
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading the reranker model (first time only)...")
def get_reranker():
    return Reranker()


@st.cache_resource(show_spinner=False)
def get_chroma():
    import chromadb

    return chromadb.PersistentClient(path=str(Path(__file__).parent / ".chroma"))


with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Retrieval method",
        [HybridIndex.label, EmbeddingIndex.label, TfidfIndex.label],
        help=(
            "Hybrid runs BOTH methods and fuses the rankings (RRF) — words catch "
            "exact terms, embeddings catch synonyms. Usually the best choice."
        ),
    )
    k = st.slider(
        "Excerpts per question (k)", 1, 8, 4,
        help="How many parent sections the model gets to read.",
    )
    chunk_words = st.slider(
        "Child chunk size (words)", 80, 300, 120, step=20,
        help=(
            "Size of the small chunks used for SEARCHING. The model always reads "
            "the bigger parent section around each hit (small-to-big retrieval)."
        ),
    )
    st.subheader("Advanced (v3)")
    use_reranker = st.toggle(
        "Rerank results", value=True,
        help="Cross-encoder re-scores question+chunk together. Slower, more accurate.",
    )
    use_agentic = st.toggle(
        "Agentic retry loop", value=True,
        help="Grade the evidence first; if weak, retry with synonym queries (CRAG).",
    )
    use_selfcheck = st.toggle(
        "Self-check answers", value=True,
        help="After answering, verify every claim against the excerpts.",
    )
    use_hyde = st.toggle(
        "HyDE search probe", value=False,
        help="Search with a model-written fake ideal answer instead of the question.",
    )
    if st.button("Clear chat"):
        st.session_state.transcript = []
        st.session_state.bot = PdfChatbot()
    if "bot" in st.session_state:
        st.caption(f"Answering model: `{st.session_state.bot.active_model}`")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more PDFs above to start chatting.")
    st.stop()

# (Re)build the index whenever the files OR the indexing settings change.
files = [(f.name, f.getvalue()) for f in uploaded_files]
fingerprint = hashlib.md5()
for name, data in files:
    fingerprint.update(name.encode())
    fingerprint.update(data)
fingerprint.update(f"|{mode}|{chunk_words}|v3".encode())
index_key = fingerprint.hexdigest()

if st.session_state.get("index_key") != index_key:
    with st.spinner("Reading and indexing (text, tables, OCR if needed)..."):
        chunks = []
        for name, data in files:
            pages = extract_pages(io.BytesIO(data))
            chunks.extend(chunk_pages(pages, source=name, chunk_words=chunk_words))
        if not chunks:
            st.error(
                "No text could be extracted from these PDFs — not even with OCR. "
                "Check that the files aren't empty or corrupted."
            )
            st.stop()
        if mode == TfidfIndex.label:
            st.session_state.index = TfidfIndex(chunks)
        else:
            embed_index = EmbeddingIndex(
                chunks, get_embedder(), get_chroma(), collection_key=index_key
            )
            if mode == HybridIndex.label:
                st.session_state.index = HybridIndex(TfidfIndex(chunks), embed_index)
            else:
                st.session_state.index = embed_index
        st.session_state.bot = PdfChatbot()
        st.session_state.transcript = []
        st.session_state.index_key = index_key
    st.success(
        f"Indexed {len(chunks)} child chunks from {len(files)} file(s) — "
        f"{st.session_state.index.label}."
    )

# Replay the conversation so far (Streamlit reruns this script on every interaction).
for role, text in st.session_state.transcript:
    with st.chat_message(role):
        st.markdown(text)

question = st.chat_input("Ask something about the PDFs...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Retrieving evidence..."):
        retrieved, trace, search_query = smart_retrieve(
            st.session_state.bot,
            st.session_state.index,
            get_reranker() if use_reranker else None,
            question,
            k=k,
            agentic=use_agentic,
            use_hyde=use_hyde,
        )

    with st.chat_message("assistant"):
        # Peek behind the curtain: every decision + exactly what the model sees.
        with st.expander("Agentic trace + retrieved excerpts"):
            for step in trace:
                st.caption(f"• {step}")
            st.divider()
            if retrieved:
                for chunk, score in retrieved:
                    st.markdown(
                        f"**{chunk.source} — page {chunk.page}** — score {score:.2f}"
                    )
                    st.text(
                        chunk.parent_text[:700]
                        + ("..." if len(chunk.parent_text) > 700 else "")
                    )
            else:
                st.markdown("*No related excerpts found in the documents.*")
        try:
            answer = st.write_stream(st.session_state.bot.ask(question, retrieved))
        except groq.AuthenticationError:
            st.error(
                "Groq API key is missing or invalid. Set the `GROQ_API_KEY` "
                "environment variable and restart the app — see README.md "
                "for setup steps."
            )
            st.stop()
        except groq.RateLimitError:
            st.error(
                "Groq free-tier rate limit reached — wait a minute, then ask again. "
                "(Turning off some Advanced toggles reduces API calls per question.)"
            )
            st.stop()

        if retrieved:
            sources = list(
                dict.fromkeys(f"{c.source} p.{c.page}" for c, _ in retrieved)
            )
            st.caption("Sources: " + " · ".join(sources))

        if use_selfcheck and retrieved:
            with st.spinner("Self-checking the answer against the sources..."):
                ok, detail = st.session_state.bot.check_groundedness(
                    answer, st.session_state.bot.format_excerpts(retrieved)
                )
            if ok:
                st.caption(f"✅ Self-check: {detail}")
            else:
                st.warning(f"⚠️ Self-check found unsupported claims:\n\n{detail}")

    st.session_state.transcript.append(("user", question))
    st.session_state.transcript.append(("assistant", answer))
