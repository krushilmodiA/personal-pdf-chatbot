"""The generation half of the RAG pipeline (v3).

Stages 5 & 6 — Augment and Generate — plus the small "thinking" calls that
power agentic retrieval (advanced_rag.py): query rewriting, query expansion,
HyDE, context grading, and the post-answer groundedness check.

Uses Groq's free tier (get a key at https://console.groq.com/keys).
"""

import groq
from groq import Groq

from rag_pipeline import Chunk

# Tried in order — if Groq ever retires the first model, the next one is
# used automatically. Current list: https://console.groq.com/docs/models
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

# Only this many recent messages are re-sent to the API. Keeps long chats
# inside the free tier's token limits.
HISTORY_LIMIT = 12  # 6 question/answer pairs

SYSTEM_PROMPT = """\
You are a chatbot that answers questions about specific PDF documents.

Each user message contains excerpts retrieved from those documents, followed by a question.
- Answer using ONLY those excerpts (plus excerpts shown earlier in this conversation). \
They are your single source of truth.
- If the excerpts don't contain the information needed, say the document doesn't appear \
to cover it. Never fill the gap with outside knowledge or guesses.
- Cite where each fact came from, like (page 3) — or (manual.pdf, page 3) when several \
documents are loaded.
- For the most important fact in your answer, include the exact supporting sentence \
from the excerpt as a short quote, like: "A full charge takes approximately four hours." \
(manual.pdf, page 3)
- Be clear and concise."""

REWRITE_PROMPT = """\
Conversation so far:
{conversation}

Follow-up question: {question}

Rewrite the follow-up as ONE standalone question that contains all the context \
needed to search a document for the answer. If it is already standalone, return \
it unchanged. Return ONLY the question, nothing else."""

EXPAND_PROMPT = """\
Question: {question}

Write 2 alternative phrasings of this question that use DIFFERENT words but mean \
the same thing (use synonyms and related terms a document might use instead). \
Return only the 2 questions, one per line, nothing else."""

HYDE_PROMPT = """\
Question: {question}

Write a short 2-3 sentence passage, in the style of a product manual or report, \
that would perfectly answer this question. Invent plausible details freely — this \
text is only used as a search probe, never shown to anyone. Return only the passage."""

GRADE_PROMPT = """\
Question: {question}

Retrieved excerpts:
{excerpts}

Can the question be fully answered using ONLY these excerpts? \
Reply with exactly one word: YES or NO."""

GROUNDED_PROMPT = """\
Excerpts the assistant was given:
{excerpts}

The assistant's answer:
{answer}

Check every factual claim in the answer against the excerpts. If every claim is \
directly supported, reply exactly: ALL SUPPORTED
Otherwise list each unsupported claim on its own line, briefly. \
(A statement that the documents don't cover something is fine and counts as supported.)"""


class PdfChatbot:
    """Holds the conversation history and streams grounded answers from Groq."""

    def __init__(self):
        # Credentials come from the GROQ_API_KEY environment variable —
        # never hardcode a key.
        self.client = Groq()
        self.messages = []  # full API-shaped history (with excerpts)
        self.turns = []  # lean (question, answer) pairs, used for query rewriting
        self.active_model = MODELS[0]

    def _complete(self, **kwargs):
        """Call Groq, falling through MODELS if one has been retired."""
        last_error = None
        for model in MODELS:
            try:
                response = self.client.chat.completions.create(model=model, **kwargs)
                self.active_model = model
                return response
            except groq.NotFoundError as error:
                last_error = error
            except groq.BadRequestError as error:
                if "model" not in str(error).lower():
                    raise  # a real request problem, not a retired model
                last_error = error
        raise last_error

    def _one_shot(self, prompt: str, max_tokens: int = 200) -> str:
        """One small non-streaming call; returns "" on any failure."""
        try:
            response = self._complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return ""

    # -- the small "thinking" helpers used by advanced_rag.py ----------------

    def standalone_question(self, question: str) -> str:
        """Stage 4½ — rewrite a follow-up into a standalone search query."""
        if not self.turns:
            return question
        conversation = "\n".join(
            f"User: {q}\nBot: {a[:300]}" for q, a in self.turns[-3:]
        )
        rewritten = self._one_shot(
            REWRITE_PROMPT.format(conversation=conversation, question=question),
            max_tokens=120,
        ).strip('"')
        return rewritten or question

    def expand_queries(self, question: str) -> list[str]:
        """Multi-query: 2 re-phrasings with different words (synonym coverage)."""
        text = self._one_shot(EXPAND_PROMPT.format(question=question), max_tokens=150)
        variants = [line.strip(" -\"") for line in text.splitlines() if line.strip()]
        return variants[:2]

    def hyde_answer(self, question: str) -> str:
        """HyDE: a fake ideal answer used as the search probe (never shown)."""
        return self._one_shot(HYDE_PROMPT.format(question=question), max_tokens=150)

    def grade_context(self, question: str, excerpts: str) -> bool:
        """CRAG judge: are these excerpts enough to answer? Defaults to True."""
        verdict = self._one_shot(
            GRADE_PROMPT.format(question=question, excerpts=excerpts[:4000]),
            max_tokens=5,
        )
        return not verdict.upper().startswith("NO")

    def check_groundedness(self, answer: str, excerpts: str) -> tuple[bool, str]:
        """Post-answer self-check: is every claim supported by the excerpts?"""
        verdict = self._one_shot(
            GROUNDED_PROMPT.format(excerpts=excerpts[:5000], answer=answer[:2000]),
            max_tokens=200,
        )
        if not verdict:
            return True, "check skipped (API error)"
        if "ALL SUPPORTED" in verdict.upper():
            return True, "every claim is supported by the excerpts"
        return False, verdict

    # -- the main answer call -------------------------------------------------

    @staticmethod
    def format_excerpts(retrieved: list[tuple[Chunk, float]]) -> str:
        if not retrieved:
            return "(The retriever found no excerpts related to this question.)"
        return "\n\n".join(
            f"[Excerpt from {chunk.source}, page {chunk.page}]\n{chunk.parent_text}"
            for chunk, _score in retrieved
        )

    def ask(self, question: str, retrieved: list[tuple[Chunk, float]]):
        """Stream the answer to `question`, grounded in the retrieved chunks."""
        excerpts = self.format_excerpts(retrieved)
        self.messages.append(
            {
                "role": "user",
                "content": f"Document excerpts:\n\n{excerpts}\n\nQuestion: {question}",
            }
        )

        answer_parts = []
        try:
            stream = self._complete(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *self.messages[-HISTORY_LIMIT:],
                ],
                temperature=0.2,  # low temperature keeps answers factual, not creative
                max_tokens=2048,
                stream=True,
            )
            for chunk in stream:
                piece = chunk.choices[0].delta.content
                if piece:
                    answer_parts.append(piece)
                    yield piece
        except Exception:
            # Keep history consistent if the API call failed partway.
            self.messages.pop()
            raise

        answer = "".join(answer_parts)
        self.messages.append({"role": "assistant", "content": answer})
        self.turns.append((question, answer))
