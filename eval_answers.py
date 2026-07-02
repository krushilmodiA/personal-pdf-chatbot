"""LLM-as-judge evaluation: score the ANSWERS, not just the retrieval.

For each question this runs the full v3 pipeline (agentic retrieval + answer),
then a judge call rates the answer 1-5 on two axes, RAGAS-style:

  faithfulness - does the answer use only the retrieved excerpts?
  relevance    - does it actually answer the question asked?

Needs GROQ_API_KEY. Each question costs a few API calls, so start small:

    python eval_answers.py --pdf sample-manual.pdf --questions sample_questions.json --n 3
"""

import argparse
import hashlib
import json

from advanced_rag import smart_retrieve
from chatbot import PdfChatbot
from eval_retrieval import build_embedding_index
from rag_pipeline import HybridIndex, Reranker, TfidfIndex, chunk_pages, extract_pages

JUDGE_PROMPT = """\
You are grading a document-QA assistant. Be strict.

Question: {question}

Excerpts the assistant was allowed to use:
{excerpts}

The assistant's answer:
{answer}

Score two things from 1 (terrible) to 5 (perfect):
- faithfulness: every claim comes from the excerpts (saying "not covered" when
  the excerpts lack the answer is a 5).
- relevance: the answer actually addresses the question.

Reply in EXACTLY this format, nothing else:
faithfulness: <number>
relevance: <number>
comment: <one short sentence>"""


def parse_score(text: str, field: str) -> int:
    for line in text.splitlines():
        if line.lower().startswith(field):
            digits = [ch for ch in line if ch.isdigit()]
            if digits:
                return max(1, min(5, int(digits[0])))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--n", type=int, default=3, help="How many questions to judge")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    pages = extract_pages(args.pdf)
    chunks = chunk_pages(pages, source="eval")
    index = HybridIndex(TfidfIndex(chunks), build_embedding_index(chunks))
    reranker = Reranker()

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)[: args.n]

    total_f, total_r = 0, 0
    for item in questions:
        bot = PdfChatbot()  # fresh bot per question: no history bleed
        retrieved, _trace, _query = smart_retrieve(
            bot, index, reranker, item["question"], k=args.k
        )
        answer = "".join(bot.ask(item["question"], retrieved))
        excerpts = bot.format_excerpts(retrieved)

        judge = PdfChatbot()
        verdict = judge._one_shot(
            JUDGE_PROMPT.format(
                question=item["question"],
                excerpts=excerpts[:5000],
                answer=answer[:2000],
            ),
            max_tokens=120,
        )
        faith = parse_score(verdict, "faithfulness")
        rel = parse_score(verdict, "relevance")
        comment = next(
            (l.split(":", 1)[1].strip() for l in verdict.splitlines()
             if l.lower().startswith("comment")),
            "",
        )
        total_f += faith
        total_r += rel
        print(f"\nQ: {item['question']}")
        print(f"A: {answer[:180]}{'...' if len(answer) > 180 else ''}")
        print(f"   faithfulness {faith}/5 - relevance {rel}/5 - {comment}")

    n = len(questions)
    print(f"\n=== Averages over {n} question(s): "
          f"faithfulness {total_f / n:.1f}/5, relevance {total_r / n:.1f}/5 ===")


if __name__ == "__main__":
    main()
