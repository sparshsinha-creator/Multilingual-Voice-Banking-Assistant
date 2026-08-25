"""
Phase 4 of 7 -- Agent

A LangGraph agent that routes each English-translated question to one of
two nodes:
  - "faq": general questions about a scheme/rule -- answered by the
    Phase 3 RAG pipeline (phase3_rag/query.py), reused as-is.
  - "eligibility": personal Atal Pension Yojana eligibility questions
    (e.g. "am I eligible", "can I join") -- the LLM extracts the four
    facts check_apy_eligibility() needs. A single confirmed disqualifier
    (age out of range, or a non-grandfathered taxpayer) is enough to
    answer "not eligible" without every field; anything else missing
    triggers a follow-up instead of guessing.

Exposes one entry point, answer_question(user_text), for Phase 5 to call.

Run with: python phase4_agent/agent.py "your question here"
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, TypedDict

# phase3_rag has no __init__.py, but Python treats it as an implicit
# namespace package once the repo root is on sys.path -- needed because
# running this file directly only puts phase4_agent/ on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer

from phase3_rag.query import EMBEDDING_MODEL, load_index
from phase3_rag.query import answer_question as rag_answer_question
from phase4_agent.tools import check_apy_eligibility

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROUTER_SYSTEM_PROMPT = """Classify the user's question about Indian government \
financial schemes into exactly one category. Reply with only one word: either \
"eligibility" or "faq".

- "eligibility": the user is asking about their OWN personal eligibility or \
ability to join/qualify for a scheme (e.g. "Am I eligible for Atal Pension \
Yojana?", "Can I join APY?", "Do I qualify?").
- "faq": any other general question about a scheme, rule, or regulation.
"""

EXTRACTION_SYSTEM_PROMPT = """Extract Atal Pension Yojana (APY) eligibility \
details from the user's question. Return ONLY a JSON object with these exact \
keys and nothing else:

- "age": integer, or null if not mentioned
- "has_bank_or_post_office_account": true/false, or null if not mentioned
- "is_income_tax_payer": true/false, or null if not mentioned
- "enrolled_on_or_before_30_sept_2022": true/false, or null if not mentioned \
-- whether the person already enrolled in APY on or before 30 September \
2022, as opposed to asking about newly joining now

Do not guess a value that isn't stated or clearly implied by the question.
"""

EXTRACTION_FIELDS = [
    "age",
    "has_bank_or_post_office_account",
    "is_income_tax_payer",
    "enrolled_on_or_before_30_sept_2022",
]

# Only these three are ever asked as a follow-up. Prior enrollment isn't a
# personal fact like age/account/tax status -- it's the framing of the
# question itself, and "am I eligible / can I join / do I qualify" phrasing
# defaults to a new-applicant reading unless the question says otherwise
# (see _resolve_eligibility).
FIELD_PROMPTS = {
    "age": "your age",
    "has_bank_or_post_office_account": "whether you have a bank or post office savings account",
    "is_income_tax_payer": "whether you currently pay income tax",
}


class AgentState(TypedDict, total=False):
    user_text: str
    route: str
    answer: str


_RESOURCES: Optional[tuple] = None


def _get_resources() -> tuple:
    """Lazily build and cache the Groq client + RAG index/model, shared by both nodes."""
    global _RESOURCES
    if _RESOURCES is None:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise SystemExit("GROQ_API_KEY not set -- copy .env.example to .env and add a key.")
        client = Groq(api_key=api_key)
        index, docs = load_index()
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
        _RESOURCES = (client, index, docs, embed_model)
    return _RESOURCES


def _classify_route(user_text: str, client: Groq) -> str:
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0,
        # openai/gpt-oss-120b is a reasoning model -- its chain-of-thought
        # counts against max_tokens before the final word is emitted, so
        # this needs real headroom even though the answer itself is one word.
        max_tokens=300,
    )
    label = (response.choices[0].message.content or "").strip().lower()
    return "eligibility" if "eligib" in label else "faq"


def _extract_eligibility_fields(user_text: str, client: Groq) -> dict:
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0,
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    return {key: data.get(key) for key in EXTRACTION_FIELDS}


def router_node(state: AgentState) -> dict:
    client, *_ = _get_resources()
    return {"route": _classify_route(state["user_text"], client)}


def faq_node(state: AgentState) -> dict:
    client, index, docs, embed_model = _get_resources()
    answer, _chunks = rag_answer_question(state["user_text"], client, index, docs, embed_model)
    return {"answer": answer}


def _resolve_eligibility(extraction: dict) -> tuple[Optional[dict], dict]:
    """Try to reach a verdict from whatever the question stated, without
    demanding every field: a single confirmed disqualifier (age out of
    range, or a taxpayer who isn't grandfathered in) already proves
    ineligibility no matter what the other fields are, so there's no need
    to ask about them. A positive "eligible" verdict still requires every
    relevant field to be confirmed, since nothing may be assumed in that
    direction. Returns (result, effective) -- result is a
    check_apy_eligibility-shaped dict once a verdict is certain, else None;
    effective carries the values (including the defaulted enrollment
    framing) used to work out what, if anything, still needs asking.
    """
    age = extraction.get("age")
    account = extraction.get("has_bank_or_post_office_account")
    tax_payer = extraction.get("is_income_tax_payer")
    enrolled = extraction.get("enrolled_on_or_before_30_sept_2022")
    if enrolled is None:
        enrolled = False

    effective = {
        "age": age,
        "has_bank_or_post_office_account": account,
        "is_income_tax_payer": tax_payer,
        "enrolled_on_or_before_30_sept_2022": enrolled,
    }

    if account is False:
        return {
            "eligible": False,
            "reason": "A bank or post office savings account is required for Atal Pension Yojana.",
        }, effective

    if not enrolled:
        if age is not None and not (18 <= age <= 40):
            return {
                "eligible": False,
                "reason": "Age must be between 18 and 40 (inclusive) to newly join Atal Pension Yojana.",
            }, effective
        if tax_payer is True:
            return {
                "eligible": False,
                "reason": (
                    "Income-tax payers are not permitted to newly join Atal Pension "
                    "Yojana on or after 1 October 2022."
                ),
            }, effective

    # Nothing disqualifies yet -- an "eligible" verdict needs every
    # relevant field confirmed (account, plus age+tax unless grandfathered).
    if account is True and (enrolled or (age is not None and tax_payer is not None)):
        result = check_apy_eligibility(
            age=age if age is not None else 0,
            has_bank_or_post_office_account=account,
            is_income_tax_payer=bool(tax_payer),
            enrolled_on_or_before_30_sept_2022=enrolled,
        )
        return result, effective

    return None, effective


def eligibility_node(state: AgentState) -> dict:
    client, *_ = _get_resources()
    extraction = _extract_eligibility_fields(state["user_text"], client)
    result, effective = _resolve_eligibility(extraction)

    if result is None:
        needed = []
        if effective["has_bank_or_post_office_account"] is None:
            needed.append("has_bank_or_post_office_account")
        if not effective["enrolled_on_or_before_30_sept_2022"]:
            if effective["age"] is None:
                needed.append("age")
            if effective["is_income_tax_payer"] is None:
                needed.append("is_income_tax_payer")
        answer = (
            "To check your Atal Pension Yojana eligibility, could you also tell me "
            + "; and ".join(FIELD_PROMPTS[key] for key in needed)
            + "?"
        )
        return {"answer": answer}

    verdict = "You are eligible" if result["eligible"] else "You are not eligible"
    return {"answer": f"{verdict} for Atal Pension Yojana. {result['reason']}"}


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("faq", faq_node)
    graph.add_node("eligibility", eligibility_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {"faq": "faq", "eligibility": "eligibility"},
    )
    graph.add_edge("faq", END)
    graph.add_edge("eligibility", END)
    return graph.compile()


_APP = _build_graph()


def answer_question(user_text: str) -> str:
    """Single entry point for Phase 5: route + answer an English-translated question."""
    result = _APP.invoke({"user_text": user_text})
    return result["answer"]


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python phase4_agent/agent.py "your question here"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    print(f"Question: {question}\n")
    print(f"Answer: {answer_question(question)}")


if __name__ == "__main__":
    main()
