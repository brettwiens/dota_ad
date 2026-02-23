# chatgpt_explain.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass(frozen=True)
class PairExplainRequest:
    ability_a: str
    ability_b: str
    score: Optional[float] = None
    synergy: Optional[float] = None
    pair_winrate: Optional[float] = None
    notes: Optional[str] = None  # optional extra context from your app


def explain_pair(req: PairExplainRequest) -> str:
    """
    Synchronous call to OpenAI. Call this from a worker thread, not the UI thread.
    """
    client = OpenAI()  # reads OPENAI_API_KEY from env by default

    prompt = f"""
You are a Dota 2 Ability Draft analyst.

Explain why this ability pair can be powerful:
- Ability A: {req.ability_a}
- Ability B: {req.ability_b}

If provided, incorporate these metrics:
- score: {req.score}
- synergy: {req.synergy}
- pair winrate: {req.pair_winrate}

Optional notes/context:
{req.notes or "(none)"}

Output format:
1) Core interaction (2-4 bullets)
2) Ideal hero archetypes (1-2 bullets)
3) Timing and execution tips (2-4 bullets)
4) Counters and weaknesses (2-4 bullets)
Keep it concise and practical.
""".strip()

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )
    # The SDK returns a structured response; .output_text is the quick way to get final text.
    return resp.output_text.strip()