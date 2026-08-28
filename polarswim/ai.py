"""Optional AI review of a workout, via the Claude API.

Two backends behind one interface, following the same split as the rest of the
project: a deterministic offline backend that needs no credentials and is what the
tests run against, and a real Claude backend. The app is fully usable without an
API key — `review` simply reports that the AI backend is not configured.

The key is read from the environment or a git-ignored `.env`, never hardcoded and
never committed.

What the model is given is deliberately narrow: the derived per-set table, the
learned parameters, and the honest uncertainty of each classification. It is asked
to reason about training structure, not to re-derive the numbers — and it is told
explicitly which labels are inferred rather than measured, so it does not present
a guess as fact.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MODEL = "claude-opus-5"
ENV_FILES = (Path.cwd() / ".env",
             Path.home() / ".polarswim" / ".env",
             Path.home() / "covered-call-app" / ".env")

SYSTEM = """You are a masters swim coach reviewing one practice.

You are given a table derived from wrist-sensor data: per-set length counts, pace
per 25 yards, heart rate above the swimmer's own resting baseline, and an INFERRED
stroke label. Understand clearly what the labels are:

- The sensor's manufacturer could not classify stroke at all; every length came
  back as "OTHER".
- Stroke labels here were inferred from pace and heart-rate cost alone. They are
  estimates with stated confidence, not measurements. Labels marked `undetermined`
  genuinely could not be resolved.
- Lengths marked `repaired` had a turn-detection defect corrected: either the
  sensor missed a wall and wrote two lengths as one, or it invented a wall and
  wrote one length as two. The correction is inferred, not measured.
- A set marked as a medley was identified by its repeating fly-back-breast-free
  structure, so those stroke labels are known rather than inferred.

Write a short review covering: the structure of the session and what it was
probably training, pacing consistency within sets, how heart rate responded and
recovered, and one or two specific things to work on. Be concrete and cite the
set numbers. Where a stroke label is low-confidence, say so rather than building
an argument on it. Do not invent data you were not given. Under 300 words."""


class AIError(RuntimeError):
    pass


@dataclass
class Review:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


def _load_env_key() -> str | None:
    """Resolve ANTHROPIC_API_KEY from the environment or a git-ignored .env."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    for path in ENV_FILES:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY"):
                _, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    return None


def available() -> bool:
    return _load_env_key() is not None


def build_prompt(header: dict, sets: list[dict], params: dict) -> str:
    """Render the derived facts the model reasons over."""
    lines = [
        f"Date: {str(header.get('start_time'))[:16]}",
        f"Distance: {round((header.get('distance_m') or 0) / 0.9144)} yd"
        f"   Duration: {round((header.get('duration_s') or 0) / 60)} min"
        f"   Avg HR: {header.get('avg_hr')}   Max HR: {header.get('max_hr')}",
        f"Pool: {header.get('pool_length_m')} m",
        "",
        "Sets (pace is seconds per 25 yd; hr_cost is bpm above this swimmer's baseline):",
        f"{'set':>4} {'n':>3} {'stroke':<13} {'conf':>5} {'pace':>6} {'hr_cost':>8} {'rest_before':>12} {'note':<10}",
    ]
    for s in sets:
        lines.append(
            f"{s['set_id']:>4} {s['n']:>3} {s['stroke']:<13} {s['confidence']:>5.2f} "
            f"{s['pace_s']:>6.1f} {s['hr_cost']:>8.1f} {s['rest_before_s']:>12.0f} "
            f"{s.get('note', ''):<10}")
    g = params.get("_global", {})
    if g:
        lines += ["", "This swimmer's own reference paces (s/25yd), learned from "
                  f"{int(g.get('n_obs', 0))} lengths across their history: "
                  f"p10={g.get('pace_p10', 0):.1f} p50={g.get('pace_p50', 0):.1f} "
                  f"p90={g.get('pace_p90', 0):.1f}"]
    return "\n".join(lines)


def review_offline(header: dict, sets: list[dict], params: dict) -> Review:
    """Deterministic fallback: a factual summary with no model call."""
    total = sum(s["n"] for s in sets)
    hard = [s for s in sets if s["hr_cost"] and s["hr_cost"] > 30]
    low = [s for s in sets if s["confidence"] < 0.4]
    fastest = min(sets, key=lambda s: s["pace_s"]) if sets else None
    parts = [
        f"{len(sets)} sets, {total} lengths ({total * 25} yd).",
        f"Hardest work: {len(hard)} set(s) above +30 bpm over baseline."
        if hard else "No set exceeded +30 bpm over baseline.",
    ]
    if fastest:
        parts.append(f"Fastest set was #{fastest['set_id']} at "
                     f"{fastest['pace_s']:.0f}s per 25.")
    if low:
        parts.append(f"{len(low)} set(s) had low-confidence stroke labels and "
                     "should not be read as definite.")
    parts.append("(Offline summary — set ANTHROPIC_API_KEY for a full AI review.)")
    return Review(text=" ".join(parts), model="offline")


def review(header: dict, sets: list[dict], params: dict,
           model: str = MODEL) -> Review:
    """Ask Claude to review the session. Falls back offline without a key."""
    key = _load_env_key()
    if not key:
        return review_offline(header, sets, params)

    try:
        import anthropic
    except ImportError as e:
        raise AIError("the `anthropic` package is required for AI review; "
                      "pip install anthropic") from e

    client = anthropic.Anthropic(api_key=key, timeout=180.0)
    prompt = build_prompt(header, sets, params)

    # Streaming because adaptive thinking plus a long system prompt can otherwise
    # push a single request past the HTTP timeout.
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise AIError("the model declined to answer this request")

    text = "\n".join(b.text for b in message.content if b.type == "text").strip()
    return Review(text=text, model=model,
                  input_tokens=message.usage.input_tokens,
                  output_tokens=message.usage.output_tokens)
