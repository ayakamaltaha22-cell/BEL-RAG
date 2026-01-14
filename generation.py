from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .types import ClaimAudit, LeasedEvidence


@dataclass
class GenerationConfig:
    max_claims: int = 6


class SimpleGenerator:
    """Offline generator (no external LLM) + structured calibrated report."""
    def __init__(self, cfg: Optional[GenerationConfig] = None):
        self.cfg = cfg or GenerationConfig()

    def draft_answer(self, query: str, leased: List[LeasedEvidence]) -> str:
        if not leased:
            return f"I could not find sufficient admissible evidence to answer: {query}"
        top = leased[: min(3, len(leased))]
        bullets = "\n".join([f"- {e.chunk.text.strip()[:240]}" for e in top])
        return f"Answer (evidence-grounded draft):\n{bullets}"

    def segment_claims(self, answer: str) -> List[str]:
        lines = [ln.strip("- ").strip() for ln in answer.splitlines() if ln.strip().startswith("-")]
        if not lines:
            import re
            lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        return lines[: self.cfg.max_claims]

    def format_report(self, answer: str, audits: List[ClaimAudit]) -> str:
        out = [answer, "", "Calibrated claim report:"]
        for i, a in enumerate(audits, 1):
            ms_ids = [f"{e.chunk.doc_id}:{e.chunk.chunk_id}" for e in a.minimal_support]
            out.append(
                f"{i}. {a.claim}\n"
                f"   Belief={a.posterior:.3f}  Interval=[{a.interval[0]:.3f},{a.interval[1]:.3f}]  CFS={a.cfs:.3f}\n"
                f"   Minimal-support={', '.join(ms_ids) if ms_ids else '∅'}"
            )
        return "\n".join(out)
