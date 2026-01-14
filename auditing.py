from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .types import LeasedEvidence, ClaimAudit


@dataclass
class AuditConfig:
    tau: float = 0.85
    alpha: float = 0.5
    beta: float = 0.5


class CounterfactualAuditor:
    """Counterfactual Auditing (CA): minimal-support, necessity/sufficiency, and CFS."""
    def __init__(self, cfg: Optional[AuditConfig] = None):
        self.cfg = cfg or AuditConfig()

    @staticmethod
    def _posterior_from_likelihoods(likelihoods: List[float], prior: float = 0.5) -> float:
        p = prior
        for lk in likelihoods:
            num = lk * p
            den = num + (1.0 - lk) * (1.0 - p)
            p = num / (den + 1e-12)
        return p

    def minimal_support(self, evidence: List[LeasedEvidence], prior: float = 0.5) -> List[LeasedEvidence]:
        remaining = evidence[:]
        support: List[LeasedEvidence] = []
        current = prior

        while current < self.cfg.tau and remaining:
            best = None
            best_gain = -1e9
            for e in remaining:
                new = self._posterior_from_likelihoods([x.likelihood for x in support] + [e.likelihood], prior=prior)
                gain = new - current
                if gain > best_gain:
                    best_gain = gain
                    best = e
            if best is None or best_gain <= 0:
                break
            support.append(best)
            remaining.remove(best)
            current = self._posterior_from_likelihoods([x.likelihood for x in support], prior=prior)

        return support

    def audit_claim(self, claim: str, evidence: List[LeasedEvidence], prior: float = 0.5) -> ClaimAudit:
        posterior_full = self._posterior_from_likelihoods([e.likelihood for e in evidence], prior=prior)

        if not evidence:
            interval = (posterior_full, posterior_full)
        else:
            removals = []
            for i in range(len(evidence)):
                lks = [e.likelihood for j, e in enumerate(evidence) if j != i]
                removals.append(self._posterior_from_likelihoods(lks, prior=prior))
            interval = (min(removals + [posterior_full]), max(removals + [posterior_full]))

        ms = self.minimal_support(evidence, prior=prior)

        necessity: Dict[str, bool] = {}
        sufficiency: Dict[str, bool] = {}

        for e in ms:
            lks_removed = [x.likelihood for x in ms if x.chunk.chunk_id != e.chunk.chunk_id]
            post_removed = self._posterior_from_likelihoods(lks_removed, prior=prior)
            necessity[e.chunk.chunk_id] = (post_removed < self.cfg.tau)

        for e in ms:
            post_alone = self._posterior_from_likelihoods([e.likelihood], prior=prior)
            sufficiency[e.chunk.chunk_id] = (post_alone >= self.cfg.tau)

        nec_score = 0.0 if not necessity else sum(1.0 for v in necessity.values() if v) / len(necessity)
        suf_score = 0.0 if not sufficiency else sum(1.0 for v in sufficiency.values() if v) / len(sufficiency)
        cfs = self.cfg.alpha * nec_score + self.cfg.beta * suf_score

        ms_post = self._posterior_from_likelihoods([e.likelihood for e in ms], prior=prior) if ms else posterior_full

        return ClaimAudit(
            claim=claim,
            posterior=ms_post,
            interval=interval,
            minimal_support=ms,
            necessity=necessity,
            sufficiency=sufficiency,
            cfs=cfs,
        )
