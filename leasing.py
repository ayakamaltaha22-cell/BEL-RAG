from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import math

from .types import DocChunk, LeasedEvidence
from .retrieval import RetrievalResult


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class LeasingConfig:
    tau: float = 0.85            # confidence threshold (τ)
    c_max: float = 10.0          # cost budget (C_max)
    alpha_cost: float = 0.1      # trade-off weight in utility
    per_snippet_cost: float = 1.0
    max_steps: int = 20


class DefaultCalibrator:
    """Lightweight likelihood proxy mapping retrieval similarity -> pseudo-likelihood."""
    def __call__(self, query: str, chunk: DocChunk, retrieval_score: float) -> float:
        x = 8.0 * (retrieval_score - 0.35)  # center around ~0.35 similarity
        return float(sigmoid(x))


class EvidenceLeaser:
    """Sequential Bayesian Evidence Leasing (EL) with utility gating and stop criteria."""
    def __init__(self, config: Optional[LeasingConfig] = None, calibrator: Optional[Callable] = None):
        self.cfg = config or LeasingConfig()
        self.calibrator = calibrator or DefaultCalibrator()

    def _bayes_update(self, prior: float, likelihood: float) -> float:
        num = likelihood * prior
        den = num + (1.0 - likelihood) * (1.0 - prior)
        return num / (den + 1e-12)

    def lease(self, query: str, ranked: List[RetrievalResult], prior: float = 0.5) -> Tuple[List[LeasedEvidence], dict]:
        leased: List[LeasedEvidence] = []
        belief = prior
        cost = 0.0
        debug = {"belief_trace": [belief], "cost_trace": [cost]}

        for rr in ranked[: self.cfg.max_steps]:
            if belief >= self.cfg.tau:
                break
            if cost + self.cfg.per_snippet_cost > self.cfg.c_max:
                break

            lk = float(self.calibrator(query, rr.chunk, rr.score))
            new_belief = self._bayes_update(belief, lk)
            delta = new_belief - belief
            utility = delta - self.cfg.alpha_cost * self.cfg.per_snippet_cost

            if utility <= 0:
                continue  # do not lease

            cost += self.cfg.per_snippet_cost
            belief = new_belief
            leased.append(
                LeasedEvidence(
                    chunk=rr.chunk,
                    retrieval_score=rr.score,
                    likelihood=lk,
                    delta_belief=delta,
                    cost=self.cfg.per_snippet_cost,
                )
            )
            debug["belief_trace"].append(belief)
            debug["cost_trace"].append(cost)

        debug["final_belief"] = belief
        debug["final_cost"] = cost
        debug["tau"] = self.cfg.tau
        debug["c_max"] = self.cfg.c_max
        return leased, debug
