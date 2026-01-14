from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .types import BELRAGOutput, DocChunk
from .policy import PolicyEngine
from .retrieval import TfidfRetriever
from .leasing import EvidenceLeaser, LeasingConfig
from .auditing import CounterfactualAuditor, AuditConfig
from .generation import SimpleGenerator, GenerationConfig


@dataclass
class BELRAGConfig:
    # PAR
    prefer_peer_reviewed: bool = True
    forbid_pii: bool = True
    exclude_stale: bool = True
    max_ttl_days: Optional[int] = None
    allowed_licenses: Optional[List[str]] = None
    allowed_jurisdictions: Optional[List[str]] = None

    # Retrieval
    top_k: int = 15

    # EL
    leasing: LeasingConfig = LeasingConfig()

    # CA
    auditing: AuditConfig = AuditConfig()

    # CG
    generation: GenerationConfig = GenerationConfig()


class BELRAGPipeline:
    def __init__(self, corpus: Sequence[DocChunk], cfg: Optional[BELRAGConfig] = None):
        self.corpus = list(corpus)
        self.cfg = cfg or BELRAGConfig()

        self.policy = PolicyEngine()
        self.base_retriever = TfidfRetriever(self.corpus)
        self.leaser = EvidenceLeaser(self.cfg.leasing)
        self.auditor = CounterfactualAuditor(self.cfg.auditing)
        self.generator = SimpleGenerator(self.cfg.generation)

    def run(self, query: str) -> BELRAGOutput:
        directive = self.policy.profile_query(
            query=query,
            prefer_peer_reviewed=self.cfg.prefer_peer_reviewed,
            forbid_pii=self.cfg.forbid_pii,
            exclude_stale=self.cfg.exclude_stale,
            max_ttl_days=self.cfg.max_ttl_days,
            allowed_licenses=self.cfg.allowed_licenses,
            allowed_jurisdictions=self.cfg.allowed_jurisdictions,
        )
        admissible, par_log = self.policy.filter_admissible(self.corpus, directive)

        retriever = TfidfRetriever(admissible) if admissible else self.base_retriever
        ranked = retriever.search(query, top_k=self.cfg.top_k)

        leased, el_debug = self.leaser.lease(query, ranked, prior=0.5)

        draft = self.generator.draft_answer(query, leased)
        claims = self.generator.segment_claims(draft)

        audits = [self.auditor.audit_claim(c, leased, prior=0.5) for c in claims]
        report = self.generator.format_report(draft, audits)

        return BELRAGOutput(
            answer=report,
            claims=audits,
            leased=leased,
            policy_log={**directive.log, **par_log},
            debug={"el": el_debug},
        )
