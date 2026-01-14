from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .types import DocChunk, PolicyMetadata, RetrievalDirective


def _parse_date(d: Optional[str]) -> Optional[_dt.date]:
    if not d:
        return None
    try:
        return _dt.date.fromisoformat(d)
    except Exception:
        return None


@dataclass
class PolicyRule:
    """A policy predicate over PolicyMetadata. If predicate returns False, the chunk is inadmissible."""
    name: str
    predicate: Callable[[PolicyMetadata, RetrievalDirective], bool]


class PolicyEngine:
    """Policy-Aware Retrieval (PAR): profile query + filter inadmissible evidence BEFORE leasing."""
    def __init__(self, rules: Optional[List[PolicyRule]] = None):
        self.rules = rules or self._default_rules()

    def profile_query(
        self,
        query: str,
        prefer_peer_reviewed: bool = True,
        forbid_pii: bool = True,
        exclude_stale: bool = True,
        max_ttl_days: Optional[int] = None,
        allowed_licenses: Optional[List[str]] = None,
        allowed_jurisdictions: Optional[List[str]] = None,
    ) -> RetrievalDirective:
        # Heuristic classifier (replace with a real classifier in production)
        q = query.lower()
        sensitive_terms = ["patient", "diagnosis", "medical", "ssn", "passport", "credit card", "account number"]
        is_sensitive = any(t in q for t in sensitive_terms)

        directive = RetrievalDirective(
            prefer_source_classes=["peer_reviewed"] if prefer_peer_reviewed else [],
            exclude_stale=exclude_stale,
            max_ttl_days=max_ttl_days,
            forbid_pii=forbid_pii or is_sensitive,
            allowed_licenses=allowed_licenses,
            allowed_jurisdictions=allowed_jurisdictions,
            log={
                "query": query,
                "is_sensitive": is_sensitive,
                "prefer_peer_reviewed": prefer_peer_reviewed,
            },
        )
        return directive

    def is_admissible(self, chunk: DocChunk, directive: RetrievalDirective) -> Tuple[bool, Dict[str, str]]:
        reasons: Dict[str, str] = {}
        for rule in self.rules:
            ok = rule.predicate(chunk.metadata, directive)
            if not ok:
                reasons[rule.name] = "blocked"
        return (len(reasons) == 0), reasons

    def filter_admissible(self, chunks: List[DocChunk], directive: RetrievalDirective) -> Tuple[List[DocChunk], Dict]:
        kept, rejected = [], []
        for c in chunks:
            ok, reasons = self.is_admissible(c, directive)
            if ok:
                kept.append(c)
            else:
                rejected.append({"doc_id": c.doc_id, "chunk_id": c.chunk_id, "reasons": reasons})
        log = {
            "kept": len(kept),
            "rejected": len(rejected),
            "rejected_details": rejected[:50],  # cap
        }
        return kept, log

    @staticmethod
    def _default_rules() -> List[PolicyRule]:
        def license_rule(meta: PolicyMetadata, d: RetrievalDirective) -> bool:
            if d.allowed_licenses is None:
                return True
            return meta.license in d.allowed_licenses

        def jurisdiction_rule(meta: PolicyMetadata, d: RetrievalDirective) -> bool:
            if d.allowed_jurisdictions is None:
                return True
            return (meta.jurisdiction or "unknown") in d.allowed_jurisdictions

        def pii_rule(meta: PolicyMetadata, d: RetrievalDirective) -> bool:
            if not d.forbid_pii:
                return True
            return not meta.contains_pii

        def stale_rule(meta: PolicyMetadata, d: RetrievalDirective) -> bool:
            if not d.exclude_stale:
                return True
            if meta.ttl_days is None and d.max_ttl_days is None:
                return True
            ttl = d.max_ttl_days if d.max_ttl_days is not None else meta.ttl_days
            if ttl is None:
                return True

            created = _parse_date(meta.created_at)
            if created is None:
                return True
            age_days = (_dt.date.today() - created).days
            return age_days <= ttl

        def source_class_rule(meta: PolicyMetadata, d: RetrievalDirective) -> bool:
            if not d.restrict_to_source_classes:
                return True
            return meta.source_class in d.restrict_to_source_classes

        return [
            PolicyRule("license", license_rule),
            PolicyRule("jurisdiction", jurisdiction_rule),
            PolicyRule("pii", pii_rule),
            PolicyRule("staleness_ttl", stale_rule),
            PolicyRule("source_class", source_class_rule),
        ]
