from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PolicyMetadata:
    """Policy metadata attached to each document chunk (license, TTL, privacy, jurisdiction, source class)."""
    license: str = "unknown"          # e.g., "internal", "cc-by", "proprietary"
    ttl_days: Optional[int] = None    # freshness TTL threshold for the chunk
    created_at: Optional[str] = None  # ISO date string: YYYY-MM-DD
    contains_pii: bool = False
    jurisdiction: Optional[str] = None  # e.g., "UAE", "US", "EU"
    source_class: str = "unknown"       # e.g., "peer_reviewed", "internal", "web"

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocChunk:
    """A retrievable unit (chunk) with text + metadata."""
    doc_id: str
    chunk_id: str
    text: str
    metadata: PolicyMetadata


@dataclass
class RetrievalDirective:
    """Output of PAR profiling: what to prefer/exclude during retrieval."""
    prefer_source_classes: List[str] = field(default_factory=list)
    restrict_to_source_classes: List[str] = field(default_factory=list)
    exclude_stale: bool = True
    max_ttl_days: Optional[int] = None
    forbid_pii: bool = True
    allowed_licenses: Optional[List[str]] = None
    allowed_jurisdictions: Optional[List[str]] = None
    log: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeasedEvidence:
    """Evidence item leased during EL with utilities and belief deltas."""
    chunk: DocChunk
    retrieval_score: float
    likelihood: float              # P(E | theta) proxy
    delta_belief: float            # change in posterior
    cost: float


@dataclass
class ClaimAudit:
    claim: str
    posterior: float
    interval: Tuple[float, float]
    minimal_support: List[LeasedEvidence]
    necessity: Dict[str, bool]
    sufficiency: Dict[str, bool]
    cfs: float


@dataclass
class BELRAGOutput:
    answer: str
    claims: List[ClaimAudit]
    leased: List[LeasedEvidence]
    policy_log: Dict[str, Any]
    debug: Dict[str, Any] = field(default_factory=dict)
