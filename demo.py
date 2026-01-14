from __future__ import annotations

from belrag.pipeline import BELRAGPipeline, BELRAGConfig
from belrag.types import DocChunk, PolicyMetadata


def build_toy_corpus():
    return [
        DocChunk(
            doc_id="doc1",
            chunk_id="c1",
            text="Ottawa is the capital city of Canada.",
            metadata=PolicyMetadata(
                license="cc-by",
                ttl_days=3650,
                created_at="2020-01-01",
                contains_pii=False,
                jurisdiction="CA",
                source_class="peer_reviewed",
            ),
        ),
        DocChunk(
            doc_id="doc2",
            chunk_id="c1",
            text="Toronto is the largest city in Canada by population.",
            metadata=PolicyMetadata(
                license="cc-by",
                ttl_days=3650,
                created_at="2020-01-01",
                contains_pii=False,
                jurisdiction="CA",
                source_class="peer_reviewed",
            ),
        ),
        DocChunk(
            doc_id="doc3",
            chunk_id="c1",
            text="This internal memo contains customer passport numbers and should never be retrieved.",
            metadata=PolicyMetadata(
                license="internal",
                ttl_days=30,
                created_at="2025-12-15",
                contains_pii=True,
                jurisdiction="UAE",
                source_class="internal",
            ),
        ),
    ]


def main():
    corpus = build_toy_corpus()
    cfg = BELRAGConfig(
        prefer_peer_reviewed=True,
        forbid_pii=True,
        exclude_stale=True,
        max_ttl_days=3650,
        allowed_licenses=["cc-by", "internal"],  # internal allowed, but PII still blocked
    )
    pipe = BELRAGPipeline(corpus, cfg=cfg)

    q = "What is the capital of Canada?"
    out = pipe.run(q)
    print(out.answer)
    print("\nPolicy log:", out.policy_log)
    print("\nLeased evidence:", [(e.chunk.doc_id, e.chunk.chunk_id, round(e.retrieval_score, 3), round(e.likelihood, 3)) for e in out.leased])


if __name__ == "__main__":
    main()
