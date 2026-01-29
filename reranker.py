from __future__ import annotations
import re
from typing import Dict, Any, List

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\u0600-\u06FF\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def lexical_overlap(q: str, t: str) -> int:
    qn = _norm(q)
    tn = _norm(t)
    qset = set(qn.split())
    tset = set(tn.split())
    return len(qset.intersection(tset))

def rerank_hits(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for h in hits:
        text = (h.get("text") or "") + "\n" + (h.get("search_text") or "")
        ov = lexical_overlap(question, text)
        h["_lex_ov"] = ov
        # deterministic blended score
        sem = float(h.get("score", 0.0) or 0.0)
        h["_blend"] = (0.75 * sem) + (0.25 * min(12, ov) / 12.0)

    hits.sort(
        key=lambda x: (
            float(x.get("_blend", 0.0)),
            int(x.get("_lex_ov", 0)),
            int(x.get("doc_rank", 0) or 0),
            str(x.get("doc_name", "")),
            int(x.get("id", 0) or 0),
        ),
        reverse=True
    )
    return hits
