"""
Canonical serialisation for squad commitment hashing.

Single source of truth for how squad_rows are serialised before hashing.
Changing CANONICAL invalidates every historical hash in the ledger —
treat it as frozen once any production GW is locked.
"""

import hashlib
import json

# Frozen-forever: sort_keys for deterministic key order; compact separators
# to eliminate whitespace ambiguity; ensure_ascii pinned explicitly so a
# future interpreter change cannot silently diverge.
CANONICAL = dict(sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def compute_squad_hash(squad_rows: list[dict]) -> str:
    """SHA-256 of the canonical squad JSON. Full 64 hex chars — no truncation."""
    return hashlib.sha256(
        json.dumps(squad_rows, **CANONICAL).encode()
    ).hexdigest()
