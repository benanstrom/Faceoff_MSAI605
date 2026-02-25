from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=indent, sort_keys=True), encoding="utf-8")


def stable_hash_to_unit_interval(s: str, seed: int, algo: str = "sha256") -> float:
    """
    Deterministic hash -> float in [0,1).
    Uses seed as a prefix so you can change the split deterministically.
    """
    h = hashlib.new(algo)
    h.update(f"{seed}|{s}".encode("utf-8"))
    digest = h.digest()
    x = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (x % (10**12)) / float(10**12)
