"""Canonical config hashing for run-id derivation (AC7, AC9)."""

import hashlib
import json


def config_hash(cfg_dict: dict) -> str:
    """
    Compute a stable 12-char hex hash of a parsed config dict.

    The hash is computed over the canonicalized JSON representation:
      json.dumps(cfg_dict, sort_keys=True, default=str)

    This makes the hash stable under key reordering and handles TOML
    datetimes (non-JSON-native values) via default=str.

    Returns a 12-character lowercase hex string.
    """
    canonical = json.dumps(cfg_dict, sort_keys=True, default=str).encode()
    return hashlib.sha256(canonical).hexdigest()[:12]
