from __future__ import annotations

import hashlib
import random

MASTER_SEED = 42
STREAM_IDS = {
    "graph_updates": 0,
    "query_parameters": 1,
    "network_variability": 2,
    "policy_bootstrap": 16,
    "method_order": 4,
    "dataset_generation": 5,
    "bootstrap_statistics": 17,
    "calibration_parameters": 14,
    "evaluation_parameters": 15,
    "update_schedules": 12,
}


def component_seed(component: str, scope: str = "default") -> int:
    """Derive a stable uint32 seed from the locked master seed and stream."""
    if component not in STREAM_IDS:
        raise KeyError(f"Unregistered RNG component: {component}")
    payload = f"{MASTER_SEED}:{STREAM_IDS[component]}:{component}:{scope}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def rng_for(component: str, scope: str = "default") -> random.Random:
    """Return an isolated MT19937 generator; never use module-global random state."""
    return random.Random(component_seed(component, scope))
