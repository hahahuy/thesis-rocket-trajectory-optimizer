import numpy as np
from typing import Dict, Tuple


def lhs_sample(n: int, bounds: Dict[str, Tuple[float, float]], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keys = list(bounds.keys())
    d = len(keys)
    # Stratify [0,1] into n bins per dimension
    cut = (np.arange(n) + rng.random(n)) / n
    samples = np.zeros((n, d), dtype=float)
    for j, key in enumerate(keys):
        low, high = bounds[key]
        perm = rng.permutation(n)
        u = cut[perm]
        samples[:, j] = low + u * (high - low)
    return samples


def sobol_sample(n: int, bounds: Dict[str, Tuple[float, float]], seed: int) -> np.ndarray:
    try:
        from scipy.stats import qmc
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("scipy is required for sobol sampling") from exc
    keys = list(bounds.keys())
    d = len(keys)
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = sampler.random_base2(m=int(np.ceil(np.log2(n))))[:n]
    lows = np.array([bounds[k][0] for k in keys], dtype=float)
    highs = np.array([bounds[k][1] for k in keys], dtype=float)
    return qmc.scale(u, lows, highs)


def persist_samples_table(path: str, keys: list, samples: np.ndarray) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [dict(zip(keys, row.tolist())) for row in samples]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
