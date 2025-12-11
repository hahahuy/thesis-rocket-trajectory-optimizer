"""
Lightweight weight sweep helper for Direction AN/AN2 using PINNLoss (v1).

Usage pattern:
    from src.train.weight_sweep_an import WEIGHT_PRESETS, run_weight_sweep
    best_cfg, results = run_weight_sweep(train_eval_fn, presets=WEIGHT_PRESETS)

`train_eval_fn` should:
    - take a loss_config dict
    - train/evaluate a model (AN or AN2) with those weights
    - return a scalar validation metric (lower is better), and optionally logs

The helper returns (best_config, all_results) so you can reuse the best weights.
"""

from typing import Callable, Dict, List, Tuple, Any

# Preset list keeps the sweep manual but automated: adjust or extend as needed.
WEIGHT_PRESETS: List[Dict[str, Any]] = [
    {
        "name": "baseline_vertical",
        "lambda_data": 1.0,
        "lambda_phys": 0.3,
        "lambda_bc": 1.0,
        "component_weights": {
            "z": 2.0,
            "vz": 2.0,
            "m": 2.0,
            "x": 0.7,
            "y": 0.7,
            "vx": 0.7,
            "vy": 0.7,
            "q0": 0.8,
            "q1": 0.8,
            "q2": 0.8,
            "q3": 0.8,
            "wx": 0.8,
            "wy": 0.8,
            "wz": 0.8,
        },
        "lambda_quat_norm": 0.2,
        "lambda_mass_flow": 0.2,
        "lambda_translation": 1.0,
        "lambda_rotation": 0.8,
        "lambda_mass": 1.5,
    },
    {
        "name": "physics_heavier",
        "lambda_data": 1.0,
        "lambda_phys": 0.6,
        "lambda_bc": 1.0,
        "component_weights": {
            "z": 2.5,
            "vz": 2.5,
            "m": 2.5,
            "x": 0.5,
            "y": 0.5,
            "vx": 0.5,
            "vy": 0.5,
            "q0": 0.7,
            "q1": 0.7,
            "q2": 0.7,
            "q3": 0.7,
            "wx": 0.7,
            "wy": 0.7,
            "wz": 0.7,
        },
        "lambda_quat_norm": 0.3,
        "lambda_mass_flow": 0.3,
        "lambda_translation": 1.0,
        "lambda_rotation": 0.7,
        "lambda_mass": 1.8,
    },
    {
        "name": "data_heavier",
        "lambda_data": 1.0,
        "lambda_phys": 0.1,
        "lambda_bc": 1.0,
        "component_weights": {
            "z": 3.0,
            "vz": 3.0,
            "m": 3.0,
            "x": 0.7,
            "y": 0.7,
            "vx": 0.7,
            "vy": 0.7,
            "q0": 0.7,
            "q1": 0.7,
            "q2": 0.7,
            "q3": 0.7,
            "wx": 0.7,
            "wy": 0.7,
            "wz": 0.7,
        },
        "lambda_quat_norm": 0.1,
        "lambda_mass_flow": 0.2,
        "lambda_translation": 1.0,
        "lambda_rotation": 0.9,
        "lambda_mass": 2.0,
    },
]


def run_weight_sweep(
    train_eval_fn: Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]],
    presets: List[Dict[str, Any]] = WEIGHT_PRESETS,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Execute a small deterministic sweep over weight presets.

    Args:
        train_eval_fn: function that takes a loss_config dict, trains/evaluates,
                       and returns (val_metric, info_dict).
        presets: list of loss_config dicts to evaluate.

    Returns:
        best_config: the preset dict with the lowest val_metric (includes its score).
        results: list of dicts with {"name", "val_metric", "config", **info}
    """
    results: List[Dict[str, Any]] = []
    for preset in presets:
        val_metric, info = train_eval_fn(preset)
        results.append(
            {
                "name": preset.get("name", "preset"),
                "val_metric": val_metric,
                "config": preset,
                **(info or {}),
            }
        )

    best = min(results, key=lambda r: r["val_metric"])
    best_config = best["config"].copy()
    best_config["val_metric"] = best["val_metric"]
    best_config["name"] = best.get("name", "best")
    return best_config, results

