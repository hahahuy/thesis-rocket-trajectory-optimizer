import os
import argparse
import json
import traceback
from pathlib import Path

import torch
import yaml

# Suppress OMP warning on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.models import PINN, ResidualNet
from src.utils.loaders import create_dataloaders
from src.eval.visualize_pinn import (
    evaluate_model,
    plot_trajectory_comparison,
    plot_3d_trajectory,
)
from src.data.preprocess import load_scales


def safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _requires_initial_state(model) -> bool:
    return bool(getattr(model, "requires_initial_state", False))


def _forward_with_initial_state_if_needed(model, t, context, state_true):
    if _requires_initial_state(model):
        initial_state = state_true[:, 0, :]
        return model(t, context, initial_state)
    return model(t, context)


def create_model(model_cfg, context_dim):
    model_type = model_cfg.get("type", "pinn").lower()

    if model_type == "pinn":
        return PINN(
            context_dim=context_dim,
            n_hidden=safe_int(model_cfg.get("n_hidden", 6)),
            n_neurons=safe_int(model_cfg.get("n_neurons", 128)),
            activation=model_cfg.get("activation", "tanh"),
            fourier_features=safe_int(model_cfg.get("fourier_features", 8)),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05),
            context_embedding_dim=safe_int(model_cfg.get("context_embedding_dim", 16)),
        )

    if model_type == "direction_d":
        from src.models.direction_d_pinn import DirectionDPINN

        return DirectionDPINN(
            context_dim=context_dim,
            fourier_features=safe_int(model_cfg.get("fourier_features", 8)),
            context_embedding_dim=safe_int(model_cfg.get("context_embedding_dim", 32)),
            backbone_hidden_dims=model_cfg.get("backbone_hidden_dims", [256, 256, 256, 256]),
            head_g3_hidden_dims=model_cfg.get("head_g3_hidden_dims", [128, 64]),
            head_g2_hidden_dims=model_cfg.get("head_g2_hidden_dims", [256, 128, 64]),
            head_g1_hidden_dims=model_cfg.get("head_g1_hidden_dims", [256, 128, 128, 64]),
            activation=model_cfg.get("activation", "gelu"),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.0),
        )

    if model_type == "direction_d1":
        from src.models.direction_d_pinn import DirectionDPINN_D1

        return DirectionDPINN_D1(
            context_dim=context_dim,
            fourier_features=safe_int(model_cfg.get("fourier_features", 8)),
            context_embedding_dim=safe_int(model_cfg.get("context_embedding_dim", 32)),
            backbone_hidden_dims=model_cfg.get("backbone_hidden_dims", [256, 256, 256, 256]),
            head_g3_hidden_dims=model_cfg.get("head_g3_hidden_dims", [128, 64]),
            head_g2_hidden_dims=model_cfg.get("head_g2_hidden_dims", [256, 128, 64]),
            head_g1_hidden_dims=model_cfg.get("head_g1_hidden_dims", [256, 128, 128, 64]),
            activation=model_cfg.get("activation", "gelu"),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.0),
            integration_method=model_cfg.get("integration_method", "rk4"),
            use_physics_aware=bool(model_cfg.get("use_physics_aware", True)),
        )

    if model_type == "residual":
        return ResidualNet(
            context_dim=context_dim,
            n_hidden=safe_int(model_cfg.get("n_hidden", 6)),
            n_neurons=safe_int(model_cfg.get("n_neurons", 128)),
            activation=model_cfg.get("activation", "tanh"),
            fourier_features=safe_int(model_cfg.get("fourier_features", 8)),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05),
            context_embedding_dim=safe_int(model_cfg.get("context_embedding_dim", 16)),
        )

    if model_type == "latent_ode":
        from src.models.latent_ode import RocketLatentODEPINN

        return RocketLatentODEPINN(
            context_dim=context_dim,
            latent_dim=safe_int(model_cfg.get("latent_dim", 64)),
            context_embedding_dim=safe_int(model_cfg.get("context_embedding_dim", 64)),
            fourier_features=safe_int(model_cfg.get("fourier_features", 8)),
            dynamics_n_hidden=safe_int(model_cfg.get("dynamics_n_hidden", 3)),
            dynamics_n_neurons=safe_int(model_cfg.get("dynamics_n_neurons", 128)),
            decoder_n_hidden=safe_int(model_cfg.get("decoder_n_hidden", 3)),
            decoder_n_neurons=safe_int(model_cfg.get("decoder_n_neurons", 128)),
            activation=model_cfg.get("activation", "tanh"),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05),
        )

    if model_type == "sequence":
        from src.models.sequence_pinn import RocketSequencePINN

        return RocketSequencePINN(
            context_dim=context_dim,
            context_embedding_dim=safe_int(model_cfg.get("context_embedding_dim", 64)),
            fourier_features=safe_int(model_cfg.get("fourier_features", 8)),
            d_model=safe_int(model_cfg.get("d_model", 128)),
            n_layers=safe_int(model_cfg.get("n_layers", 4)),
            n_heads=safe_int(model_cfg.get("n_heads", 4)),
            dim_feedforward=safe_int(model_cfg.get("dim_feedforward", 512)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05),
            activation=model_cfg.get("transformer_activation", "gelu"),
        )

    if model_type == "hybrid":
        from src.models.hybrid_pinn import RocketHybridPINN

        return RocketHybridPINN(
            context_dim=context_dim,
            latent_dim=safe_int(model_cfg.get("latent_dim", 64)),
            context_embedding_dim=safe_int(model_cfg.get("context_embedding_dim", 64)),
            fourier_features=safe_int(model_cfg.get("fourier_features", 8)),
            d_model=safe_int(model_cfg.get("d_model", 128)),
            n_layers=safe_int(model_cfg.get("n_layers", 2)),
            n_heads=safe_int(model_cfg.get("n_heads", 4)),
            dim_feedforward=safe_int(model_cfg.get("dim_feedforward", 512)),
            encoder_window=safe_int(model_cfg.get("encoder_window", 10)),
            activation=model_cfg.get("activation", "tanh"),
            transformer_activation=model_cfg.get("transformer_activation", "gelu"),
            dynamics_n_hidden=safe_int(model_cfg.get("dynamics_n_hidden", 3)),
            dynamics_n_neurons=safe_int(model_cfg.get("dynamics_n_neurons", 128)),
            decoder_n_hidden=safe_int(model_cfg.get("decoder_n_hidden", 3)),
            decoder_n_neurons=safe_int(model_cfg.get("decoder_n_neurons", 128)),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05),
        )

    if model_type == "hybrid_c1":
        from src.models.hybrid_pinn import RocketHybridPINNC1

        return RocketHybridPINNC1(
            context_dim=context_dim,
            latent_dim=safe_int(model_cfg.get("latent_dim", 64)),
            context_embedding_dim=safe_int(model_cfg.get("context_embedding_dim", 32)),
            fourier_features=safe_int(model_cfg.get("fourier_features", 8)),
            d_model=safe_int(model_cfg.get("d_model", 128)),
            n_layers=safe_int(model_cfg.get("n_layers", 2)),
            n_heads=safe_int(model_cfg.get("n_heads", 4)),
            dim_feedforward=safe_int(model_cfg.get("dim_feedforward", 512)),
            encoder_window=safe_int(model_cfg.get("encoder_window", 10)),
            activation=model_cfg.get("activation", "tanh"),
            transformer_activation=model_cfg.get("transformer_activation", "gelu"),
            dynamics_n_hidden=safe_int(model_cfg.get("dynamics_n_hidden", 3)),
            dynamics_n_neurons=safe_int(model_cfg.get("dynamics_n_neurons", 128)),
            decoder_n_hidden=safe_int(model_cfg.get("decoder_n_hidden", 3)),
            decoder_n_neurons=safe_int(model_cfg.get("decoder_n_neurons", 128)),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05),
            debug_stats=bool(model_cfg.get("debug_stats", True)),
        )

    if model_type == "hybrid_c2":
        from src.models.hybrid_pinn import RocketHybridPINNC2

        return RocketHybridPINNC2(
            context_dim=context_dim,
            latent_dim=safe_int(model_cfg.get("latent_dim", 64)),
            fourier_features=safe_int(model_cfg.get("fourier_features", 8)),
            shared_stem_hidden_dim=safe_int(
                model_cfg.get("shared_stem_hidden_dim", 128)
            ),
            temporal_type=model_cfg.get("temporal_type", "transformer"),
            temporal_n_layers=safe_int(model_cfg.get("temporal_n_layers", 4)),
            temporal_n_heads=safe_int(model_cfg.get("temporal_n_heads", 4)),
            temporal_dim_feedforward=safe_int(
                model_cfg.get("temporal_dim_feedforward", 512)
            ),
            encoder_window=safe_int(model_cfg.get("encoder_window", 10)),
            translation_branch_dims=model_cfg.get(
                "translation_branch_dims", [128, 128]
            ),
            rotation_branch_dims=model_cfg.get("rotation_branch_dims", [256, 256]),
            mass_branch_dims=model_cfg.get("mass_branch_dims", [64]),
            activation=model_cfg.get("activation", "tanh"),
            transformer_activation=model_cfg.get("transformer_activation", "gelu"),
            dynamics_n_hidden=safe_int(model_cfg.get("dynamics_n_hidden", 3)),
            dynamics_n_neurons=safe_int(model_cfg.get("dynamics_n_neurons", 128)),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05),
            debug_stats=bool(model_cfg.get("debug_stats", True)),
        )

    if model_type == "hybrid_c3":
        from src.models.hybrid_pinn import RocketHybridPINNC3

        return RocketHybridPINNC3(
            context_dim=context_dim,
            latent_dim=safe_int(model_cfg.get("latent_dim", 64)),
            fourier_features=safe_int(model_cfg.get("fourier_features", 8)),
            shared_stem_hidden_dim=safe_int(
                model_cfg.get("shared_stem_hidden_dim", 128)
            ),
            temporal_type=model_cfg.get("temporal_type", "transformer"),
            temporal_n_layers=safe_int(model_cfg.get("temporal_n_layers", 4)),
            temporal_n_heads=safe_int(model_cfg.get("temporal_n_heads", 4)),
            temporal_dim_feedforward=safe_int(
                model_cfg.get("temporal_dim_feedforward", 512)
            ),
            encoder_window=safe_int(model_cfg.get("encoder_window", 10)),
            translation_branch_dims=model_cfg.get(
                "translation_branch_dims", [128, 128]
            ),
            rotation_branch_dims=model_cfg.get("rotation_branch_dims", [256, 256]),
            mass_branch_dims=model_cfg.get("mass_branch_dims", [64]),
            activation=model_cfg.get("activation", "tanh"),
            transformer_activation=model_cfg.get("transformer_activation", "gelu"),
            dynamics_n_hidden=safe_int(model_cfg.get("dynamics_n_hidden", 3)),
            dynamics_n_neurons=safe_int(model_cfg.get("dynamics_n_neurons", 128)),
            layer_norm=bool(model_cfg.get("layer_norm", True)),
            dropout=safe_float(model_cfg.get("dropout"), 0.05),
            debug_stats=bool(model_cfg.get("debug_stats", True)),
            use_physics_aware_translation=bool(
                model_cfg.get("use_physics_aware_translation", False)
            ),
            use_coordinated_branches=bool(
                model_cfg.get("use_coordinated_branches", False)
            ),
        )

    raise ValueError(
        f"Unsupported model type '{model_type}'. "
        "Supported: pinn, residual, latent_ode, sequence, hybrid, hybrid_c1, hybrid_c2, hybrid_c3, direction_d, direction_d1."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained experiment.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt).")
    parser.add_argument(
        "--data_dir", default="data/processed", help="Processed data directory."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Evaluation dataloader batch size."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the evaluation dataloader.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection policy.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional override for evaluation outputs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional explicit config path (otherwise inferred from experiment).",
    )
    parser.add_argument(
        "--time_subsample",
        type=int,
        default=None,
        help="Optional time subsampling factor for evaluation dataloader.",
    )
    parser.add_argument(
        "--max_plots",
        type=int,
        default=3,
        help="Maximum number of sample trajectories to plot.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable trajectory plotting.",
    )
    return parser.parse_args()


def resolve_config(args, logs_dir, experiment_dir, checkpoint_path):
    candidates = []
    if args.config:
        candidates.append(Path(args.config))
    candidates.extend(
        [
            logs_dir / "config.yaml",
            experiment_dir / "config.yaml",
            checkpoint_path.parent / "config.yaml",
        ]
    )

    for candidate in candidates:
        if candidate and candidate.exists():
            print(f"Loading config from {candidate}")
            with open(candidate, "r") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("Config not found for evaluation.")


def resolve_device(arg_choice: str):
    if arg_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()

    try:
        checkpoint_path = Path(args.checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = checkpoint["model_state_dict"]

        experiment_dir = checkpoint_path.parent.parent
        logs_dir = experiment_dir / "logs"

        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else experiment_dir / "figures"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

        config = resolve_config(args, logs_dir, experiment_dir, checkpoint_path)
        model_cfg = config.get("model", {})

        scales_path = config.get("scales_config", "configs/scales.yaml")
        scales = load_scales(scales_path)
        print(
            "Scales loaded: "
            f"L={scales.L}, V={scales.V}, T={scales.T}, M={scales.M}, "
            f"F={scales.F}, W={scales.W}"
        )

        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        print(f"Creating dataloaders from {data_dir}")
        train_loader, _, test_loader = create_dataloaders(
            data_dir=str(data_dir),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            time_subsample=args.time_subsample,
        )

        context_dim = train_loader.dataset.context_dim
        print(f"Context dimension: {context_dim}")
        print(f"Test dataset size: {len(test_loader.dataset)}")

        model = create_model(model_cfg, context_dim)
        model.load_state_dict(model_state)

        device = resolve_device(args.device)
        print(f"Using device: {device}")
        model.to(device)
        model.eval()

        print("\n" + "=" * 60)
        print("Evaluating on test set...")
        print("=" * 60)
        metrics = evaluate_model(model, test_loader, device, scales)

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")
        print(json.dumps(metrics, indent=2))

        if not args.no_plots:
            print("\n" + "=" * 60)
            print("Generating sample plots with improved visualization...")
            print("=" * 60)

            plot_count = 0
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    if i >= args.max_plots:
                        break

                    try:
                        t = batch["t"].to(device)
                        context = batch["context"].to(device)
                        state_true = batch["state"].to(device)

                        if t.dim() == 2:
                            t = t.unsqueeze(-1)

                        state_pred = _forward_with_initial_state_if_needed(
                            model, t, context, state_true
                        )

                        t_np = t[0].cpu().detach().squeeze(-1).numpy()
                        pred_np = state_pred[0].cpu().detach().numpy()
                        true_np = state_true[0].cpu().detach().numpy()

                        plot_path = output_dir / f"trajectory_case_{i}.png"
                        plot_trajectory_comparison(
                            t_np,
                            pred_np,
                            true_np,
                            scales,
                            save_path=str(plot_path),
                            title=f"Case {i} - Trajectory Comparison",
                        )

                        plot_3d_path = output_dir / f"trajectory_3d_case_{i}.png"
                        plot_3d_trajectory(
                            t_np,
                            pred_np,
                            true_np,
                            scales,
                            save_path=str(plot_3d_path),
                            title=f"Case {i} - 3D Trajectory Comparison",
                        )

                        plot_count += 1

                    except Exception as plot_err:  # pragma: no cover - diagnostic only
                        print(f"[WARNING] Plotting failed for case {i}: {plot_err}")
                        traceback.print_exc()

            print(f"Plots generated: {plot_count}/{args.max_plots}")

        print("\n" + "=" * 60)
        print("Evaluation complete!")
        print(f"  - Metrics saved to: {metrics_path}")
        print(f"  - Output directory: {output_dir}")
        print("=" * 60)
        return 0

    except Exception as exc:
        print(f"\nFatal error: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
