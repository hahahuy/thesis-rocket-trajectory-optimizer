# Evaluation Report Generation Guide

This guide explains how to generate all figures, tables, and metrics required for Section 3 of the evaluation report.

## Overview

The evaluation report generation system creates:

- **11 Figures** (3.1-3.11): Trajectory reconstructions, error plots, and physics residuals
- **2 Tables** (3.1-3.2): RMSE metrics and summary statistics
- **Evaluation Setup Info**: Test set size, time horizon, sampling rate, metrics
- **Complete Metrics**: Comprehensive evaluation metrics in JSON format

## Quick Start

### 1. Generate the Report

Run the evaluation script with your trained model checkpoint:

```bash
python scripts/generate_evaluation_report.py \
    --checkpoint experiments/exp15_02_12_direction_an_baseline/checkpoints/best.pt \
    --data_dir data/processed \
    --output_dir experiments/exp15_02_12_direction_an_baseline/evaluation_report
```

### 2. View the Results

All generated files will be in the `evaluation_report` directory:

```
evaluation_report/
├── figures/
│   ├── figure_3_1_vertical_position.png
│   ├── figure_3_2_vertical_velocity.png
│   ├── figure_3_3_mass_evolution.png
│   ├── figure_3_4_horizontal_position.png
│   ├── figure_3_5_quaternion_norm.png
│   ├── figure_3_6_error_vertical_position.png
│   ├── figure_3_7_error_vertical_velocity.png
│   ├── figure_3_8_error_mass.png
│   ├── figure_3_9_rmse_distribution.png
│   ├── figure_3_10_physics_residuals.png
│   └── figure_3_11_mass_monotonicity.png
├── tables/
│   ├── table_3_1_rmse_per_state.csv
│   ├── table_3_1_rmse_per_state.tex
│   ├── table_3_2_summary.csv
│   └── table_3_2_summary.tex
├── evaluation_setup.json
└── metrics.json
```

### 3. Access the Index

See `docs/EVALUATION_REPORT_INDEX.md` for hyperlinks to all generated items.

## Command-Line Options

```bash
python scripts/generate_evaluation_report.py \
    --checkpoint PATH_TO_CHECKPOINT.pt \    # Required: Path to model checkpoint
    --data_dir data/processed \            # Optional: Data directory (default: data/processed)
    --output_dir PATH_TO_OUTPUT \           # Optional: Output directory (default: experiment_dir/evaluation_report)
    --config PATH_TO_CONFIG.yaml \          # Optional: Explicit config path
    --device auto \                         # Optional: Device (auto/cpu/cuda, default: auto)
    --batch_size 8 \                       # Optional: Batch size (default: 8)
    --num_workers 0                         # Optional: Number of workers (default: 0)
```

## Generated Content

### Section 3.1: Evaluation Setup

**File:** `evaluation_setup.json`

Contains:
- Test set size (number of trajectories)
- Time horizon (0–30 s)
- Sampling rate (1501 steps)
- Metrics used (RMSE, mean ± SD)

### Section 3.2: Trajectory Reconstruction Results

**Figures:**
- **Figure 3.1:** Vertical position trajectory (z vs time)
- **Figure 3.2:** Vertical velocity trajectory (vz vs time)
- **Figure 3.3:** Mass evolution (m vs time)
- **Figure 3.4:** Horizontal position components (x, y vs time)
- **Figure 3.5:** Quaternion norm over time

### Section 3.3: Error Trajectories

**Figures:**
- **Figure 3.6:** Absolute error in vertical position
- **Figure 3.7:** Absolute error in vertical velocity
- **Figure 3.8:** Absolute error in mass

### Section 3.4: Aggregated Quantitative Metrics

**Table 3.1:** RMSE per state variable (CSV and LaTeX formats)
**Figure 3.9:** RMSE distribution across test trajectories (boxplot)

### Section 3.5: Physics and Constraint Residuals

**Figures:**
- **Figure 3.10:** Physics residual magnitudes over time
- **Figure 3.11:** Mass monotonicity check (dm/dt)

### Section 3.6: Summary Table

**Table 3.2:** Summary of key observed results (CSV and LaTeX formats)

## Technical Details

### Representative Trajectory

Figures 3.1-3.8 and 3.10-3.11 use the first trajectory in the test set (index 0) as the representative case. This ensures consistency across all trajectory plots.

### Physics Residuals

Physics residuals are computed using `PhysicsResidualLayer`, which:
1. Computes time derivatives using autograd
2. Evaluates the dynamics function f(s, u, p)
3. Computes residual: r = ds/dt - f(s, u, p)

### Mass Monotonicity

Mass monotonicity is checked by computing dm/dt using finite differences. The plot shows:
- Predicted dm/dt
- Reference dm/dt
- Count of violations (dm/dt > 0)

### RMSE Computation

RMSE is computed per state component across all trajectories and time steps:
- Per-trajectory RMSE: `sqrt(mean((pred - true)^2, axis=time))`
- Mean RMSE: `mean(per_trajectory_RMSE, axis=trajectories)`
- Standard deviation: `std(per_trajectory_RMSE, axis=trajectories)`

## Troubleshooting

### Physics Residuals Fail

If physics residual computation fails (Figure 3.10), the script will:
1. Print a warning
2. Generate a placeholder figure with error message
3. Continue with other figures

This can happen if:
- Physics parameters are missing
- Model architecture doesn't support autograd for time derivatives
- Device mismatch issues

### Missing Config

The script searches for config files in this order:
1. Explicit `--config` path
2. `experiment_dir/logs/config.yaml`
3. `experiment_dir/config.yaml`
4. `checkpoint_dir/config.yaml`

If no config is found, the script will fail with a clear error message.

### Memory Issues

If you encounter memory issues:
- Reduce `--batch_size` (default: 8)
- Process fewer trajectories by modifying the script
- Use CPU device: `--device cpu`

## Integration with LaTeX

Tables are generated in both CSV and LaTeX formats:

```latex
% Include Table 3.1
\input{evaluation_report/tables/table_3_1_rmse_per_state.tex}

% Include Table 3.2
\input{evaluation_report/tables/table_3_2_summary.tex}
```

Figures can be included directly:

```latex
% Include Figure 3.1
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{evaluation_report/figures/figure_3_1_vertical_position.png}
    \caption{Vertical Position Trajectory}
    \label{fig:3.1}
\end{figure}
```

## Notes

- All figures are saved at 300 DPI for publication quality
- Tables use 6 decimal places for precision
- The script automatically handles model architecture differences
- Evaluation is performed in eval mode (no gradient computation)

## Support

For issues or questions:
1. Check the error messages in the console output
2. Verify that your checkpoint and data directories are correct
3. Ensure all dependencies are installed
4. Check that the model architecture matches the checkpoint

