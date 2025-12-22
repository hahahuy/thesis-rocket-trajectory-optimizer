# Evaluation Report Index

This document provides hyperlinks to all figures, tables, and reported results for Section 3 of the evaluation report.

**Note:** Run `scripts/generate_evaluation_report.py` with your checkpoint to generate all figures and tables. The script will create an `evaluation_report` directory in your experiment folder.

---

## 3.1 Evaluation Setup

**Text only, no figures.**

### Reported Items

- **Test set size:** See [evaluation_setup.json](#evaluation-setup-info) (number of trajectories)
- **Time horizon:** 0–30 s (see [evaluation_setup.json](#evaluation-setup-info))
- **Sampling rate:** 1501 steps (see [evaluation_setup.json](#evaluation-setup-info))
- **Metrics used:** RMSE, mean ± SD (see [evaluation_setup.json](#evaluation-setup-info))

**Source:** `evaluation_report/evaluation_setup.json`

---

## 3.2 Trajectory Reconstruction Results

### [Figure 3.1 – Vertical Position Trajectory (z vs time)](figures/figure_3_1_vertical_position.png)

**Type:** Line plot  
**Content:**
- Reference trajectory (ODE solver)
- PINN-predicted trajectory
- One representative test case

**File:** `evaluation_report/figures/figure_3_1_vertical_position.png`

**Text reference example:**
> Figure 3.1 shows the reference and predicted vertical position for a representative test trajectory over the 30 s launch phase.

---

### [Figure 3.2 – Vertical Velocity Trajectory (vz vs time)](figures/figure_3_2_vertical_velocity.png)

**Type:** Line plot  
**Content:**
- Reference vz
- Predicted vz
- Same trajectory as Figure 3.1

**File:** `evaluation_report/figures/figure_3_2_vertical_velocity.png`

---

### [Figure 3.3 – Mass Evolution (m vs time)](figures/figure_3_3_mass_evolution.png)

**Type:** Line plot  
**Content:**
- Reference mass
- Predicted mass

**File:** `evaluation_report/figures/figure_3_3_mass_evolution.png`

---

### [Figure 3.4 – Horizontal Position Components (x, y vs time)](figures/figure_3_4_horizontal_position.png)

**Type:** Line plot (two subplots)  
**Content:**
- Reference x, y
- Predicted x, y

**File:** `evaluation_report/figures/figure_3_4_horizontal_position.png`

---

### [Figure 3.5 – Quaternion Norm over Time](figures/figure_3_5_quaternion_norm.png)

**Type:** Line plot  
**Content:**
- Quaternion norm ‖q‖ for predicted trajectory
- Possibly overlay multiple test trajectories (thin lines)

**File:** `evaluation_report/figures/figure_3_5_quaternion_norm.png`

---

## 3.3 Error Trajectories

### [Figure 3.6 – Absolute Error in Vertical Position |z_pred − z_ref|](figures/figure_3_6_error_vertical_position.png)

**Type:** Line plot  
**Content:**
- Absolute error vs time
- One representative trajectory

**File:** `evaluation_report/figures/figure_3_6_error_vertical_position.png`

---

### [Figure 3.7 – Absolute Error in Vertical Velocity |vz_pred − vz_ref|](figures/figure_3_7_error_vertical_velocity.png)

**Type:** Line plot  
**Content:**
- Absolute error vs time
- One representative trajectory

**File:** `evaluation_report/figures/figure_3_7_error_vertical_velocity.png`

---

### [Figure 3.8 – Absolute Error in Mass |m_pred − m_ref|](figures/figure_3_8_error_mass.png)

**Type:** Line plot  
**Content:**
- Absolute error vs time
- One representative trajectory

**File:** `evaluation_report/figures/figure_3_8_error_mass.png`

---

## 3.4 Aggregated Quantitative Metrics

### [Table 3.1 – RMSE per State Variable (Test Set)](tables/table_3_1_rmse_per_state.csv)

**Type:** Table

**Columns:**
- State variable
- Mean RMSE
- Standard deviation
- Units or nondimensional indicator

**Rows:**
- x, y, z
- vx, vy, vz
- quaternion components or norm
- angular rates
- mass

**Files:**
- CSV: `evaluation_report/tables/table_3_1_rmse_per_state.csv`
- LaTeX: `evaluation_report/tables/table_3_1_rmse_per_state.tex`

**Text reference example:**
> Table 3.1 reports the mean RMSE and standard deviation computed over all test trajectories for each state variable.

---

### [Figure 3.9 – RMSE Distribution across Test Trajectories](figures/figure_3_9_rmse_distribution.png)

**Type:** Boxplot or violin plot  
**Content:**
- One box per state group:
  - position
  - velocity
  - mass
  - rotation

**File:** `evaluation_report/figures/figure_3_9_rmse_distribution.png`

---

## 3.5 Physics and Constraint Residuals

### [Figure 3.10 – Physics Residual Magnitudes over Time](figures/figure_3_10_physics_residuals.png)

**Type:** Line plot  
**Content:**
- Norm of physics residual (or key components)
- Representative trajectory

**File:** `evaluation_report/figures/figure_3_10_physics_residuals.png`

---

### [Figure 3.11 – Mass Monotonicity Check](figures/figure_3_11_mass_monotonicity.png)

**Type:** Line plot or histogram  
**Content:**
- dm/dt over time
- Or count of dm/dt > 0 violations (if zero, say so)

**File:** `evaluation_report/figures/figure_3_11_mass_monotonicity.png`

---

## 3.6 Summary Table (Optional but Recommended)

### [Table 3.2 – Summary of Key Observed Results](tables/table_3_2_summary.csv)

**Type:** Table

**Rows:**
- Vertical position accuracy
- Velocity accuracy
- Mass behavior
- Quaternion norm range
- Residual magnitude range

**Columns:**
- Metric
- Observed range
- Test set size

**Files:**
- CSV: `evaluation_report/tables/table_3_2_summary.csv`
- LaTeX: `evaluation_report/tables/table_3_2_summary.tex`

**No interpretation. No adjectives.**

---

## Additional Files

### [Evaluation Setup Information](evaluation_setup.json)

**File:** `evaluation_report/evaluation_setup.json`

Contains:
- Test set size (number of trajectories)
- Time horizon (0–30 s)
- Sampling rate (1501 steps)
- Metrics used (RMSE, mean ± SD)

---

### [Complete Metrics](metrics.json)

**File:** `evaluation_report/metrics.json`

Contains comprehensive evaluation metrics including:
- RMSE per component
- Aggregated RMSE (translation, rotation, mass)
- Quaternion norm statistics
- Delta state norm diagnostics

---

## Usage Instructions

1. **Generate the report:**
   ```bash
   python scripts/generate_evaluation_report.py \
       --checkpoint experiments/exp15_02_12_direction_an_baseline/checkpoints/best.pt \
       --data_dir data/processed \
       --output_dir experiments/exp15_02_12_direction_an_baseline/evaluation_report
   ```

2. **View the generated files:**
   - Figures: `evaluation_report/figures/`
   - Tables: `evaluation_report/tables/`
   - Metrics: `evaluation_report/metrics.json`
   - Setup info: `evaluation_report/evaluation_setup.json`

3. **Update this index:**
   - After generating the report, update the relative paths in this document to match your experiment directory structure.

---

## Notes

- All figures are saved at 300 DPI for publication quality
- Tables are available in both CSV (for data analysis) and LaTeX (for document inclusion) formats
- The representative trajectory used for Figures 3.1-3.8 and 3.10-3.11 is the first trajectory in the test set (index 0)
- Figure 3.5 shows up to 10 trajectories overlaid
- Figure 3.9 aggregates metrics across all test trajectories

