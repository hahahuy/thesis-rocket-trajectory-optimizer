import os
import glob


def test_generator_runs_minimal(tmp_path):
    # Just verify script runs and produces files with placeholder pipeline
    os.chdir(str(tmp_path))
    os.makedirs("configs", exist_ok=True)
    with open("configs/dataset.yaml", "w") as f:
        f.write(
            """
            dataset:
              n_train: 2
              n_val: 1
              n_test: 1
              sampler: lhs
              seed: 1
              time_horizon_s: 1.0
              grid_hz: 2
              retries_per_case: 0
              parallel_workers: 1
              store_format: hdf5
            params:
              m0: [1.0, 2.0]
              Isp: [1.0, 2.0]
            constraints:
              qmax: 1.0
              nmax: 1.0
            scaling: use_scales_yaml
            ocp:
              kkt_tol: 1e-6
              max_iter: 1
              mesh_points: 5
              warm_start: true
            """
        )
    import subprocess
    subprocess.check_call(["python", "-m", "src.data.generator", "--config", "configs/dataset.yaml"])  # noqa: E501
    files = glob.glob("data/raw/case_*_*.h5")
    assert len(files) == 4
