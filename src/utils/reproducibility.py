"""
Reproducibility utilities for logging git hash, versions, seeds, etc.
"""

import subprocess
import sys
import platform
from pathlib import Path
from datetime import datetime
import json
import random
import numpy as np
import torch


def get_git_hash(project_root=None):
    """Get current git commit hash."""
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    return 'unknown'


def get_git_status(project_root=None):
    """Get git status (dirty/clean)."""
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            is_dirty = len(result.stdout.strip()) > 0
            return 'dirty' if is_dirty else 'clean'
    except Exception:
        pass
    
    return 'unknown'


def get_package_versions():
    """Get versions of key packages."""
    versions = {}
    
    packages = ['casadi', 'numpy', 'scipy', 'matplotlib', 'pandas']
    
    for pkg in packages:
        try:
            mod = __import__(pkg)
            if hasattr(mod, '__version__'):
                versions[pkg] = mod.__version__
            else:
                versions[pkg] = 'unknown'
        except ImportError:
            versions[pkg] = 'not_installed'
        except Exception:
            versions[pkg] = 'unknown'
    
    return versions


def get_system_info():
    """Get system information."""
    return {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'python_executable': sys.executable
    }


def get_reproducibility_metadata(project_root=None, seed=None):
    """
    Get complete reproducibility metadata.
    
    Args:
        project_root: Project root directory
        seed: Random seed (if used)
    
    Returns:
        Dictionary with reproducibility information
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'git_hash': get_git_hash(project_root),
        'git_status': get_git_status(project_root),
        'package_versions': get_package_versions(),
        'system_info': get_system_info()
    }
    
    if seed is not None:
        metadata['random_seed'] = seed
    
    return metadata


def log_reproducibility_info(file_path=None, project_root=None, seed=None):
    """
    Log reproducibility information to file.
    
    Args:
        file_path: Path to JSON file (optional)
        project_root: Project root directory
        seed: Random seed (if used)
    
    Returns:
        Metadata dictionary
    """
    metadata = get_reproducibility_metadata(project_root, seed)
    
    if file_path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return metadata


def print_reproducibility_info(project_root=None, seed=None):
    """Print reproducibility information to console."""
    metadata = get_reproducibility_metadata(project_root, seed)
    
    print("Reproducibility Information:")
    print(f"  Git hash: {metadata['git_hash'][:12]} ({metadata['git_status']})")
    print(f"  Timestamp: {metadata['timestamp']}")
    print(f"  Python: {metadata['system_info']['python_version']}")
    
    print("  Package versions:")
    for pkg, version in metadata['package_versions'].items():
        print(f"    {pkg}: {version}")
    
    if seed is not None:
        print(f"  Random seed: {seed}")


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

