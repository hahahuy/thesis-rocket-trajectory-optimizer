from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import h5py
import numpy as np


@dataclass
class CasePaths:
    root: str
    split: str
    idx: int

    @property
    def h5_path(self) -> str:
        return os.path.join(self.root, f"case_{self.split}_{self.idx}.h5")


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_hdf5_case(path: str, payload: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write HDF5 structure
    with h5py.File(path, "w") as f:
        # Scalars and arrays
        time = payload["time"]
        state = payload["state"]
        control = payload["control"]
        f.create_dataset("time", data=time, dtype="f8")
        f.create_dataset("state", data=state, dtype="f8")
        f.create_dataset("control", data=control, dtype="f8")

        monitors = payload.get("monitors", {})
        g_mon = f.create_group("monitors")
        for k, v in monitors.items():
            g_mon.create_dataset(k, data=np.asarray(v), dtype="f8")

        ocp = payload.get("ocp", {})
        g_ocp = f.create_group("ocp")
        for k, v in ocp.items():
            g_ocp.create_dataset(k, data=np.asarray(v), dtype="f8")

        g_meta = f.create_group("meta")
        # NumPy 2.0 compatibility: use np.bytes_ instead of np.string_
        try:
            string_dtype = np.string_  # NumPy < 2.0
        except AttributeError:
            string_dtype = np.bytes_  # NumPy >= 2.0
        
        for k, v in metadata.items():
            if isinstance(v, (str, bytes)):
                if isinstance(v, str):
                    v_bytes = v.encode('utf-8')
                else:
                    v_bytes = v
                g_meta.create_dataset(k, data=np.array(v_bytes, dtype=string_dtype))
            elif isinstance(v, (dict, list)):
                # Serialize dicts/lists as JSON strings
                v_str = json.dumps(v)
                g_meta.create_dataset(k, data=np.array(v_str.encode('utf-8'), dtype=string_dtype))
            elif isinstance(v, (int, float, np.integer, np.floating)):
                g_meta.create_dataset(k, data=np.asarray(v, dtype=type(v)))
            else:
                # Try to convert to numpy array, fallback to JSON string
                try:
                    arr = np.asarray(v)
                    if arr.dtype == object:
                        v_str = json.dumps(v)
                        g_meta.create_dataset(k, data=np.array(v_str.encode('utf-8'), dtype=string_dtype))
                    else:
                        g_meta.create_dataset(k, data=arr)
                except (TypeError, ValueError):
                    v_str = json.dumps(v)
                    g_meta.create_dataset(k, data=np.array(v_str.encode('utf-8'), dtype=string_dtype))

    # Compute checksum over the file bytes
    with open(path, "rb") as fh:
        checksum = sha256_of_bytes(fh.read())

    with h5py.File(path, "a") as f:
        g_meta = f["meta"]
        # NumPy 2.0 compatibility
        try:
            string_dtype = np.string_
        except AttributeError:
            string_dtype = np.bytes_
        checksum_str = f"sha256:{checksum}".encode('utf-8')
        g_meta.create_dataset("checksum", data=np.array(checksum_str, dtype=string_dtype))

    return checksum


def write_npz_case(path: str, payload: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)
    with open(path, "rb") as fh:
        checksum = sha256_of_bytes(fh.read())
    sidecar = os.path.splitext(path)[0] + ".json"
    meta = dict(metadata)
    meta["checksum"] = f"sha256:{checksum}"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return checksum
