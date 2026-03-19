#!/usr/bin/env python3

# Prepare and save a deterministic train/val split ONCE for track-regression branch graphs.
"""
Example:
python TrackGraph_splitter.py \
  --data-glob "./data/track_graphs_pu0_part*.h5" \
  --val-fraction 0.1 \
  --seed 12345 \
  --out split_track_graphs_pu0_seed12345.npz

Then in training, load that .npz and build Subset(ds, train_idx/val_idx).
"""

import argparse
import glob
import os
from pathlib import Path
import atexit

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------
# Dataset
# ----------------------------

class H5TrackGraphDataset(Dataset):
    """
    Loads branch-graphs from multiple H5 part files.

    Expected structure:
      /graphs/<id>/x
      /graphs/<id>/edge_index
      /graphs/<id>/edge_attr
      /graphs/<id>/y_track     # shape (3,) = [pt_over_q, eta, phi]

    Worker-safe optimization: keep h5py.File handles OPEN per DataLoader worker.
    """
    def __init__(self, h5_paths):
        self.h5_paths = list(h5_paths)
        if not self.h5_paths:
            raise ValueError("No H5 files provided.")

        # Build global index: dataset_idx -> (file_idx, graph_key)
        self.index = []
        for fi, p in enumerate(self.h5_paths):
            with h5py.File(p, "r") as f:
                if "graphs" not in f:
                    continue
                keys = list(f["graphs"].keys())
                keys.sort()
                for k in keys:
                    self.index.append((fi, k))

        if not self.index:
            raise ValueError("No graphs found in provided H5 files.")

        # Per-worker cache
        self._files = None
        self._pid = None

    def __len__(self):
        return len(self.index)

    def _ensure_open(self):
        pid = os.getpid()
        if self._files is not None and self._pid == pid:
            return

        self._close_files()
        self._pid = pid
        self._files = [h5py.File(p, "r") for p in self.h5_paths]
        atexit.register(self._close_files)

    def _close_files(self):
        if self._files is None:
            return
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass
        self._files = None

    def __getitem__(self, idx):
        self._ensure_open()

        fi, k = self.index[idx]
        f = self._files[fi]
        g = f["graphs"][k]

        x = torch.from_numpy(g["x"][...]).float()
        edge_index = torch.from_numpy(g["edge_index"][...]).long()
        edge_attr = torch.from_numpy(g["edge_attr"][...]).float()

        if "y_track" not in g:
            raise RuntimeError(
                f"Missing 'y_track' in {self.h5_paths[fi]} /graphs/{k}. "
                f"Please update your converter to store y_track = [pt_over_q, eta, phi]."
            )
        y_track = torch.from_numpy(g["y_track"][...]).float()

        sample = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y_track": y_track,
        }

        # Optional metadata if present
        if "original_node_ids" in g:
            sample["original_node_ids"] = torch.from_numpy(g["original_node_ids"][...]).long()

        return sample


# ----------------------------
# Split preparation
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-glob",
        required=True,
        help='Glob for H5 parts, e.g. "./data/track_graphs_pu0_part*.h5"',
    )
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument(
        "--out",
        required=True,
        help="Output split file (.npz). Will contain train_idx, val_idx, and metadata.",
    )
    ap.add_argument(
        "--max-train-graphs",
        type=int,
        default=-1,
        help="Optional cap for train size (debug). -1=all.",
    )
    args = ap.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        raise SystemExit(f"--val-fraction must be in (0,1), got {args.val_fraction}")

    paths = sorted(glob.glob(args.data_glob))
    if not paths:
        raise SystemExit(f"No H5 files matched: {args.data_glob}")

    ds = H5TrackGraphDataset(paths)
    n = len(ds)
    if n < 2:
        raise SystemExit(f"Not enough graphs to split: n={n}")

    # Deterministic split
    rng = np.random.RandomState(args.seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)

    n_val = max(1, int(args.val_fraction * n))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    if args.max_train_graphs > 0:
        train_idx = train_idx[:args.max_train_graphs]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    abs_paths = np.array([str(Path(p).resolve()) for p in paths], dtype=object)

    np.savez(
        out_path,
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        seed=np.int64(args.seed),
        val_fraction=np.float64(args.val_fraction),
        data_glob=np.array(args.data_glob, dtype=object),
        h5_paths=abs_paths,
        n_graphs=np.int64(n),
    )

    print(f"[ok] wrote split: {out_path}")
    print(f"[i] graphs: total={n} train={len(train_idx)} val={len(val_idx)}")
    print(f"[i] seed={args.seed} val_fraction={args.val_fraction}")


if __name__ == "__main__":
    main()
