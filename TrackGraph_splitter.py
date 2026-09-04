#!/usr/bin/env python3

# Prepare and save a deterministic train/val split ONCE for track-regression branch graphs.
"""
Example:
python TrackGraph_splitter.py \
  --data-glob "/eos/atlas/atlascerngroupdisk/det-muon/muonSW/data_ml/track_graphs_mu200/data_track_graphs_mu200_part*.h5" \
  --val-fraction 0.1 \
  --seed 12345 \
  --out /eos/atlas/atlascerngroupdisk/det-muon/muonSW/data_ml/track_graphs_mu200/split_track_graphs_mu200_seed12345.npz

Before creating the split, every matched H5 file is checked. Unreadable or
structurally invalid files are excluded and written to a removal manifest.

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


REQUIRED_GRAPH_DATASETS = ("x", "edge_index", "edge_attr", "y_track")


# ----------------------------
# H5 health checks
# ----------------------------

def _check_graph_structure(g, path, graph_key):
    """Validate the datasets and basic shape relationships of one graph."""
    missing = [name for name in REQUIRED_GRAPH_DATASETS if name not in g]
    if missing:
        raise ValueError(f"/graphs/{graph_key} is missing datasets: {missing}")

    x = g["x"]
    edge_index = g["edge_index"]
    edge_attr = g["edge_attr"]
    y_track = g["y_track"]

    if x.ndim != 2:
        raise ValueError(f"/graphs/{graph_key}/x has shape {x.shape}, expected 2D")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"/graphs/{graph_key}/edge_index has shape {edge_index.shape}, expected (2, E)"
        )
    if edge_attr.ndim != 2:
        raise ValueError(
            f"/graphs/{graph_key}/edge_attr has shape {edge_attr.shape}, expected 2D"
        )
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"/graphs/{graph_key}: edge_attr rows={edge_attr.shape[0]} "
            f"but edge_index edges={edge_index.shape[1]}"
        )
    if y_track.shape != (3,):
        raise ValueError(
            f"/graphs/{graph_key}/y_track has shape {y_track.shape}, expected (3,)"
        )


def check_h5_file(path, check_mode="structure"):
    """
    Check one H5 part and return its graph count.

    check_mode:
      open      - only open the file and locate /graphs
      structure - validate every graph's required datasets and shapes
      full      - structure checks plus read every required dataset
    """
    try:
        with h5py.File(path, "r") as f:
            if "graphs" not in f:
                raise ValueError("missing /graphs group")
            if not isinstance(f["graphs"], h5py.Group):
                raise ValueError("/graphs exists but is not an HDF5 group")

            graphs = f["graphs"]
            keys = sorted(graphs.keys())
            if not keys:
                raise ValueError("/graphs contains no graphs")

            if check_mode in ("structure", "full"):
                for graph_key in keys:
                    g = graphs[graph_key]
                    if not isinstance(g, h5py.Group):
                        raise ValueError(f"/graphs/{graph_key} is not an HDF5 group")
                    _check_graph_structure(g, path, graph_key)

                    if check_mode == "full":
                        # Force HDF5 to read all stored chunks. This can catch
                        # corruption that is not visible from metadata alone.
                        for dataset_name in REQUIRED_GRAPH_DATASETS:
                            _ = g[dataset_name][...]

            return len(keys)
    except Exception as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc


def scan_h5_files(paths, check_mode):
    """Return (healthy_paths, bad_records, graph_counts)."""
    healthy_paths = []
    bad_records = []
    graph_counts = {}

    print(f"[i] checking {len(paths)} H5 file(s), mode={check_mode}")
    for path in paths:
        try:
            n_graphs = check_h5_file(path, check_mode=check_mode)
        except Exception as exc:
            error = str(exc)
            bad_records.append((path, error))
            print(f"[BAD] {path} :: {error}")
        else:
            healthy_paths.append(path)
            graph_counts[path] = n_graphs
            print(f"[ok]  {path} :: graphs={n_graphs}")

    return healthy_paths, bad_records, graph_counts


def write_bad_file_manifest(manifest_path, bad_records):
    """Write bad H5 paths and errors to a text file suitable for review/removal."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as out:
        out.write("# Corrupted or invalid HDF5 files excluded from the split.\n")
        out.write("# Review these paths, then remove or regenerate them.\n")
        out.write("# Format: absolute_path<TAB>error\n")
        for path, error in bad_records:
            clean_error = error.replace("\n", " ").replace("\t", " ")
            out.write(f"{Path(path).resolve()}\t{clean_error}\n")

    return manifest_path


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
            try:
                with h5py.File(p, "r") as f:
                    keys = sorted(f["graphs"].keys())
                    for k in keys:
                        self.index.append((fi, k))
            except Exception as exc:
                raise RuntimeError(
                    f"H5 file became unreadable after the health check: {p} :: {exc}"
                ) from exc

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
    ap.add_argument(
        "--h5-check",
        choices=("open", "structure", "full"),
        default="structure",
        help=(
            "H5 health check before splitting: 'open' only opens each file; "
            "'structure' also validates every graph; 'full' reads every required dataset."
        ),
    )
    ap.add_argument(
        "--bad-files-out",
        default=None,
        help=(
            "Removal manifest for corrupted/invalid H5 files. "
            "Default: <out>.bad_h5.txt"
        ),
    )
    ap.add_argument(
        "--fail-on-bad-files",
        action="store_true",
        help="Write the bad-file manifest, then abort instead of excluding bad files.",
    )
    args = ap.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        raise SystemExit(f"--val-fraction must be in (0,1), got {args.val_fraction}")

    matched_paths = sorted(glob.glob(args.data_glob))
    if not matched_paths:
        raise SystemExit(f"No H5 files matched: {args.data_glob}")

    healthy_paths, bad_records, graph_counts = scan_h5_files(
        matched_paths,
        check_mode=args.h5_check,
    )

    bad_manifest_path = Path(args.bad_files_out or f"{args.out}.bad_h5.txt")
    if bad_records:
        write_bad_file_manifest(bad_manifest_path, bad_records)
        print(f"[!] marked {len(bad_records)} bad H5 file(s) in: {bad_manifest_path}")
        print("[!] bad files will not be included in the split")
        if args.fail_on_bad_files:
            raise SystemExit(
                f"Found {len(bad_records)} bad H5 file(s); see {bad_manifest_path}"
            )
    elif bad_manifest_path.exists():
        # Do not leave a stale manifest suggesting that currently healthy files are bad.
        bad_manifest_path.unlink()

    if not healthy_paths:
        raise SystemExit("No healthy H5 files remain after validation.")

    print(
        f"[i] H5 summary: matched={len(matched_paths)} "
        f"healthy={len(healthy_paths)} bad={len(bad_records)} "
        f"graphs_in_healthy_files={sum(graph_counts.values())}"
    )

    ds = H5TrackGraphDataset(healthy_paths)
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

    abs_paths = np.array([str(Path(p).resolve()) for p in healthy_paths], dtype=object)
    h5_file_names = np.array([Path(p).name for p in healthy_paths], dtype=object)
    h5_graph_counts = np.array(
        [int(graph_counts[p]) for p in healthy_paths],
        dtype=np.int64,
    )
    if len(set(h5_file_names.tolist())) != len(h5_file_names):
        raise SystemExit(
            "Duplicate H5 basenames make relocation-safe split validation ambiguous. "
            "Rename the files or keep them under one stable absolute path."
        )

    bad_paths = np.array(
        [str(Path(p).resolve()) for p, _ in bad_records],
        dtype=object,
    )
    bad_errors = np.array([error for _, error in bad_records], dtype=object)

    np.savez(
        out_path,
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        seed=np.int64(args.seed),
        val_fraction=np.float64(args.val_fraction),
        data_glob=np.array(args.data_glob, dtype=object),
        split_schema_version=np.int64(2),
        h5_paths=abs_paths,
        h5_file_names=h5_file_names,
        h5_graph_counts=h5_graph_counts,
        bad_h5_paths=bad_paths,
        bad_h5_errors=bad_errors,
        bad_h5_manifest=np.array(
            str(bad_manifest_path.resolve()) if bad_records else "",
            dtype=object,
        ),
        h5_check=np.array(args.h5_check, dtype=object),
        n_graphs=np.int64(n),
    )

    print(f"[ok] wrote split: {out_path}")
    print(f"[i] graphs: total={n} train={len(train_idx)} val={len(val_idx)}")
    print(f"[i] seed={args.seed} val_fraction={args.val_fraction}")


if __name__ == "__main__":
    main()
