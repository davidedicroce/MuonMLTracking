#!/usr/bin/env python3
"""
Two-phase Optuna tuner for train_TrackGraph.py (track-parameter GNN regressor).

This is adapted from tune_SegmentClassifier_optuna.py, but the objective is regression:
minimize overall validation SMAPE across the three target components:
  y_track = [ptq, eta, phi]

Important:
- train_TrackGraph.py prints validation SMAPE per epoch, but the checkpoint does not
  currently store val_smape_ptq/eta/phi. This tuner therefore parses the trainer log
  and uses the best observed validation mean SMAPE as the Optuna objective.
- The trainer checkpoint is still selected by train_TrackGraph.py according to its
  own --early-stop-monitor, usually val_loss. If you want the saved checkpoint to
  correspond exactly to best SMAPE, add SMAPE checkpoint metadata/monitoring to the
  trainer. For tuning/ranking, this script uses the log-derived SMAPE.

Example:
python tune_TrackGraph.py \
  --train-script ./train_TrackGraph.py \
  --data-glob "./data/track_graph_part*.h5" \
  --split-file "./data/split_track_graph_seed12345.npz" \
  --out-dir "./tuning_track_graph_smape" \
  --storage-path "/shared/wp2p5/sqlite/optuna_track_graph_smape.db" \
  --study-name "track_graph_mean_smape" \
  --n-trials 80 \
  --fast-gpus-per-trial 2 \
  --n-jobs 4 \
  --fast-epochs 40 \
  --fast-max-train-graphs 20000 \
  --refit-topk 5 \
  --refit-epochs 200 \
  --refit-max-train-graphs -1 \
  --num-workers 4 \
  --pin-memory \
  --wandb-mode disabled 2>&1 | tee log_tune_track_graph.txt

Run a separate architecture-family study:
python tune_TrackGraph.py \
  --train-script ./train_TrackGraph.py \
  --data-glob "./data/track_graph_part*.h5" \
  --split-file "./data/split_track_graph_seed12345.npz" \
  --out-dir "./tuning_track_graph_mpnn" \
  --storage-path "/shared/wp2p5/sqlite/optuna_track_graph_mpnn.db" \
  --study-name "track_graph_smape_mpnn" \
  --fixed-layer-type mpnn \
  --n-trials 80 \
  --fast-gpus-per-trial 2 \
  --n-jobs 4 \
  --num-workers 4 \
  --pin-memory \
  --wandb-mode disabled
"""

import argparse
import errno
try:
    import fcntl
except Exception:  # pragma: no cover - fcntl is Unix-only
    fcntl = None
import glob
import json
import math
import os
import random
import re
import shlex
import sqlite3
import subprocess
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna
import optuna.exceptions
import optuna.storages
import torch
from sqlalchemy import event
from sqlalchemy.pool import NullPool


# -----------------------------
# Small utilities
# -----------------------------

def now_utc_compact() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


_jsonl_lock = threading.Lock()
_mkdir_lock = threading.RLock()

_TRANSIENT_MKDIR_ERRNOS = {
    errno.ENOENT,
    errno.EEXIST,
    getattr(errno, "ESTALE", 116),
    getattr(errno, "EAGAIN", 11),
    getattr(errno, "EINTR", 4),
    getattr(errno, "ETIMEDOUT", 110),
}


def mkdir(p: Path, *, retries: int = 40, base_sleep_s: float = 0.10, max_sleep_s: float = 5.0) -> None:
    """Create a directory robustly on shared filesystems.

    With Optuna ``n_jobs > 1`` several worker threads can try to create the same
    output directory at nearly the same time. On EOS/NFS-like filesystems,
    ``Path.mkdir(..., exist_ok=True)`` can still surface a transient
    ``FileExistsError`` because directory metadata is not immediately visible to
    the process that lost the race. Retry and explicitly accept the successful
    end state: the path exists and is a directory.
    
    EOS/NFS-like filesystems can also briefly report inconsistent metadata while
    recursive parent creation is in progress. This implementation serializes
    mkdir attempts inside this Python process, uses os.makedirs(..., exist_ok=True),
    and retries transient metadata errors with jittered exponential backoff.
    """
    p = Path(p).expanduser()
    last_err: Optional[BaseException] = None
    
    if p.is_dir():
        return

    for i in range(max(1, int(retries))):
        try:
            with _mkdir_lock:
                if p.is_dir():
                    return

                os.makedirs(str(p), exist_ok=True)

                if p.is_dir():
                    return
        
        except FileExistsError as e:
            last_err = e
            if p.is_dir():
                return
            if p.exists() and not p.is_dir():
                raise RuntimeError(f"[io] Path exists but is not a directory: {p}") from e
        except OSError as e:
            last_err = e
            eno = getattr(e, "errno", None)

            if eno == errno.EEXIST:
                if p.is_dir():
                    return
                if p.exists() and not p.is_dir():
                    raise RuntimeError(f"[io] Path exists but is not a directory: {p}") from e

            elif eno not in _TRANSIENT_MKDIR_ERRNOS:
                if p.is_dir():
                    return
                raise RuntimeError(f"[io] Cannot create directory: {p} ({e!r})") from e

        sleep_s = min(float(max_sleep_s), float(base_sleep_s) * (2 ** min(i, 8)))
        sleep_s *= random.uniform(0.75, 1.25)
        time.sleep(sleep_s)
        if p.is_dir():
            return

    if p.is_dir():
        return
    if p.exists() and not p.is_dir():
        raise RuntimeError(f"[io] Path exists but is not a directory: {p}") from last_err
    raise RuntimeError(f"[io] Cannot create directory: {p} ({last_err!r})") from last_err

def ensure_dir_writable(p: Path, what: str) -> None:
    p = Path(p).expanduser()
    try:
        mkdir(p)
    except Exception as e:
        raise RuntimeError(f"[io] Cannot create {what} directory: {p} ({e!r})") from e

    probe = p / f".write_probe_{os.getpid()}_{threading.get_ident()}"
    try:
        with probe.open("w", encoding="utf-8") as f:
            f.write("ok\n")
        probe.unlink(missing_ok=True)
    except Exception as e:
        raise RuntimeError(f"[io] {what} directory is not writable: {p} ({e!r})") from e


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path = Path(path).expanduser()
    mkdir(path.parent)
    line = json.dumps(record, sort_keys=True) + "\n"

    # Protect against concurrent Optuna workers writing to the same JSONL. The
    # threading lock covers this process; fcntl, when available, also protects
    # accidental multiple tuner processes sharing the same out-dir.
    with _jsonl_lock:
        with path.open("a", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                # JSONL is the durable source of truth when --no-optuna-sqlite is used.
                # fsync is a little slower, but losing a completed trial record is worse
                # than spending a few extra milliseconds per finished trial.
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    out.append(json.loads(s))
                except Exception:
                    continue
    except Exception:
        return out
    return out



_TRAIN_IO_FAILURE_PATTERNS: List[Tuple[str, List[str]]] = [
    (
        "hdf5_eos_read",
        [
            "bad symbol table node signature",
            "unable to get group info",
            "unable to open file",
            "unable to read superblock",
            "file signature not found",
            "truncated file",
            "h5py",
        ],
    ),
    (
        "filesystem_transient",
        [
            "input/output error",
            "stale file handle",
            "transport endpoint is not connected",
            "no such file or directory",
            "connection timed out",
            "temporarily unavailable",
        ],
    ),
]


def read_text_tail(path: Path, *, max_bytes: int = 256_000) -> str:
    """Read the tail of a text log without loading huge torchrun logs into RAM."""
    path = Path(path)
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - int(max_bytes)), os.SEEK_SET)
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def classify_training_failure(
    log_path: Path,
    *,
    returncode: Optional[int],
    objective_value: Optional[float],
) -> Dict[str, Any]:
    """Classify trainer failures without changing HDF5/EOS reading behavior.

    The trainer should still fail fast if an H5/EOS read is broken. The tuner
    only labels that failure in JSONL so later ranking/refit can ignore it and
    operators can decide whether to rerun the same hyperparameters.
    """
    if int(returncode or 0) == 0 and objective_value is not None:
        return {"status": "ok", "failure_type": None, "likely_eos_io": False}

    tail = read_text_tail(log_path)
    tail_l = tail.lower()

    matched_type: Optional[str] = None
    matched_patterns: List[str] = []
    for failure_type, patterns in _TRAIN_IO_FAILURE_PATTERNS:
        for pat in patterns:
            if pat in tail_l:
                matched_type = failure_type
                matched_patterns.append(pat)
        if matched_type is not None:
            break

    likely_eos_io = matched_type in {"hdf5_eos_read", "filesystem_transient"}
    if returncode not in (None, 0):
        status = "infra_fail" if likely_eos_io else "train_fail"
    else:
        status = "objective_missing"

    summary_lines: List[str] = []
    for line in tail.splitlines():
        low = line.lower()
        if (
            "runtimeerror:" in low
            or "childfailederror" in low
            or "traceback" in low
            or "h5py" in low
            or any(p in low for _, pats in _TRAIN_IO_FAILURE_PATTERNS for p in pats)
        ):
            summary_lines.append(line.strip())
    summary = "\n".join(summary_lines[-20:])

    return {
        "status": status,
        "failure_type": matched_type or ("nonzero_returncode" if returncode not in (None, 0) else "missing_objective"),
        "likely_eos_io": bool(likely_eos_io),
        "matched_patterns": matched_patterns,
        "summary": summary[-4000:],
    }


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def parse_three_floats_csv(s: str, *, default: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tuple[float, float, float]:
    try:
        parts = [float(x.strip()) for x in str(s).split(",")]
        if len(parts) != 3:
            raise ValueError
        if not all(math.isfinite(x) for x in parts):
            raise ValueError
        return (parts[0], parts[1], parts[2])
    except Exception:
        return default


def weighted_mean3(values: Tuple[float, float, float], weights: Tuple[float, float, float]) -> float:
    denom = float(sum(weights))
    if denom <= 0:
        weights = (1.0, 1.0, 1.0)
        denom = 3.0
    return float(sum(v * w for v, w in zip(values, weights)) / denom)


def _abspath_glob(pattern: str, *, base_dir: Path) -> str:
    """
    Make a glob pattern absolute while preserving wildcards.
    """
    pattern = str(pattern)
    if os.path.isabs(pattern):
        return pattern
    return str((base_dir / pattern).resolve())


def validate_training_inputs(*, data_glob: str, split_file: Path) -> None:
    split_file = Path(split_file)
    if not split_file.exists():
        raise RuntimeError(f"[input] split file does not exist: {split_file}")

    matches = glob.glob(data_glob)
    if not matches:
        raise RuntimeError(
            f"[input] data_glob matched 0 files: {data_glob}\n"
            "        Tip: pass an absolute path, or run from the repo root.\n"
            "        Tip: if files are on a remote FS, ensure the mount is visible on this node."
        )


def build_trial_ckpt_path(ckpt_dir: Path, save_base: str, run_id: str) -> Path:
    base = os.path.basename(save_base)
    stem, ext = os.path.splitext(base)
    ext = ext if ext else ".pt"
    return Path(ckpt_dir) / f"{stem}_{run_id}{ext}"


def find_completed_refit_record(refit_results_path: Path, *, source_trial_number: int) -> Optional[Dict[str, Any]]:
    records = load_jsonl_records(refit_results_path)
    matches: List[Dict[str, Any]] = []
    for r in records:
        try:
            if r.get("phase") != "refit":
                continue
            if int(r.get("source_trial_number")) != int(source_trial_number):
                continue
            if int(r.get("returncode", 1)) != 0:
                continue
            ckpt = r.get("ckpt", {}) or {}
            objective_value = safe_float(r.get("objective_value"))
            if not bool(ckpt.get("ckpt_exists", False)):
                continue
            if objective_value is None:
                continue
            matches.append(r)
        except Exception:
            continue
    if not matches:
        return None
    matches.sort(key=lambda r: str(r.get("timestamp_utc", "")))
    return matches[-1]


# -----------------------------
# Progress DB (separate from Optuna storage)
# -----------------------------

PROGRESS_DB_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trial_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at_utc TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL,

  phase TEXT NOT NULL,              -- "fast" | "refit"
  study_name TEXT,

  trial_number INTEGER,             -- Optuna trial number for fast phase
  source_trial_number INTEGER,      -- for refit rows: originating fast trial number
  rank INTEGER,                     -- for refit rows: rank among finalists

  run_id TEXT NOT NULL,
  status TEXT NOT NULL,             -- "running" | "ok" | "fail" | "pruned"
  returncode INTEGER,

  seconds REAL,
  gpu_ids TEXT,
  master_port INTEGER,

  best_objective REAL,
  ckpt_path TEXT,
  log_path TEXT,

  hparams_json TEXT,
  error TEXT
);

CREATE INDEX IF NOT EXISTS idx_trial_runs_phase_status ON trial_runs(phase, status);
CREATE INDEX IF NOT EXISTS idx_trial_runs_trial_number ON trial_runs(trial_number);
CREATE INDEX IF NOT EXISTS idx_trial_runs_run_id ON trial_runs(run_id);
"""

_progress_db_lock = threading.Lock()


def _progress_db_connect(path: Path, *, timeout_s: int = 30) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=float(timeout_s), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=%d;" % int(timeout_s * 1000))
    return conn


def progress_db_init(db_path: Path, *, timeout_s: int = 30) -> None:
    db_path = Path(db_path)
    mkdir(db_path.parent)
    with _progress_db_lock:
        conn = _progress_db_connect(db_path, timeout_s=timeout_s)
        try:
            conn.executescript(PROGRESS_DB_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()


def progress_db_insert_running(
    db_path: Path,
    *,
    phase: str,
    study_name: Optional[str],
    trial_number: Optional[int],
    source_trial_number: Optional[int],
    rank: Optional[int],
    run_id: str,
    gpu_ids: List[str],
    master_port: int,
    hparams: Dict[str, Any],
    ckpt_path: Path,
    log_path: Path,
    timeout_s: int = 30,
) -> int:
    now = now_utc_compact()
    with _progress_db_lock:
        conn = _progress_db_connect(db_path, timeout_s=timeout_s)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO trial_runs (
                  created_at_utc, updated_at_utc, phase, study_name,
                  trial_number, source_trial_number, rank,
                  run_id, status, gpu_ids, master_port,
                  ckpt_path, log_path, hparams_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now, now, str(phase), study_name,
                    trial_number, source_trial_number, rank,
                    str(run_id), "running", ",".join(gpu_ids), int(master_port),
                    str(ckpt_path), str(log_path), json.dumps(hparams, sort_keys=True),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def progress_db_update_done(
    db_path: Path,
    row_id: int,
    *,
    status: str,
    returncode: Optional[int],
    seconds: Optional[float],
    best_objective: Optional[float],
    error: Optional[str],
    timeout_s: int = 30,
) -> None:
    now = now_utc_compact()
    with _progress_db_lock:
        conn = _progress_db_connect(db_path, timeout_s=timeout_s)
        try:
            conn.execute(
                """
                UPDATE trial_runs
                SET updated_at_utc=?,
                    status=?,
                    returncode=?,
                    seconds=?,
                    best_objective=?,
                    error=?
                WHERE id=?
                """,
                (now, str(status), returncode, seconds, best_objective, error, int(row_id)),
            )
            conn.commit()
        finally:
            conn.close()


# -----------------------------
# GPU allocation / Optuna storage
# -----------------------------

def _detect_gpus_via_nvidia_smi() -> Optional[List[str]]:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        ids = []
        for line in out.splitlines():
            m = re.match(r"^\s*GPU\s+(\d+)\s*:", line)
            if m:
                ids.append(m.group(1))
        return ids if ids else None
    except Exception:
        return None


def parse_cuda_visible_devices(cvd: Optional[str]) -> List[str]:
    if cvd is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is not None and str(cvd).strip() != "":
        parts = [p.strip() for p in str(cvd).split(",") if p.strip() != ""]
        if parts:
            return parts

    smi_ids = _detect_gpus_via_nvidia_smi()
    if smi_ids is not None:
        return smi_ids

    try:
        n = int(torch.cuda.device_count())
    except Exception:
        n = 0
    return [str(i) for i in range(max(1, n))]


class GPUAllocator:
    """
    In-process GPU pool allocator for optuna.study.optimize(..., n_jobs>1).
    Each objective acquires a subset and launches a torchrun process with its own
    CUDA_VISIBLE_DEVICES.
    """

    def __init__(self, devices: List[str]):
        self._all = list(devices)
        self._free = list(devices)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    @contextmanager
    def acquire(self, n: int, *, timeout_s: Optional[float] = None):
        if n <= 0:
            raise ValueError("n must be >= 1")
        t0 = time.time()
        with self._cv:
            while len(self._free) < n:
                if timeout_s is not None:
                    remain = timeout_s - (time.time() - t0)
                    if remain <= 0:
                        raise TimeoutError(f"Timed out waiting for {n} GPUs (free={len(self._free)}/{len(self._all)}).")
                    self._cv.wait(timeout=remain)
                else:
                    self._cv.wait()
            got = self._free[:n]
            self._free = self._free[n:]
        try:
            yield got
        finally:
            with self._cv:
                self._free = list(got) + self._free
                self._cv.notify_all()


def pick_master_port(base: int, trial_number: int) -> int:
    jitter = random.randint(0, 127)
    return int(base + (trial_number % 896) + jitter)


def make_sqlite_storage(sqlite_path: Path, *, timeout_s: int, enable_wal: bool, pool: str = "null"):
    sqlite_path = Path(sqlite_path)
    mkdir(sqlite_path.parent)
    url = f"sqlite:///{sqlite_path}"

    connect_args = {
        "timeout": int(timeout_s),
        "check_same_thread": False,
    }
    engine_kwargs = {
        "connect_args": connect_args,
    }
    if pool == "null":
        engine_kwargs["poolclass"] = NullPool

    storage = optuna.storages.RDBStorage(url=url, engine_kwargs=engine_kwargs)

    @event.listens_for(storage.engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute(f"PRAGMA busy_timeout={int(timeout_s) * 1000};")
            if enable_wal:
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA temp_store=MEMORY;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()
        except Exception:
            pass

    return storage


def infer_nproc_per_node(requested: int, env: Optional[dict] = None) -> int:
    if requested and requested > 0:
        return int(requested)

    env = env or os.environ
    cvd = env.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is not None and str(cvd).strip() != "":
        n = len([x for x in str(cvd).split(",") if x.strip() != ""])
        if n > 0:
            return n

    try:
        return max(1, int(torch.cuda.device_count()))
    except Exception:
        return 1


# -----------------------------
# Regression objective extraction
# -----------------------------

# Example line emitted by train_TrackGraph.py:
# [epoch 001] train ... smape=(..%,..%,..%) ... | val loss=... ... smape=(ptq%,eta%,phi%) ... rmse_mean=...
VAL_LINE_RE = re.compile(
    r"\[epoch\s+(?P<epoch>\d+)\].*?"
    r"\|\s*val\s+loss=(?P<val_loss>[-+0-9.eE]+).*?"
    r"smape=\(\s*(?P<ptq>[-+0-9.eE]+)%\s*,\s*(?P<eta>[-+0-9.eE]+)%\s*,\s*(?P<phi>[-+0-9.eE]+)%\s*\).*?"
    r"rmse_mean=(?P<rmse_mean>[-+0-9.eE]+)",
    re.IGNORECASE,
)


def parse_log_regression_metrics(log_path: Path, *, weights: Tuple[float, float, float]) -> Dict[str, Any]:
    """
    Parse all validation epochs from train_TrackGraph.py log and return:
      - best_overall_smape: min weighted mean of val/smape_{ptq,eta,phi}
      - best epoch and per-component values
      - last epoch values
    """
    out: Dict[str, Any] = {
        "log_path": str(log_path),
        "log_exists": Path(log_path).exists(),
        "objective_name": "val_smape_mean",
        "objective_weights": list(weights),
        "epochs_seen": 0,
    }
    if not Path(log_path).exists():
        return out

    best: Optional[Dict[str, Any]] = None
    last: Optional[Dict[str, Any]] = None

    try:
        with Path(log_path).open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = VAL_LINE_RE.search(line)
                if not m:
                    continue
                ptq = float(m.group("ptq"))
                eta = float(m.group("eta"))
                phi = float(m.group("phi"))
                overall = weighted_mean3((ptq, eta, phi), weights)
                rec: Dict[str, Any] = {
                    "epoch": int(m.group("epoch")),
                    "val_loss": float(m.group("val_loss")),
                    "val_smape_ptq": ptq,
                    "val_smape_eta": eta,
                    "val_smape_phi": phi,
                    "val_smape_mean": overall,
                    "val_rmse_mean": float(m.group("rmse_mean")),
                }
                last = rec
                if best is None or overall < float(best["val_smape_mean"]):
                    best = dict(rec)
    except Exception as e:
        out["parse_error"] = f"{type(e).__name__}: {e}"
        return out

    if last is not None:
        out["epochs_seen"] = int(last["epoch"])
        out["last"] = last
    if best is not None:
        out["best"] = best
        out["best_overall_smape"] = safe_float(best.get("val_smape_mean"))
        out["best_epoch"] = best.get("epoch")
        out["val_smape_ptq"] = best.get("val_smape_ptq")
        out["val_smape_eta"] = best.get("val_smape_eta")
        out["val_smape_phi"] = best.get("val_smape_phi")
        out["val_loss_at_best_smape"] = best.get("val_loss")
        out["val_rmse_mean_at_best_smape"] = best.get("val_rmse_mean")
    return out


def read_ckpt_metrics(ckpt_path: Path) -> Dict[str, Any]:
    """
    Read lightweight metadata from the trainer checkpoint. SMAPE itself is taken
    from the log because train_TrackGraph.py currently does not persist SMAPE in
    the checkpoint.
    """
    out: Dict[str, Any] = {"ckpt_path": str(ckpt_path)}
    if not Path(ckpt_path).exists():
        out["ckpt_exists"] = False
        return out
    out["ckpt_exists"] = True

    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as e:
        out["read_error"] = f"{type(e).__name__}: {e}"
        return out

    for k in [
        "best_monitor", "early_stop_monitor", "best_ckpt_epoch", "run_id",
        "hidden_dim", "layers", "dropout", "layer_type", "gat_heads",
        "sage_aggr", "edgeconv_aggr", "graph_pool", "loss_type",
        "fourier", "fourier_base", "fourier_min_exp", "fourier_max_exp",
        "target_scale_mode", "target_scale_eps", "phi_mode", "phi_period",
        "weight_decay", "edge_dropout", "feat_noise_std", "ema", "ema_decay",
        "lr_schedule", "warmup_epochs", "min_lr_ratio", "code_version", "epoch",
        "bad_epochs",
    ]:
        if k in ckpt:
            v = ckpt.get(k)
            if torch.is_tensor(v):
                try:
                    v = v.detach().cpu().tolist()
                except Exception:
                    v = str(v)
            out[k] = v

    if "target_loss_weights" in ckpt:
        v = ckpt.get("target_loss_weights")
        if torch.is_tensor(v):
            v = v.detach().cpu().tolist()
        out["target_loss_weights"] = v
    if "target_center_linear" in ckpt:
        v = ckpt.get("target_center_linear")
        if torch.is_tensor(v):
            v = v.detach().cpu().tolist()
        out["target_center_linear"] = v
    if "target_scale_linear" in ckpt:
        v = ckpt.get("target_scale_linear")
        if torch.is_tensor(v):
            v = v.detach().cpu().tolist()
        out["target_scale_linear"] = v

    out["best_monitor"] = safe_float(out.get("best_monitor"))
    return out


def objective_display_name(args) -> str:
    return f"val_smape_mean[{args.objective_component_weights}]"


def objective_direction(_args) -> str:
    return "minimize"


def objective_value_from_log(log_path: Path, args) -> Tuple[Optional[float], Dict[str, Any]]:
    weights = parse_three_floats_csv(args.objective_component_weights)
    log_metrics = parse_log_regression_metrics(log_path, weights=weights)
    return safe_float(log_metrics.get("best_overall_smape")), log_metrics


# -----------------------------
# Hyperparameters and command construction
# -----------------------------

def normalize_hparams(hp: Dict[str, Any]) -> Dict[str, Any]:
    hp = dict(hp)
    layer_type = hp.get("layer_type", "mpnn")
    hp.setdefault("layer_type", layer_type)
    hp.setdefault("gat_heads", 4)
    hp.setdefault("sage_aggr", "mean")
    hp.setdefault("edgeconv_aggr", "mean")
    hp.setdefault("graph_pool", "meanmax")
    hp.setdefault("fourier", True)
    hp.setdefault("fourier_base", 3.0)
    hp.setdefault("fourier_min_exp", -6)
    hp.setdefault("fourier_max_exp", 6)
    hp.setdefault("loss_type", "smoothl1")
    hp.setdefault("target_scale", "robust")
    hp.setdefault("phi_mode", "sincos")
    hp.setdefault("phi_vec_weight", 1.0)
    hp.setdefault("target_loss_weights", "1.0,1.0,1.0")

    if hp["layer_type"] == "gat_residual":
        hidden_dim = int(hp.get("hidden_dim", 192))
        valid_heads = [h for h in [1, 2, 4, 8] if hidden_dim % h == 0]
        if not valid_heads:
            valid_heads = [1]
        if int(hp.get("gat_heads", 4)) not in valid_heads:
            hp["gat_heads"] = 4 if 4 in valid_heads else valid_heads[0]
    else:
        hp["gat_heads"] = int(hp.get("gat_heads", 4))

    return hp


def sample_hparams(trial: optuna.Trial, args) -> Dict[str, Any]:
    hp: Dict[str, Any] = {}
    hp["_trial_idx"] = trial.number

    hp["lr"] = trial.suggest_float("lr", args.lr_low, args.lr_high, log=True)
    hp["weight_decay"] = trial.suggest_float("weight_decay", args.weight_decay_low, args.weight_decay_high, log=True)
    hp["dropout"] = trial.suggest_float("dropout", args.dropout_low, args.dropout_high)
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", args.hidden_dim_choices)
    hp["layers"] = trial.suggest_categorical("layers", args.layer_choices)

    hp["edge_dropout"] = trial.suggest_float("edge_dropout", args.edge_dropout_low, args.edge_dropout_high)
    hp["feat_noise_std"] = trial.suggest_categorical("feat_noise_std", args.feat_noise_choices)
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", args.ema_decay_choices)

    hp["lr_schedule"] = trial.suggest_categorical("lr_schedule", args.lr_schedule_choices)
    hp["warmup_epochs"] = trial.suggest_categorical("warmup_epochs", args.warmup_epoch_choices)
    hp["min_lr_ratio"] = trial.suggest_categorical("min_lr_ratio", args.min_lr_ratio_choices)

    hp["loss_type"] = trial.suggest_categorical("loss_type", args.loss_type_choices)
    hp["target_scale"] = trial.suggest_categorical("target_scale", args.target_scale_choices)
    hp["graph_pool"] = trial.suggest_categorical("graph_pool", args.graph_pool_choices)

    if args.fixed_phi_mode is not None:
        hp["phi_mode"] = args.fixed_phi_mode
    elif args.tune_phi_mode:
        hp["phi_mode"] = trial.suggest_categorical("phi_mode", ["sincos", "scalar"])
    else:
        hp["phi_mode"] = "sincos"

    hp["phi_vec_weight"] = trial.suggest_float("phi_vec_weight", args.phi_vec_weight_low, args.phi_vec_weight_high, log=True)

    if args.tune_target_loss_weights:
        ptq_w = trial.suggest_float("target_loss_weight_ptq", args.target_loss_weight_low, args.target_loss_weight_high, log=True)
        eta_w = trial.suggest_float("target_loss_weight_eta", args.target_loss_weight_low, args.target_loss_weight_high, log=True)
        phi_w = trial.suggest_float("target_loss_weight_phi", args.target_loss_weight_low, args.target_loss_weight_high, log=True)
        hp["target_loss_weights"] = f"{ptq_w:.8g},{eta_w:.8g},{phi_w:.8g}"
    else:
        hp["target_loss_weights"] = args.fixed_target_loss_weights

    if args.fixed_layer_type is not None:
        hp["layer_type"] = args.fixed_layer_type
    else:
        hp["layer_type"] = trial.suggest_categorical(
            "layer_type",
            ["mpnn", "edge_residual", "sage_residual", "gat_residual"],
        )

    if hp["layer_type"] == "gat_residual":
        valid_heads = [h for h in [1, 2, 4, 8] if int(hp["hidden_dim"]) % h == 0]
        if not valid_heads:
            raise optuna.TrialPruned()
        hp["gat_heads"] = trial.suggest_categorical("gat_heads", valid_heads)
    else:
        hp["gat_heads"] = 4

    if hp["layer_type"] == "sage_residual":
        hp["sage_aggr"] = trial.suggest_categorical("sage_aggr", ["mean", "sum", "max"])
    else:
        hp["sage_aggr"] = "mean"

    if hp["layer_type"] == "edge_residual":
        hp["edgeconv_aggr"] = trial.suggest_categorical("edgeconv_aggr", ["mean", "sum", "max"])
    else:
        hp["edgeconv_aggr"] = "mean"

    if args.tune_fourier:
        hp["fourier"] = trial.suggest_categorical("fourier", [True, False])
    else:
        hp["fourier"] = bool(args.fixed_fourier)

    if hp["fourier"] and args.tune_fourier_knobs:
        hp["fourier_base"] = trial.suggest_categorical("fourier_base", args.fourier_base_choices)
        # Keep this bounded; very large Fourier expansions can explode memory.
        hp["fourier_min_exp"] = trial.suggest_categorical("fourier_min_exp", args.fourier_min_exp_choices)
        hp["fourier_max_exp"] = trial.suggest_categorical("fourier_max_exp", args.fourier_max_exp_choices)
        if int(hp["fourier_min_exp"]) > int(hp["fourier_max_exp"]):
            raise optuna.TrialPruned()
    else:
        hp["fourier_base"] = args.fourier_base
        hp["fourier_min_exp"] = args.fourier_min_exp
        hp["fourier_max_exp"] = args.fourier_max_exp

    return normalize_hparams(hp)


def build_command(
    args,
    run_id: str,
    hp: Dict[str, Any],
    ckpt_dir: Path,
    env: dict,
    *,
    phase: str,
    epochs: int,
    max_train_graphs: int,
    early_stop_patience: int,
    master_port: int,
) -> Tuple[List[str], Path, Path]:
    hp = normalize_hparams(hp)
    ckpt_path = build_trial_ckpt_path(ckpt_dir, args.save, run_id)
    log_path = args.out_dir / "logs" / f"{phase}_{run_id}.log"

    nproc = infer_nproc_per_node(args.nproc_per_node, env=env)
    save_dir_abs = Path(ckpt_dir).expanduser().resolve()
    bad_graphs_file = args.out_dir / "logs" / f"bad_graphs_{phase}_{run_id}.jsonl"

    cmd = [
        "torchrun",
        "--standalone",
        f"--master_port={int(master_port)}",
        f"--nproc_per_node={nproc}",
        str(args.train_script),
        "--data-glob", args.data_glob,
        "--split-file", args.split_file,
        "--epochs", str(int(epochs)),
        "--num-workers", str(int(args.num_workers)),
        "--save", args.save,
        "--save-dir", str(save_dir_abs),
        "--run-id", run_id,
        "--resume" if phase == "refit" and args.refit_resume else "--no-resume",
        "--seed", str(int(args.seed + hp.get("_trial_idx", 0) * 100)),
        "--early-stop-monitor", args.early_stop_monitor,
        "--early-stop-patience", str(int(early_stop_patience)),
        "--early-stop-min-delta", str(float(args.early_stop_min_delta)),
        "--prefetch-factor", str(int(args.prefetch_factor)),
        "--worker-start-method", str(args.worker_start_method),
        "--bad-graphs-file", str(bad_graphs_file),
        "--target-stats-max-events", str(int(args.target_stats_max_events)),
        "--h5-open-retries", str(int(args.h5_open_retries)),
        "--h5-open-retry-sleep", str(float(args.h5_open_retry_sleep)),
        "--h5-read-retries", str(int(args.h5_read_retries)),
        "--h5-read-retry-sleep", str(float(args.h5_read_retry_sleep)),
        "--h5-max-open-files-per-worker", str(int(args.h5_max_open_files_per_worker)),
        "--worker-open-jitter-s", str(float(args.worker_open_jitter_s)),
    ]

    if args.h5_close_after_read:
        cmd.append("--h5-close-after-read")
    if args.h5_swmr:
        cmd.append("--h5-swmr")

    if getattr(args, "code_version", None):
        cmd += ["--code-version", str(args.code_version)]

    if args.pin_memory:
        cmd.append("--pin-memory")

    if args.persistent_workers:
        cmd.append("--persistent-workers")
    else:
        cmd.append("--no-persistent-workers")

    if args.trainer_amp:
        cmd.append("--amp")
    else:
        cmd.append("--no-amp")
    cmd += ["--amp-dtype", str(args.amp_dtype)]

    if max_train_graphs > 0:
        cmd += ["--max-train-graphs", str(int(max_train_graphs))]

    if args.no_early_stop:
        cmd.append("--no-early-stop")
    else:
        cmd.append("--early-stop")

    if args.reload_best_half_patience:
        cmd.append("--reload-best-half-patience")

    # W&B
    if args.wandb_mode == "disabled":
        cmd += ["--wandb-mode", "disabled"]
    else:
        cmd += ["--wandb", "--wandb-mode", args.wandb_mode, "--wandb-project", args.wandb_project]
        cmd += ["--wandb-name", f"{phase}_{run_id}"]

    # Core hyperparameters
    cmd += ["--lr", f"{float(hp['lr']):.8g}"]
    cmd += ["--weight-decay", f"{float(hp['weight_decay']):.8g}"]
    cmd += ["--dropout", f"{float(hp['dropout']):.8g}"]
    cmd += ["--hidden-dim", str(int(hp["hidden_dim"]))]
    cmd += ["--layers", str(int(hp["layers"]))]

    cmd += ["--edge-dropout", f"{float(hp['edge_dropout']):.8g}"]
    cmd += ["--feat-noise-std", f"{float(hp['feat_noise_std']):.8g}"]
    cmd += ["--ema-decay", f"{float(hp['ema_decay']):.8g}"]

    cmd += ["--lr-schedule", str(hp["lr_schedule"])]
    cmd += ["--warmup-epochs", f"{float(hp['warmup_epochs']):.8g}"]
    cmd += ["--min-lr-ratio", f"{float(hp['min_lr_ratio']):.8g}"]
    cmd += ["--lr-plateau-factor", f"{float(args.lr_plateau_factor):.8g}"]
    cmd += ["--lr-plateau-patience", str(int(args.lr_plateau_patience))]
    cmd += ["--lr-plateau-min-lr", f"{float(args.lr_plateau_min_lr):.8g}"]

    # Regression-specific knobs
    cmd += ["--loss-type", str(hp["loss_type"])]
    cmd += ["--target-scale", str(hp["target_scale"])]
    cmd += ["--target-scale-eps", f"{float(args.target_scale_eps):.8g}"]
    cmd += ["--target-loss-weights", str(hp["target_loss_weights"])]
    cmd += ["--phi-mode", str(hp["phi_mode"])]
    cmd += ["--phi-period", f"{float(args.phi_period):.12g}"]
    cmd += ["--phi-vec-weight", f"{float(hp['phi_vec_weight']):.8g}"]
    cmd += ["--graph-pool", str(hp["graph_pool"])]

    # Layer family knobs
    cmd += ["--layer-type", str(hp["layer_type"])]
    cmd += ["--gat-heads", str(int(hp.get("gat_heads", 4)))]
    cmd += ["--sage-aggr", str(hp.get("sage_aggr", "mean"))]
    cmd += ["--edgeconv-aggr", str(hp.get("edgeconv_aggr", "mean"))]

    # Fourier knobs
    if bool(hp["fourier"]):
        cmd += ["--fourier"]
    else:
        cmd += ["--no-fourier"]
    cmd += ["--fourier-base", f"{float(hp['fourier_base']):.8g}"]
    cmd += ["--fourier-min-exp", str(int(hp["fourier_min_exp"]))]
    cmd += ["--fourier-max-exp", str(int(hp["fourier_max_exp"]))]

    return cmd, ckpt_path, log_path


def run_trial(cmd: List[str], log_path: Path, env: dict, *, save_dir: Optional[Path] = None) -> int:
    mkdir(log_path.parent)
    shell_cmd = " ".join(shlex.quote(x) for x in cmd)
    if save_dir is not None:
        save_dir = Path(save_dir).expanduser().resolve()
        shell_cmd = f"mkdir -p {shlex.quote(str(save_dir))} && exec {shell_cmd}"

    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("COMMAND:\n" + shell_cmd + "\n\n")
        lf.flush()
        p = subprocess.run(
            ["bash", "-lc", shell_cmd],
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return int(p.returncode)



# -----------------------------
# JSONL-backed trial selection
# -----------------------------

@dataclass
class JsonlTrialLike:
    number: int
    value: float
    params: Dict[str, Any]
    user_attrs: Dict[str, Any]


def completed_trials_from_results_jsonl(
    results_path: Path,
    *,
    session_id: Optional[str] = None,
) -> List[JsonlTrialLike]:
    """Recover completed fast trials from results_fast.jsonl.

    This is used as the source of truth when --no-optuna-sqlite is set. Keeping
    the optional session filter avoids accidentally refitting stale trials when
    an output directory is reused across several runs.
    """
    out: List[JsonlTrialLike] = []
    records = load_jsonl_records(results_path)
    for i, r in enumerate(records):
        try:
            if r.get("phase") != "fast":
                continue
            if session_id is not None and r.get("session_id") != session_id:
                continue
            if int(r.get("returncode", 1)) != 0:
                continue
            value = safe_float(r.get("objective_value"))
            if value is None:
                continue
            hparams = r.get("hparams") or {}
            if not isinstance(hparams, dict) or not hparams:
                continue
            trial_number = int(r.get("trial_number", i))
            best_metrics = (r.get("log_metrics") or {}).get("best") or {}
            user_attrs = {
                "run_id": r.get("run_id"),
                "log_path": r.get("log_path"),
                "ckpt_path": (r.get("ckpt") or {}).get("ckpt_path"),
                "objective_value": value,
                "hparams": hparams,
                "jsonl_record_index": i,
                "jsonl_session_id": r.get("session_id"),
            }
            if isinstance(best_metrics, dict):
                user_attrs.update(best_metrics)
            out.append(JsonlTrialLike(number=trial_number, value=float(value), params=hparams, user_attrs=user_attrs))
        except Exception:
            continue
    out.sort(key=lambda t: float(t.value))
    return out


def print_fast_phase_best_from_trials(trials: List[Any], args) -> None:
    if not trials:
        print("best objective value: <none> (no completed trials in selected source)")
        print("best params: <none>")
        return

    best = min(trials, key=lambda t: float(t.value))
    hparams = best.user_attrs.get("hparams") if hasattr(best, "user_attrs") else None
    params = hparams if isinstance(hparams, dict) and hparams else dict(getattr(best, "params", {}))
    print(f"best objective value: {best.value}")
    print("best params:", json.dumps(params, indent=2, sort_keys=True, default=str))
    user_attrs = getattr(best, "user_attrs", {}) or {}
    best_attrs = {
        k: user_attrs.get(k)
        for k in ["epoch", "val_smape_ptq", "val_smape_eta", "val_smape_phi", "val_smape_mean", "val_loss", "val_rmse_mean"]
        if k in user_attrs
    }
    if best_attrs:
        print("best log metrics:", json.dumps(best_attrs, indent=2, sort_keys=True, default=str))


# -----------------------------
# Optuna objective + refit
# -----------------------------

def objective_factory(
    args,
    ckpt_dir: Path,
    results_path: Path,
    allocator: GPUAllocator,
    progress_db_path: Optional[Path],
    *,
    session_id: Optional[str] = None,
):
    def objective(trial: optuna.Trial) -> float:
        hp = sample_hparams(trial, args)
        run_id = f"t{trial.number:04d}_{now_utc_compact()}"

        progress_row_id: Optional[int] = None
        gpus_per_trial = int(args.fast_gpus_per_trial)

        with allocator.acquire(gpus_per_trial, timeout_s=args.gpu_acquire_timeout_s) as gpu_ids:
            mkdir(Path(ckpt_dir).expanduser().resolve())
            mkdir((args.out_dir / "logs").expanduser().resolve())

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
            env.setdefault("NCCL_DEBUG", "WARN")
            env.setdefault("OMP_NUM_THREADS", str(args.omp_num_threads))
            env.setdefault("MKL_NUM_THREADS", str(args.mkl_num_threads))
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("HDF5_USE_FILE_LOCKING", str(args.hdf5_use_file_locking))
            env.setdefault("TRACKGRAPH_H5_WORKER_OPEN_JITTER_S", str(float(args.worker_open_jitter_s)))

            master_port = pick_master_port(args.master_port_base, trial.number)

            try:
                validate_training_inputs(data_glob=args.data_glob, split_file=Path(args.split_file))
            except Exception as e:
                raise optuna.TrialPruned(str(e))

            cmd, ckpt_path, log_path = build_command(
                args, run_id, hp, ckpt_dir,
                env=env,
                phase="fast",
                epochs=args.fast_epochs,
                max_train_graphs=args.fast_max_train_graphs,
                early_stop_patience=args.fast_early_stop_patience,
                master_port=master_port,
            )

            if progress_db_path is not None:
                try:
                    progress_row_id = progress_db_insert_running(
                        progress_db_path,
                        phase="fast",
                        study_name=getattr(args, "study_name", None),
                        trial_number=int(trial.number),
                        source_trial_number=None,
                        rank=None,
                        run_id=run_id,
                        gpu_ids=gpu_ids,
                        master_port=int(master_port),
                        hparams={k: v for k, v in hp.items() if not k.startswith("_")},
                        ckpt_path=ckpt_path,
                        log_path=log_path,
                        timeout_s=int(args.progress_db_timeout),
                    )
                except Exception:
                    progress_row_id = None

            if args.resume_skip_existing and ckpt_path.exists() and log_path.exists():
                objective_value, log_metrics = objective_value_from_log(log_path, args)
                ckpt_info = read_ckpt_metrics(ckpt_path)
                if objective_value is None:
                    if progress_db_path is not None and progress_row_id is not None:
                        progress_db_update_done(
                            progress_db_path, progress_row_id,
                            status="pruned", returncode=None, seconds=0.0, best_objective=None,
                            error="resume_skip_existing: checkpoint/log exist but SMAPE objective was not parseable",
                            timeout_s=int(args.progress_db_timeout),
                        )
                    raise optuna.TrialPruned()
                if progress_db_path is not None and progress_row_id is not None:
                    progress_db_update_done(
                        progress_db_path, progress_row_id,
                        status="ok", returncode=0, seconds=0.0, best_objective=float(objective_value),
                        error=None, timeout_s=int(args.progress_db_timeout),
                    )

                trial.set_user_attr("ckpt_path", str(ckpt_path))
                trial.set_user_attr("log_path", str(log_path))
                trial.set_user_attr("run_id", run_id)
                trial.set_user_attr("objective_value", float(objective_value))
                trial.set_user_attr("hparams", {k: v for k, v in hp.items() if not k.startswith("_")})
                for k, v in (log_metrics.get("best") or {}).items():
                    trial.set_user_attr(k, v)
                return float(objective_value)

            t0 = time.time()
            rc = run_trial(cmd, log_path, env, save_dir=ckpt_dir)
            dt = time.time() - t0

            ckpt_info = read_ckpt_metrics(ckpt_path)
            objective_value, log_metrics = objective_value_from_log(log_path, args)
            failure_info = classify_training_failure(
                log_path,
                returncode=int(rc),
                objective_value=objective_value,
            )
            hparams_clean = {k: v for k, v in hp.items() if not k.startswith("_")}

            record = {
                "trial_number": trial.number,
                "session_id": session_id,
                "returncode": rc,
                "status": failure_info.get("status"),
                "failure": failure_info if failure_info.get("status") != "ok" else None,
                "run_id": run_id,
                "seconds": dt,
                "phase": "fast",
                "gpus": gpu_ids,
                "master_port": int(master_port),
                "objective_metric": "val_smape_mean",
                "objective_display_name": objective_display_name(args),
                "objective_direction": objective_direction(args),
                "objective_value": objective_value,
                "hparams": hparams_clean,
                "ckpt": ckpt_info,
                "log_metrics": log_metrics,
                "log_path": str(log_path),
                "timestamp_utc": now_utc_compact(),
            }
            try:
                append_jsonl(results_path, record)
            except Exception as e:
                print(
                    f"[trial {trial.number}] WARN: failed to append to {results_path}: {e!r}. "
                    "Continuing, but this trial may be missing from JSONL-backed ranking.",
                    flush=True,
                )

            if rc != 0 or objective_value is None:
                if progress_db_path is not None and progress_row_id is not None:
                    progress_db_update_done(
                        progress_db_path, progress_row_id,
                        status="fail" if rc != 0 else "pruned",
                        returncode=int(rc),
                        seconds=float(dt),
                        best_objective=None,
                        error=(
                            f"{failure_info.get('failure_type')}: nonzero returncode {rc}" if rc != 0
                            else "SMAPE objective missing or log parse failed"
                        ),
                        timeout_s=int(args.progress_db_timeout),
                    )
                raise optuna.TrialPruned()

            if progress_db_path is not None and progress_row_id is not None:
                progress_db_update_done(
                    progress_db_path, progress_row_id,
                    status="ok", returncode=int(rc), seconds=float(dt),
                    best_objective=float(objective_value), error=None,
                    timeout_s=int(args.progress_db_timeout),
                )

            trial.set_user_attr("ckpt_path", str(ckpt_path))
            trial.set_user_attr("log_path", str(log_path))
            trial.set_user_attr("run_id", run_id)
            trial.set_user_attr("gpus", ",".join(gpu_ids))
            trial.set_user_attr("master_port", int(master_port))
            trial.set_user_attr("objective_metric", "val_smape_mean")
            trial.set_user_attr("objective_display_name", objective_display_name(args))
            trial.set_user_attr("objective_value", float(objective_value))
            trial.set_user_attr("hparams", hparams_clean)
            for k, v in (log_metrics.get("best") or {}).items():
                trial.set_user_attr(k, v)

            return float(objective_value)

    return objective


def refit_topk(
    args,
    ckpt_dir: Path,
    study: Optional[optuna.Study],
    k: int,
    refit_results_path: Path,
    allocator: GPUAllocator,
    progress_db_path: Optional[Path],
    *,
    trials_override: Optional[List[Any]] = None,
) -> None:
    if trials_override is not None:
        trials_sorted = sorted(trials_override, key=lambda t: float(t.value), reverse=False)
    elif study is not None:
        trials_sorted = sorted(
            [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE],
            key=lambda t: t.value,
            reverse=False,
        )
    else:
        trials_sorted = []
    all_topk = trials_sorted[:k]

    selected_ranks: Optional[List[int]] = getattr(args, "refit_ranks", None)
    if selected_ranks:
        selected_rank_set = set(int(x) for x in selected_ranks)
        finalists_with_rank = [
            (rank, t)
            for rank, t in enumerate(all_topk, start=1)
            if rank in selected_rank_set
        ]
        missing = sorted(selected_rank_set - {rank for rank, _ in finalists_with_rank})
        if missing:
            print(
                f"[refit] Requested ranks not available within top-{k}: {missing}. "
                f"Available ranks are 1..{len(all_topk)}.",
                flush=True,
            )
    else:
        finalists_with_rank = list(enumerate(all_topk, start=1))

    if not finalists_with_rank:
        print("[refit] No successful trials to refit.")
        return

    if selected_ranks:
        rank_msg = ",".join(str(r) for r, _ in finalists_with_rank)
        print(
            f"\n[refit] Re-training selected ranks [{rank_msg}] within top-{k} "
            f"with epochs={args.refit_epochs}, max_train_graphs={args.refit_max_train_graphs} ...\n",
            flush=True,
        )
    else:
        print(
            f"\n[refit] Re-training top {len(finalists_with_rank)} configs with "
            f"objective={objective_display_name(args)}, epochs={args.refit_epochs}, "
            f"max_train_graphs={args.refit_max_train_graphs} ...\n",
            flush=True,
        )

    for rank, t in finalists_with_rank:
        # Prefer fully materialized hparams from JSONL/user_attrs. Optuna params do
        # not include derived values such as the target_loss_weights CSV string.
        hp_src = (getattr(t, "user_attrs", {}) or {}).get("hparams")
        hp = dict(hp_src) if isinstance(hp_src, dict) and hp_src else dict(t.params)
        if args.fixed_layer_type is not None:
            hp["layer_type"] = args.fixed_layer_type
        hp["_trial_idx"] = t.number
        hp = normalize_hparams(hp)

        if args.resume_skip_existing_refit:
            prev = find_completed_refit_record(refit_results_path, source_trial_number=int(t.number))
            if prev is not None:
                prev_ckpt = (prev.get("ckpt", {}) or {}).get("ckpt_path", "<unknown>")
                prev_run_id = prev.get("run_id", "<unknown>")
                prev_metric = prev.get("objective_value", None)
                print(
                    f"[refit top{rank:02d}] SKIP: already completed earlier "
                    f"(source_trial={t.number}, run_id={prev_run_id}, objective={prev_metric}, ckpt={prev_ckpt})",
                    flush=True,
                )
                continue

        run_id = f"refit_top{rank:02d}_from_t{t.number:04d}"
        gpus_per_trial = int(args.refit_gpus_per_trial)
        progress_row_id: Optional[int] = None

        with allocator.acquire(gpus_per_trial, timeout_s=args.gpu_acquire_timeout_s) as gpu_ids:
            mkdir(Path(ckpt_dir).expanduser().resolve())
            mkdir((args.out_dir / "logs").expanduser().resolve())

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
            env.setdefault("NCCL_DEBUG", "WARN")
            env.setdefault("OMP_NUM_THREADS", str(args.omp_num_threads))
            env.setdefault("MKL_NUM_THREADS", str(args.mkl_num_threads))
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("HDF5_USE_FILE_LOCKING", str(args.hdf5_use_file_locking))
            env.setdefault("TRACKGRAPH_H5_WORKER_OPEN_JITTER_S", str(float(args.worker_open_jitter_s)))
            master_port = pick_master_port(args.master_port_base, 10_000 + rank)

            validate_training_inputs(data_glob=args.data_glob, split_file=Path(args.split_file))

            cmd, ckpt_path, log_path = build_command(
                args, run_id, hp, ckpt_dir,
                env=env,
                phase="refit",
                epochs=args.refit_epochs,
                max_train_graphs=args.refit_max_train_graphs,
                early_stop_patience=args.refit_early_stop_patience,
                master_port=master_port,
            )

            if progress_db_path is not None:
                try:
                    progress_row_id = progress_db_insert_running(
                        progress_db_path,
                        phase="refit",
                        study_name=getattr(args, "study_name", None),
                        trial_number=None,
                        source_trial_number=int(t.number),
                        rank=int(rank),
                        run_id=run_id,
                        gpu_ids=gpu_ids,
                        master_port=int(master_port),
                        hparams={k: v for k, v in hp.items() if not k.startswith("_")},
                        ckpt_path=ckpt_path,
                        log_path=log_path,
                        timeout_s=int(args.progress_db_timeout),
                    )
                except Exception:
                    progress_row_id = None

            t0 = time.time()
            rc = run_trial(cmd, log_path, env, save_dir=ckpt_dir)
            dt = time.time() - t0

            ckpt_info = read_ckpt_metrics(ckpt_path)
            refit_objective_value, log_metrics = objective_value_from_log(log_path, args)
            failure_info = classify_training_failure(
                log_path,
                returncode=int(rc),
                objective_value=refit_objective_value,
            )
            hparams_clean = {k: v for k, v in hp.items() if not k.startswith("_")}

            record = {
                "source_trial_number": t.number,
                "source_objective_value": t.value,
                "rank": rank,
                "session_id": getattr(args, "jsonl_session_id", None),
                "returncode": rc,
                "status": failure_info.get("status"),
                "failure": failure_info if failure_info.get("status") != "ok" else None,
                "run_id": run_id,
                "seconds": dt,
                "phase": "refit",
                "gpus": gpu_ids,
                "master_port": int(master_port),
                "objective_metric": "val_smape_mean",
                "objective_display_name": objective_display_name(args),
                "objective_direction": objective_direction(args),
                "objective_value": refit_objective_value,
                "hparams": hparams_clean,
                "ckpt": ckpt_info,
                "log_metrics": log_metrics,
                "log_path": str(log_path),
                "timestamp_utc": now_utc_compact(),
            }
            try:
                append_jsonl(refit_results_path, record)
            except Exception as e:
                print(
                    f"[refit top{rank:02d}] WARN: failed to append to {refit_results_path}: {e!r}. "
                    "Continuing because training already finished.",
                    flush=True,
                )

            if progress_db_path is not None and progress_row_id is not None:
                status = "ok" if (rc == 0 and refit_objective_value is not None) else "fail"
                progress_db_update_done(
                    progress_db_path,
                    progress_row_id,
                    status=status,
                    returncode=int(rc),
                    seconds=float(dt),
                    best_objective=safe_float(refit_objective_value),
                    error=None if status == "ok" else f"{failure_info.get('failure_type')}: refit failed rc={rc} or missing SMAPE objective",
                    timeout_s=int(args.progress_db_timeout),
                )

            best = log_metrics.get("best", {}) if isinstance(log_metrics, dict) else {}
            print(
                f"[refit top{rank:02d}] rc={rc} "
                f"val_smape_mean={refit_objective_value} "
                f"ptq={best.get('val_smape_ptq')} eta={best.get('val_smape_eta')} phi={best.get('val_smape_phi')} "
                f"ckpt={ckpt_path.name}",
                flush=True,
            )


# -----------------------------
# CLI
# -----------------------------

def _csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def _csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def _csv_strings(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip() != ""]


def _params_key(params: Dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def enqueue_retry_trials(
    study: optuna.Study,
    *,
    retry_failed: bool,
    retry_pruned: bool,
) -> int:
    """Re-enqueue failed/pruned trial parameter sets for another attempt.

    Optuna does not automatically retry FAILED trials. This helper is useful for
    transient infrastructure failures, e.g. EOS/NFS metadata races, CUDA/NCCL
    hiccups, or temporary worker-node problems.
    """
    retry_states = set()
    if retry_failed:
        retry_states.add(optuna.trial.TrialState.FAIL)
    if retry_pruned:
        retry_states.add(optuna.trial.TrialState.PRUNED)

    if not retry_states:
        return 0

    existing_active_or_complete = {
        _params_key(t.params)
        for t in study.trials
        if t.params and t.state in {
            optuna.trial.TrialState.WAITING,
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.COMPLETE,
        }
    }

    enqueued = 0
    for t in study.trials:
        if t.state not in retry_states:
            continue
        if not t.params:
            continue

        key = _params_key(t.params)
        if key in existing_active_or_complete:
            continue

        study.enqueue_trial(
            dict(t.params),
            user_attrs={
                "retry_of_trial_number": int(t.number),
                "retry_of_state": str(t.state.name),
            },
        )
        existing_active_or_complete.add(key)
        enqueued += 1

    return enqueued


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-script", type=Path, required=True)
    ap.add_argument("--data-glob", required=True)
    ap.add_argument("--split-file", required=True)

    ap.add_argument("--out-dir", type=Path, default=Path("./tuning_track_graph_smape"))
    ap.add_argument("--save", default="track_graph_regressor_v2.pt")

    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--n-startup-trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)

    # DDP / GPU scheduling
    ap.add_argument("--nproc-per-node", type=int, default=0, help="0 = auto-detect from per-trial CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--cuda-visible-devices", default=None)
    ap.add_argument("--n-jobs", type=int, default=4, help="Parallel Optuna jobs in fast phase.")
    ap.add_argument("--fast-gpus-per-trial", type=int, default=2)
    ap.add_argument("--refit-gpus-per-trial", type=int, default=8)
    ap.add_argument("--gpu-acquire-timeout-s", type=float, default=None)
    ap.add_argument("--master-port-base", type=int, default=29500)
    ap.add_argument("--omp-num-threads", type=int, default=1)
    ap.add_argument("--mkl-num-threads", type=int, default=1)

    # Restrict architecture family if desired
    ap.add_argument("--fixed-layer-type", default=None,
        choices=["mpnn", "edge_residual", "sage_residual", "gat_residual"],
        help="If set, restrict this Optuna study to one layer_type.",
    )

    # Dataloader/trainer controls
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--prefetch-factor", type=int, default=1, help="Lower values reduce /dev/shm pressure under many DDP jobs.")
    ap.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=True)
    ap.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    ap.add_argument("--worker-start-method", default="spawn", choices=["fork", "forkserver", "spawn"])
    ap.add_argument("--hdf5-use-file-locking", default="FALSE", choices=["FALSE", "TRUE", "BEST_EFFORT"])
    ap.add_argument("--h5-open-retries", type=int, default=8)
    ap.add_argument("--h5-open-retry-sleep", type=float, default=0.5)
    ap.add_argument("--h5-read-retries", type=int, default=5)
    ap.add_argument("--h5-read-retry-sleep", type=float, default=0.2)
    ap.add_argument("--h5-max-open-files-per-worker", type=int, default=8)
    ap.add_argument("--h5-close-after-read", action="store_true", default=False)
    ap.add_argument("--h5-swmr", action="store_true", default=False)
    ap.add_argument("--worker-open-jitter-s", type=float, default=0.5)
    ap.add_argument("--trainer-amp", dest="trainer_amp", action="store_true", default=True)
    ap.add_argument("--no-trainer-amp", dest="trainer_amp", action="store_false")
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])

    # Fast proxy phase
    ap.add_argument("--fast-epochs", type=int, default=40)
    ap.add_argument("--fast-max-train-graphs", type=int, default=20000)
    ap.add_argument("--fast-early-stop-patience", type=int, default=10)

    # Refit phase
    ap.add_argument("--refit-topk", type=int, default=5)
    ap.add_argument("--refit-epochs", type=int, default=200)
    ap.add_argument("--refit-max-train-graphs", type=int, default=-1)
    ap.add_argument("--refit-ranks", type=int, nargs="+", default=None)
    ap.add_argument("--refit-early-stop-patience", type=int, default=30)
    ap.add_argument("--refit-resume", action="store_true", default=False,
                    help="Forward --resume during refit. Default is --no-resume for clean final fits.")

    ap.add_argument("--resume-skip-existing", action="store_true", default=True)
    ap.add_argument("--no-resume-skip-existing", dest="resume_skip_existing", action="store_false")
    ap.add_argument("--resume-skip-existing-refit", action="store_true", default=True)
    ap.add_argument("--no-resume-skip-existing-refit", dest="resume_skip_existing_refit", action="store_false")

    # Objective
    ap.add_argument("--objective-component-weights", default="1.0,1.0,1.0",
                    help="Weights for overall SMAPE objective as ptq,eta,phi. Default = unweighted mean.")

    # Trainer early stop still uses trainer-supported monitor.
    ap.add_argument("--early-stop-monitor", default="val_loss", choices=["val_loss", "val_rmse_mean"],
                    help="Trainer checkpoint/early-stop monitor. Optuna objective remains log-parsed mean SMAPE.")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0)
    ap.add_argument("--no-early-stop", action="store_true", default=False)
    ap.add_argument("--reload-best-half-patience", action="store_true", default=False)

    # Search-space ranges
    ap.add_argument("--lr-low", type=float, default=5e-5)
    ap.add_argument("--lr-high", type=float, default=8e-4)
    ap.add_argument("--weight-decay-low", type=float, default=1e-5)
    ap.add_argument("--weight-decay-high", type=float, default=5e-2)
    ap.add_argument("--dropout-low", type=float, default=0.0)
    ap.add_argument("--dropout-high", type=float, default=0.30)
    ap.add_argument("--hidden-dim-choices", type=_csv_ints, default=[96, 128, 160, 192, 256],
                    help="Comma-separated, e.g. 96,128,192,256")
    ap.add_argument("--layer-choices", type=_csv_ints, default=[3, 4, 5, 6, 8],
                    help="Comma-separated, e.g. 3,4,5,6")
    ap.add_argument("--edge-dropout-low", type=float, default=0.0)
    ap.add_argument("--edge-dropout-high", type=float, default=0.20)
    ap.add_argument("--feat-noise-choices", type=_csv_floats, default=[0.0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
                    help="Comma-separated, e.g. 0,0.001,0.002,0.005")
    ap.add_argument("--ema-decay-choices", type=_csv_floats, default=[0.995, 0.998, 0.999, 0.9995, 0.9998])

    ap.add_argument("--lr-schedule-choices", type=_csv_strings, default=["plateau", "cosine"],
                    help="Comma-separated subset of plateau,cosine")
    ap.add_argument("--warmup-epoch-choices", type=_csv_floats, default=[0.0, 1.0, 2.0, 3.0])
    ap.add_argument("--min-lr-ratio-choices", type=_csv_floats, default=[0.02, 0.05, 0.10])
    ap.add_argument("--lr-plateau-factor", type=float, default=0.5)
    ap.add_argument("--lr-plateau-patience", type=int, default=5)
    ap.add_argument("--lr-plateau-min-lr", type=float, default=0.0)

    # Regression-specific search controls
    ap.add_argument("--loss-type-choices", type=_csv_strings, default=["smoothl1", "l1", "mse"])
    ap.add_argument("--target-scale-choices", type=_csv_strings, default=["robust", "standard", "none"])
    ap.add_argument("--target-scale-eps", type=float, default=1e-6)
    ap.add_argument("--target-stats-max-events", type=int, default=-1)
    ap.add_argument("--graph-pool-choices", type=_csv_strings, default=["meanmax", "mean", "max"])

    ap.add_argument("--fixed-phi-mode", default=None, choices=["sincos", "scalar"])
    ap.add_argument("--tune-phi-mode", action="store_true", default=False)
    ap.add_argument("--phi-period", type=float, default=2.0 * math.pi)
    ap.add_argument("--phi-vec-weight-low", type=float, default=0.25)
    ap.add_argument("--phi-vec-weight-high", type=float, default=4.0)

    ap.add_argument("--tune-target-loss-weights", action="store_true", default=True,
                    help="Tune ptq/eta/phi loss weights passed to trainer.")
    ap.add_argument("--fixed-target-loss-weights", default="1.0,1.0,1.0",
                    help="Used only with --no-tune-target-loss-weights.")
    ap.add_argument("--no-tune-target-loss-weights", dest="tune_target_loss_weights", action="store_false")
    ap.add_argument("--target-loss-weight-low", type=float, default=0.5)
    ap.add_argument("--target-loss-weight-high", type=float, default=2.5)

    ap.add_argument("--tune-fourier", action="store_true", default=True)
    ap.add_argument("--no-tune-fourier", dest="tune_fourier", action="store_false")
    ap.add_argument("--fixed-fourier", action="store_true", default=True)
    ap.add_argument("--no-fixed-fourier", dest="fixed_fourier", action="store_false")
    ap.add_argument("--tune-fourier-knobs", action="store_true", default=False)
    ap.add_argument("--fourier-base", type=float, default=3.0)
    ap.add_argument("--fourier-min-exp", type=int, default=-6)
    ap.add_argument("--fourier-max-exp", type=int, default=6)
    ap.add_argument("--fourier-base-choices", type=_csv_floats, default=[2.0, 3.0, 4.0])
    ap.add_argument("--fourier-min-exp-choices", type=_csv_ints, default=[-6, -5, -4])
    ap.add_argument("--fourier-max-exp-choices", type=_csv_ints, default=[4, 5, 6])

    # W&B
    ap.add_argument("--wandb-mode", default="disabled", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-project", default="track_graph_regressor_tuning")

    ap.add_argument("--study-name", default="track_graph_mean_smape")
    ap.add_argument("--code-version", default=None)

    ap.add_argument("--refit-only", action="store_true", default=False,
                    help="Skip fast Optuna optimization and only refit from an existing study.")

    # SQLite / progress DB
    ap.add_argument("--storage-path", type=Path, default=None,
                    help="Path to Optuna SQLite DB file. Prefer local scratch for n_jobs>1.")
    ap.add_argument("--no-optuna-sqlite", action="store_true", default=False,
                    help="Use Optuna in-memory storage and use results_fast.jsonl/results_refit.jsonl for durable records.")
    ap.add_argument("--jsonl-session-id", default=None,
                    help="Session tag written to JSONL records. Defaults to a fresh UTC timestamp; useful when reusing out-dir.")
    ap.add_argument("--sqlite-timeout", type=int, default=300)
    ap.add_argument("--sqlite-wal", dest="sqlite_wal", action="store_true", default=True)
    ap.add_argument("--no-sqlite-wal", dest="sqlite_wal", action="store_false")
    ap.add_argument("--sqlite-pool", default="null", choices=["null", "default"])

    ap.add_argument("--progress-db", type=Path, default=None)
    ap.add_argument("--no-progress-db", action="store_true", default=False)
    ap.add_argument("--progress-db-timeout", type=int, default=30)
    ap.add_argument("--retry-failed-trials", action="store_true", default=False,
                    help="Re-enqueue FAILED trials from this study before optimizing.")
    ap.add_argument("--retry-pruned-trials", action="store_true", default=False,
                    help="Also re-enqueue PRUNED trials. Use carefully; many pruned trials may be genuinely bad.")
    ap.add_argument("--continue-on-trial-error", dest="continue_on_trial_error", action="store_true", default=True,
                    help="Keep the study running when a trial raises RuntimeError/OSError.")
    ap.add_argument("--stop-on-trial-error", dest="continue_on_trial_error", action="store_false")

    args = ap.parse_args()

    args.jsonl_session_id_user_supplied = args.jsonl_session_id is not None
    args.jsonl_session_id = str(args.jsonl_session_id or now_utc_compact())

    args.out_dir = Path(args.out_dir).expanduser().resolve()
    args.train_script = Path(args.train_script).expanduser().resolve()

    if args.refit_ranks is not None:
        bad = [r for r in args.refit_ranks if int(r) < 1]
        if bad:
            raise SystemExit(f"--refit-ranks must be >= 1, got: {bad}")
        seen = set()
        args.refit_ranks = [int(r) for r in args.refit_ranks if not (int(r) in seen or seen.add(int(r)))]

    base_dir = Path.cwd()
    args.data_glob = _abspath_glob(args.data_glob, base_dir=base_dir)
    args.split_file = str(Path(args.split_file).expanduser().resolve())

    if args.storage_path is not None:
        args.storage_path = Path(args.storage_path).expanduser().resolve()
    if args.progress_db is not None:
        args.progress_db = Path(args.progress_db).expanduser().resolve()

    if not args.train_script.exists():
        raise SystemExit(f"--train-script does not exist: {args.train_script}")

    validate_training_inputs(data_glob=args.data_glob, split_file=Path(args.split_file))

    warnings.filterwarnings(
        "ignore",
        message=r"Argument ``multivariate`` is an experimental feature\.",
        category=optuna.exceptions.ExperimentalWarning,
    )

    base_devices = parse_cuda_visible_devices(args.cuda_visible_devices)
    if len(base_devices) < 1:
        raise SystemExit("No CUDA devices visible to tuner.")

    try:
        torch_n = int(torch.cuda.device_count())
    except Exception:
        torch_n = -1
    print(
        f"[gpu-detect] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')!r} "
        f"torch.cuda.device_count()={torch_n} pool={base_devices}",
        flush=True,
    )

    max_parallel = max(1, len(base_devices) // max(1, int(args.fast_gpus_per_trial)))
    if args.n_jobs > max_parallel:
        print(
            f"[warn] --n-jobs={args.n_jobs} is too high for ngpus={len(base_devices)} and "
            f"--fast-gpus-per-trial={args.fast_gpus_per_trial}. Clamping to {max_parallel}.",
            flush=True,
        )
        args.n_jobs = max_parallel

    mkdir(args.out_dir)
    ckpt_dir = (args.out_dir / "ckpts").resolve()
    mkdir(ckpt_dir)
    ensure_dir_writable(args.out_dir, "--out-dir")
    ensure_dir_writable(ckpt_dir, "checkpoint (--out-dir/ckpts)")
    ensure_dir_writable((args.out_dir / "logs").resolve(), "logs (--out-dir/logs)")

    results_fast = args.out_dir / "results_fast.jsonl"
    results_refit = args.out_dir / "results_refit.jsonl"
    ensure_dir_writable(results_fast.parent, "results (--out-dir)")

    storage_path: Optional[Path] = None

    if args.no_optuna_sqlite:
        if args.refit_only:
            print(
                "[storage] --refit-only with --no-optuna-sqlite: loading finalists from results_fast.jsonl.",
                flush=True,
            )

        if args.storage_path is not None:
            print(
                "[storage] WARN: --storage-path was provided but will be ignored "
                "because --no-optuna-sqlite is set.",
                flush=True,
            )

        storage = None  # Optuna uses in-memory storage.
        print("[storage] Using Optuna in-memory storage: no Optuna SQLite DB will be written.", flush=True)

    else:
        storage_path = (
            Path(args.storage_path)
            if args.storage_path is not None
            else (args.out_dir / "optuna_study.db").resolve()
        )
        mkdir(storage_path.parent)
        print(f"[storage] Using SQLite DB at: {storage_path}", flush=True)

        storage = make_sqlite_storage(
            storage_path,
            timeout_s=int(args.sqlite_timeout),
            enable_wal=bool(args.sqlite_wal),
            pool=str(args.sqlite_pool),
        )

    progress_db_path: Optional[Path] = None
    if not args.no_progress_db:
        progress_db_path = (args.progress_db if args.progress_db is not None else (args.out_dir / "progress.sqlite")).resolve()
        try:
            progress_db_init(progress_db_path, timeout_s=int(args.progress_db_timeout))
            print(f"[progress-db] Writing trial progress to: {progress_db_path}", flush=True)
        except Exception as e:
            print(f"[progress-db] WARN: could not init progress DB at {progress_db_path}: {e!r}", flush=True)
            progress_db_path = None

    sampler = optuna.samplers.TPESampler(
        seed=int(args.seed),
        n_startup_trials=int(args.n_startup_trials),
        multivariate=True,
    )

    allocator = GPUAllocator(base_devices)
    objective = objective_factory(
        args,
        ckpt_dir,
        results_fast,
        allocator,
        progress_db_path,
        session_id=args.jsonl_session_id,
    )

    study = optuna.create_study(
        study_name=str(args.study_name),
        direction=objective_direction(args),
        sampler=sampler,
        storage=storage,
        load_if_exists=not bool(args.no_optuna_sqlite),
    )

    n_requeued = enqueue_retry_trials(
        study,
        retry_failed=bool(args.retry_failed_trials),
        retry_pruned=bool(args.retry_pruned_trials),
    )
    if n_requeued:
        print(f"[retry] Enqueued {n_requeued} previous failed/pruned trial configs for retry.", flush=True)

    print(
        f"[objective] metric=val_smape_mean display={objective_display_name(args)} "
        f"direction={objective_direction(args)}",
        flush=True,
    )
    print(
        f"[objective] tuner will parse validation SMAPE from trainer logs; "
        f"trainer checkpoint monitor remains --early-stop-monitor={args.early_stop_monitor}",
        flush=True,
    )

    if args.refit_only:
        print("[mode] --refit-only set: skipping FAST Optuna optimization; loading existing study only.", flush=True)
    else:
        catch = (RuntimeError, OSError) if args.continue_on_trial_error else ()
        study.optimize(objective, n_trials=int(args.n_trials), n_jobs=int(args.n_jobs), gc_after_trial=True, catch=catch,)

    print(f"\n=== FAST PHASE BEST ({objective_display_name(args)}) ===")
    jsonl_complete: Optional[List[JsonlTrialLike]] = None
    if args.no_optuna_sqlite:
        # In no-SQLite mode, the JSONL files are the durable source of truth.
        # During normal fast+refit runs, restrict to the current session to avoid
        # accidentally refitting stale trials from an older run in the same out-dir.
        session_filter = (
            args.jsonl_session_id
            if (not args.refit_only or args.jsonl_session_id_user_supplied)
            else None
        )
        jsonl_complete = completed_trials_from_results_jsonl(results_fast, session_id=session_filter)
        print(
            f"[jsonl] completed fast trials available for ranking: {len(jsonl_complete)} "
            f"(session_filter={session_filter!r})",
            flush=True,
        )
        print_fast_phase_best_from_trials(jsonl_complete, args)
    else:
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        print_fast_phase_best_from_trials(complete, args)

    if storage_path is None:
        print("Optuna DB: <disabled; in-memory only>")
    else:
        print(
            f"Optuna DB: {storage_path} "
            f"(sqlite_wal={args.sqlite_wal} timeout={args.sqlite_timeout}s pool={args.sqlite_pool})"
        )
    print(f"Fast JSONL: {results_fast}")
    print(f"JSONL session_id: {args.jsonl_session_id}")

    refit_topk(
        args,
        ckpt_dir,
        study if not args.no_optuna_sqlite else None,
        int(args.refit_topk),
        results_refit,
        allocator,
        progress_db_path,
        trials_override=jsonl_complete if args.no_optuna_sqlite else None,
    )
    print(f"\nRefit JSONL: {results_refit}")


if __name__ == "__main__":
    main()
