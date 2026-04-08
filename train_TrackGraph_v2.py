#!/usr/bin/env python3
"""
train_TrackGraph_v2.py

Main changes vs v1:
- The training loss is explicit per target instead of a blind mean over all outputs.
- phi can be trained as a 2D angle head (sin(phi), cos(phi)) to avoid wrap/discontinuity issues.
- MAPE is reported only for the pt-like target by default; eta/phi instead use SMAPE,
  because percentage error is usually ill-defined around zero for eta and especially phi.
- Per-component losses are logged, so you can see whether one target dominates.
- Optional manual target weights allow you to stop one component from dominating.

The rest of the data format and most CLI conventions are kept close to v1.
"""

import argparse
import atexit
import faulthandler
import gc
import glob
import json
import math
import os
import random
import signal
import time
import traceback
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import wandb


# ----------------------------
# IO helpers
# ----------------------------

def _ensure_parent_dir(path: str) -> None:
    try:
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _torch_save_atomic_with_retries(
    obj: Dict[str, Any],
    final_path: str,
    *,
    retries: int = 6,
    base_sleep_s: float = 1.0,
) -> None:
    final_p = Path(final_path).expanduser().resolve()
    parent = final_p.parent
    tmp_p = parent / (final_p.name + ".tmp")

    last_err: Optional[Exception] = None
    for i in range(int(retries)):
        try:
            parent.mkdir(parents=True, exist_ok=True)
            torch.save(obj, str(tmp_p))
            os.replace(str(tmp_p), str(final_p))
            return
        except Exception as e:
            last_err = e
            try:
                if tmp_p.exists():
                    tmp_p.unlink()
            except Exception:
                pass
            sleep_s = base_sleep_s * (2 ** i)
            print(
                f"[ckpt] WARN: save failed (attempt {i+1}/{retries}) to {final_p}: "
                f"{type(e).__name__}: {e}. Retrying in {sleep_s:.1f}s",
                flush=True,
            )
            time.sleep(sleep_s)

    raise RuntimeError(
        f"[ckpt] Failed to save checkpoint to {final_p} after {retries} attempts: {last_err!r}"
    ) from last_err


def _build_save_path(args, run_id: str) -> str:
    base = os.path.basename(args.save)
    stem, ext = os.path.splitext(base)
    ext = ext if ext else ".pt"
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        return os.path.join(args.save_dir, f"{stem}_{run_id}{ext}")
    return os.path.join(os.path.dirname(args.save) or ".", f"{stem}_{run_id}{ext}")


def _append_line_atomic(path: str, line: str) -> None:
    _ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def _jsonable_error(exc: Exception) -> Dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


# ----------------------------
# DDP helpers
# ----------------------------

def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def ddp_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0


def ddp_world_size() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1


def ddp_is_main() -> bool:
    return ddp_rank() == 0


def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if not torch.cuda.is_available():
            raise SystemExit("[ddp] CUDA is not available but torchrun/DDP was requested.")

        n_visible = torch.cuda.device_count()
        if n_visible <= 0:
            raise SystemExit("[ddp] No CUDA devices visible.")

        if local_rank < 0 or local_rank >= n_visible:
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
            raise SystemExit(
                f"[ddp] LOCAL_RANK={local_rank} but only {n_visible} CUDA device(s) are visible. "
                f"CUDA_VISIBLE_DEVICES={cvd}."
            )

        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)


def ddp_cleanup():
    if ddp_is_initialized():
        dist.destroy_process_group()


def ddp_all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if ddp_is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def ddp_barrier():
    if ddp_is_initialized():
        dist.barrier()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def timed_section(name: str, device: torch.device, enabled: bool = True):
    t = {"seconds": 0.0}
    if not enabled:
        yield t
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t["seconds"] = time.perf_counter() - t0


# ----------------------------
# Dataset
# ----------------------------

class H5TrackGraphDataset(Dataset):
    """
    Expected structure:
      /graphs/<id>/x
      /graphs/<id>/edge_index
      /graphs/<id>/edge_attr
      /graphs/<id>/y_track   # [ptq, eta, phi]
    """

    def __init__(self, h5_paths):
        self.h5_paths = list(h5_paths)
        if not self.h5_paths:
            raise ValueError("No H5 files provided.")

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

        self._files = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = None
        state["_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
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
        g = self._files[fi]["graphs"][k]

        x = torch.from_numpy(g["x"][...]).float()
        edge_index = torch.from_numpy(g["edge_index"][...]).long()
        edge_attr = torch.from_numpy(g["edge_attr"][...]).float()

        if "y_track" not in g:
            raise RuntimeError(
                f"Missing 'y_track' in {self.h5_paths[fi]} /graphs/{k}. "
                f"Expected y_track = [ptq, eta, phi]."
            )
        y_track = torch.from_numpy(g["y_track"][...]).float()

        out = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y_track": y_track,
            "graph_key": k,
            "source_path": self.h5_paths[fi],
        }

        if "original_node_ids" in g:
            out["original_node_ids"] = torch.from_numpy(g["original_node_ids"][...]).long()

        return out


def collate_one(batch):
    assert len(batch) == 1
    return batch[0]


def _batch_id_string(batch: Dict[str, Any]) -> str:
    graph_key = batch.get("graph_key", "<unknown_graph>")
    source_path = batch.get("source_path", "<unknown_file>")
    return f"{source_path} :: graphs/{graph_key}"


def validate_graph_batch(batch: Dict[str, Any], *, where: str = "") -> None:
    bid = _batch_id_string(batch)

    for req in ("x", "edge_index", "edge_attr", "y_track"):
        if req not in batch:
            raise RuntimeError(f"[validate:{where}] missing key '{req}' in batch {bid}")

    x = batch["x"]
    edge_index = batch["edge_index"]
    edge_attr = batch["edge_attr"]
    y = batch["y_track"]

    if x.ndim != 2:
        raise RuntimeError(f"[validate:{where}] x must have shape [N, F], got {tuple(x.shape)} in {bid}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise RuntimeError(
            f"[validate:{where}] edge_index must have shape [2, E], got {tuple(edge_index.shape)} in {bid}"
        )
    if edge_attr.ndim != 2:
        raise RuntimeError(
            f"[validate:{where}] edge_attr must have shape [E, Fe], got {tuple(edge_attr.shape)} in {bid}"
        )
    if y.ndim != 1 or y.numel() != 3:
        raise RuntimeError(
            f"[validate:{where}] y_track must have shape [3], got {tuple(y.shape)} in {bid}"
        )

    if edge_index.shape[1] != edge_attr.shape[0]:
        raise RuntimeError(
            f"[validate:{where}] edge count mismatch: edge_index has E={edge_index.shape[1]} "
            f"but edge_attr has E={edge_attr.shape[0]} in {bid}"
        )

    if x.shape[0] <= 0:
        raise RuntimeError(f"[validate:{where}] graph has no nodes in {bid}")

    if edge_index.numel() > 0:
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise RuntimeError(
                f"[validate:{where}] edge_index must be integer, got {edge_index.dtype} in {bid}"
            )
        ei_min = int(edge_index.min().item())
        ei_max = int(edge_index.max().item())
        n_nodes = int(x.shape[0])
        if ei_min < 0 or ei_max >= n_nodes:
            bad_mask = (edge_index < 0) | (edge_index >= n_nodes)
            bad_pos = bad_mask.nonzero(as_tuple=False)
            bad_preview = bad_pos[:8].tolist()
            raise RuntimeError(
                f"[validate:{where}] invalid edge_index in {bid}: "
                f"min={ei_min}, max={ei_max}, n_nodes={n_nodes}, "
                f"bad_positions={bad_preview}"
            )

    for name, t in (("x", x), ("edge_attr", edge_attr), ("y_track", y)):
        if not torch.isfinite(t).all():
            nz = (~torch.isfinite(t)).nonzero(as_tuple=False)
            preview = nz[:8].tolist()
            raise RuntimeError(
                f"[validate:{where}] non-finite values found in {name} for {bid}; positions={preview}"
            )

    if not x.is_contiguous():
        batch["x"] = x.contiguous()
    if not edge_index.is_contiguous():
        batch["edge_index"] = edge_index.contiguous()
    if not edge_attr.is_contiguous():
        batch["edge_attr"] = edge_attr.contiguous()
    if not y.is_contiguous():
        batch["y_track"] = y.contiguous()


# ----------------------------
# EMA
# ----------------------------

@dataclass
class EMA:
    decay: float
    shadow: dict

    @staticmethod
    def create(model: nn.Module, decay: float):
        raw = model.module if hasattr(model, "module") else model
        shadow = {k: v.detach().clone() for k, v in raw.state_dict().items()}
        return EMA(decay=decay, shadow=shadow)

    @torch.no_grad()
    def update(self, model: nn.Module):
        raw = model.module if hasattr(model, "module") else model
        msd = raw.state_dict()
        for k, v in msd.items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def apply_to(self, model: nn.Module):
        raw = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            cur = raw.state_dict()
            backup = {}
            for k, v in cur.items():
                backup[k] = v.detach().clone()
                if k in self.shadow:
                    v.copy_(self.shadow[k])
        try:
            yield
        finally:
            with torch.no_grad():
                cur2 = raw.state_dict()
                for k, v in cur2.items():
                    if k in backup:
                        v.copy_(backup[k])


# ----------------------------
# Model blocks
# ----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, n_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CostumEdgeConvLayer(nn.Module):
    def __init__(self, nn_module: nn.Module, aggregation: str = "mean", add_self_loops: bool = True):
        super().__init__()
        self.nn = nn_module
        self.aggregation = aggregation
        self.add_self_loops = add_self_loops

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]

        if self.add_self_loops:
            self_loops = torch.arange(x.size(0), device=x.device).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
            src = edge_index[0]
            dst = edge_index[1]

        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        edge_out = self.nn(edge_features)

        aggregated_out = torch.zeros((x.size(0), edge_out.size(1)), device=x.device, dtype=edge_out.dtype)

        if self.aggregation == "mean":
            index = dst.unsqueeze(-1).expand_as(edge_out)
            aggregated_out.scatter_add_(0, index, edge_out)
            counts = torch.zeros(x.size(0), device=x.device, dtype=edge_out.dtype)
            counts.scatter_add_(0, dst, torch.ones_like(dst, dtype=edge_out.dtype))
            aggregated_out = aggregated_out / counts.clamp(min=1).unsqueeze(-1)
        elif self.aggregation == "max":
            aggregated_out = torch.full_like(aggregated_out, float("-inf"))
            index = dst.unsqueeze(-1).expand_as(edge_out)
            if hasattr(aggregated_out, "scatter_reduce_"):
                aggregated_out.scatter_reduce_(0, index, edge_out, reduce="amax", include_self=True)
            else:
                aggregated_out.scatter_(0, index, edge_out)
            aggregated_out = torch.where(torch.isfinite(aggregated_out), aggregated_out, torch.zeros_like(aggregated_out))
        elif self.aggregation == "sum":
            index = dst.unsqueeze(-1).expand_as(edge_out)
            aggregated_out.scatter_add_(0, index, edge_out)
        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggregation}")

        return aggregated_out


class CustomGAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.add_self_loops = add_self_loops
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.attn_l = nn.Parameter(torch.empty(1, heads, out_channels))
        self.attn_r = nn.Parameter(torch.empty(1, heads, out_channels))
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

        if not concat:
            self.out_proj = nn.Linear(heads * out_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        x = self.linear(x).view(num_nodes, self.heads, self.out_channels)
        out_dtype = x.dtype
        x_f = x.float()

        if self.add_self_loops:
            self_loops = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)

        src = edge_index[0]
        dst = edge_index[1]

        alpha_l = (x_f[src] * self.attn_l).sum(dim=-1)
        alpha_r = (x_f[dst] * self.attn_r).sum(dim=-1)
        alpha = F.leaky_relu(alpha_l + alpha_r, negative_slope=0.2)

        alpha = torch.exp(alpha - alpha.max(dim=0, keepdim=True)[0])
        alpha_sum = torch.zeros((num_nodes, self.heads), device=x.device, dtype=torch.float32)
        alpha_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(alpha), alpha)
        alpha = alpha / alpha_sum[dst].clamp(min=1e-6)
        alpha = self.dropout(alpha)

        out = torch.zeros((num_nodes, self.heads, self.out_channels), device=x.device, dtype=torch.float32)
        for h in range(self.heads):
            out[:, h].scatter_add_(
                0,
                dst.unsqueeze(-1).expand_as(x_f[src, h]),
                alpha[:, h].unsqueeze(-1) * x_f[src, h],
            )

        if self.concat:
            return out.view(num_nodes, self.heads * self.out_channels).to(out_dtype)
        out = out.mean(dim=1)
        return self.out_proj(out.to(out_dtype))


class CustomSAGEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean", normalize: bool = True, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.aggr = aggr
        self.normalize = normalize
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        agg_features = torch.zeros((num_nodes, self.in_channels), device=x.device, dtype=x.dtype)

        if self.aggr == "mean":
            agg_features.scatter_add_(0, dst.unsqueeze(-1).expand_as(x[src]), x[src])
            counts = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
            counts.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            agg_features = agg_features / counts.clamp(min=1).unsqueeze(-1)
        elif self.aggr == "sum":
            agg_features.scatter_add_(0, dst.unsqueeze(-1).expand_as(x[src]), x[src])
        elif self.aggr == "max":
            agg_features = torch.full((num_nodes, self.in_channels), float("-inf"), device=x.device, dtype=x.dtype)
            index = dst.unsqueeze(-1).expand_as(x[src])
            if hasattr(agg_features, "scatter_reduce_"):
                agg_features.scatter_reduce_(0, index, x[src], reduce="amax", include_self=True)
            else:
                agg_features.scatter_(0, index, x[src])
            agg_features = torch.where(torch.isfinite(agg_features), agg_features, torch.zeros_like(agg_features))
        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggr}")

        h_neigh = self.lin_neigh(agg_features)
        h_self = self.lin_self(x)
        h = h_self + h_neigh
        if self.bias is not None:
            h = h + self.bias
        if self.normalize:
            h = torch.where(torch.isfinite(h), h, torch.zeros_like(h))
            h = F.normalize(h, p=2.0, dim=-1)
        return h


class Edge_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float, aggr: str = "mean", add_self_loops: bool = True):
        super().__init__()
        self.project = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None
        self.edge_conv1 = CostumEdgeConvLayer(
            nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            aggregation=aggr,
            add_self_loops=add_self_loops,
        )
        self.edge_conv2 = CostumEdgeConvLayer(
            nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            aggregation=aggr,
            add_self_loops=add_self_loops,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.project is not None:
            x = self.project(x)
        identity = x
        x = F.relu(self.edge_conv1(x, edge_index))
        x = self.edge_conv2(x, edge_index)
        x = self.dropout(x)
        return identity + x


class GATResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.project = nn.Linear(in_channels, out_channels * heads) if in_channels != out_channels * heads else None
        self.gat = CustomGAT(
            in_channels=(out_channels * heads if self.project else in_channels),
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            concat=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.project is not None:
            x = self.project(x)
        identity = x
        x = F.relu(self.gat(x, edge_index))
        x = self.dropout(x)
        return identity + x


class SAGEResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean", dropout: float = 0.2, normalize: bool = True):
        super().__init__()
        self.project = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.conv = CustomSAGEConv(in_channels=out_channels, out_channels=out_channels, aggr=aggr, normalize=normalize)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.project is not None:
            x = self.project(x)
        identity = x
        x = F.relu(self.conv(x, edge_index))
        x = self.dropout(x)
        return identity + x


class FourierEncoder(nn.Module):
    def __init__(self, xdim: int, base: float = 3.0, min_exp: int = -6, max_exp: int = 6):
        super().__init__()
        self.exps = list(range(int(min_exp), int(max_exp) + 1))
        divs = torch.tensor([float(base) ** e for e in self.exps], dtype=torch.float32)
        self.register_buffer("divs", divs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = x.unsqueeze(-1) / self.divs.view(1, 1, -1)
        out = torch.cat([torch.sin(x3), torch.cos(x3)], dim=-1)
        return out.reshape(x.size(0), -1)


class EdgeMPNNLayer(nn.Module):
    def __init__(self, hdim, edim, msg_hidden=128, upd_hidden=128, dropout=0.0):
        super().__init__()
        self.edge_mlp = MLP(in_dim=2 * hdim + edim, out_dim=hdim, hidden_dim=msg_hidden, n_layers=2, dropout=dropout)
        self.node_mlp = MLP(in_dim=2 * hdim, out_dim=hdim, hidden_dim=upd_hidden, n_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(hdim)

    def forward(self, h, edge_index, edge_attr, edge_dropout_p: float = 0.0):
        E = edge_attr.size(0)
        if E == 0:
            agg = torch.zeros_like(h)
            h_upd = self.node_mlp(torch.cat([h, agg], dim=1))
            return self.norm(h + h_upd)

        src = edge_index[0]
        dst = edge_index[1]

        if self.training and edge_dropout_p > 0.0:
            keep = (torch.rand(E, device=edge_attr.device) >= edge_dropout_p)
            if keep.sum().item() == 0:
                j = torch.randint(0, E, (1,), device=keep.device).item()
                keep[j] = True
            src = src[keep]
            dst = dst[keep]
            edge_attr = edge_attr[keep]

        if edge_attr.size(0) == 0:
            agg = torch.zeros_like(h)
            h_upd = self.node_mlp(torch.cat([h, agg], dim=1))
            return self.norm(h + h_upd)

        h_src = h[src]
        h_dst = h[dst]
        m_in = torch.cat([h_src, h_dst, edge_attr], dim=1)
        m = self.edge_mlp(m_in)

        agg = torch.zeros((h.size(0), m.size(1)), device=h.device, dtype=m.dtype)
        agg.index_add_(0, dst, m.to(agg.dtype))
        if agg.dtype != h.dtype:
            agg = agg.to(h.dtype)

        h_upd = self.node_mlp(torch.cat([h, agg], dim=1))
        return self.norm(h + h_upd)


class TrackGraphRegressorGNN(nn.Module):
    def __init__(
        self,
        xdim,
        edim,
        hdim=128,
        n_layers=4,
        dropout=0.1,
        layer_type: str = "mpnn",
        gat_heads: int = 4,
        sage_aggr: str = "mean",
        edgeconv_aggr: str = "mean",
        use_fourier=False,
        fourier_base=3.0,
        fourier_min_exp=-6,
        fourier_max_exp=6,
        graph_pool: str = "meanmax",
        phi_mode: str = "sincos",
    ):
        super().__init__()
        self.fourier = None
        self.layer_type = layer_type
        self.gat_heads = gat_heads
        self.sage_aggr = sage_aggr
        self.edgeconv_aggr = edgeconv_aggr
        self.graph_pool = graph_pool
        self.phi_mode = phi_mode

        if use_fourier:
            self.fourier = FourierEncoder(xdim, base=fourier_base, min_exp=fourier_min_exp, max_exp=fourier_max_exp)
            xdim_in = xdim * 2 * (fourier_max_exp - fourier_min_exp + 1)
        else:
            xdim_in = xdim

        self.node_enc = MLP(xdim_in, hdim, hidden_dim=hdim, n_layers=2, dropout=dropout)

        if layer_type == "mpnn":
            self.layers = nn.ModuleList([EdgeMPNNLayer(hdim, edim, dropout=dropout) for _ in range(n_layers)])
            self._uses_edge_attr = True
        elif layer_type == "edge_residual":
            self.layers = nn.ModuleList([
                Edge_ResidualBlock(in_channels=hdim, hidden_channels=hdim, dropout=dropout, aggr=edgeconv_aggr)
                for _ in range(n_layers)
            ])
            self._uses_edge_attr = False
        elif layer_type == "sage_residual":
            self.layers = nn.ModuleList([
                SAGEResidualBlock(in_channels=hdim, out_channels=hdim, aggr=sage_aggr, dropout=dropout)
                for _ in range(n_layers)
            ])
            self._uses_edge_attr = False
        elif layer_type == "gat_residual":
            if gat_heads <= 0:
                raise ValueError("--gat-heads must be >= 1")
            if hdim % gat_heads != 0:
                raise ValueError(f"hidden_dim={hdim} must be divisible by gat_heads={gat_heads}")
            per_head = hdim // gat_heads
            self.layers = nn.ModuleList([
                GATResidualBlock(in_channels=hdim, out_channels=per_head, heads=gat_heads, dropout=dropout)
                for _ in range(n_layers)
            ])
            self._uses_edge_attr = False
        else:
            raise ValueError(f"Unknown layer_type={layer_type}")

        pooled_dim = hdim if graph_pool in ("mean", "max") else 2 * hdim
        self.head_pt = MLP(pooled_dim, 1, hidden_dim=hdim, n_layers=2, dropout=dropout)
        self.head_eta = MLP(pooled_dim, 1, hidden_dim=hdim, n_layers=2, dropout=dropout)
        phi_out_dim = 2 if phi_mode == "sincos" else 1
        self.head_phi = MLP(pooled_dim, phi_out_dim, hidden_dim=hdim, n_layers=2, dropout=dropout)

    def _pool_graph(self, h: torch.Tensor) -> torch.Tensor:
        pooled_dim = self.head_pt.net[0].in_features
        if h.size(0) == 0:
            return h.new_zeros((1, pooled_dim), dtype=h.dtype)

        if self.graph_pool == "mean":
            return h.mean(dim=0, keepdim=True)
        if self.graph_pool == "max":
            return h.max(dim=0, keepdim=True).values
        if self.graph_pool == "meanmax":
            h_mean = h.mean(dim=0, keepdim=True)
            h_max = h.max(dim=0, keepdim=True).values
            return torch.cat([h_mean, h_max], dim=1)
        raise ValueError(f"Unsupported graph_pool={self.graph_pool}")

    def forward(self, x, edge_index, edge_attr, edge_dropout_p: float = 0.0):
        if self.fourier is not None:
            x = self.fourier(x)

        h = self.node_enc(x)
        for layer in self.layers:
            if self._uses_edge_attr:
                h = layer(h, edge_index, edge_attr, edge_dropout_p=edge_dropout_p)
            else:
                h = layer(h, edge_index)

        g = self._pool_graph(h)
        pt = self.head_pt(g).squeeze(0).squeeze(-1)
        eta = self.head_eta(g).squeeze(0).squeeze(-1)
        phi_raw = self.head_phi(g).squeeze(0)
        return {
            "pt": pt,
            "eta": eta,
            "phi_raw": phi_raw,
        }


# ----------------------------
# Stats / scaling
# ----------------------------

def _make_stats_loader(dataset, *, num_workers: int = 0):
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_one,
    )


@torch.no_grad()
def estimate_target_transform(
    train_ds,
    device: torch.device,
    mode: str = "none",
    max_events: int = -1,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mode = str(mode).lower()
    eps = float(eps)

    if mode == "none":
        center = torch.zeros(2, dtype=torch.float32, device=device)
        scale = torch.ones(2, dtype=torch.float32, device=device)
        return center, scale

    n_total = len(train_ds)
    rank = ddp_rank()
    world = ddp_world_size()

    if world > 1 and mode in ("standard", "minmax"):
        local_indices = list(range(rank, n_total, world))
        if max_events > 0:
            local_cap = max(1, int(np.ceil(max_events / world)))
            local_indices = local_indices[:local_cap]
        local_ds = torch.utils.data.Subset(train_ds, local_indices)
    else:
        local_ds = train_ds

    def get_y2(batch):
        y = batch["y_track"].view(3)
        return y[:2]  # only ptq and eta are scaled here; phi handled separately

    if mode == "standard":
        loader = _make_stats_loader(local_ds, num_workers=0)
        count = torch.zeros(1, dtype=torch.float64, device=device)
        sum_y = torch.zeros(2, dtype=torch.float64, device=device)
        sumsq_y = torch.zeros(2, dtype=torch.float64, device=device)

        for i, batch in enumerate(loader):
            if world == 1 and max_events > 0 and i >= max_events:
                break
            y = get_y2(batch).to(device=device, dtype=torch.float64).view(1, 2)
            count += 1.0
            sum_y += y.sum(dim=0)
            sumsq_y += (y * y).sum(dim=0)

        if ddp_is_initialized():
            ddp_all_reduce_sum(count)
            ddp_all_reduce_sum(sum_y)
            ddp_all_reduce_sum(sumsq_y)

        count = count.clamp(min=1.0)
        mean = sum_y / count
        var = (sumsq_y / count) - (mean * mean)
        var = torch.clamp(var, min=0.0)
        center = mean.to(torch.float32)
        scale = torch.sqrt(var).clamp(min=eps).to(torch.float32)
        return center, scale

    if mode == "minmax":
        loader = _make_stats_loader(local_ds, num_workers=0)
        local_min = torch.full((2,), float("inf"), dtype=torch.float32, device=device)
        local_max = torch.full((2,), float("-inf"), dtype=torch.float32, device=device)
        seen_any = False

        for i, batch in enumerate(loader):
            if world == 1 and max_events > 0 and i >= max_events:
                break
            y = get_y2(batch).to(device=device, dtype=torch.float32).view(2)
            local_min = torch.minimum(local_min, y)
            local_max = torch.maximum(local_max, y)
            seen_any = True

        if not seen_any:
            local_min = torch.zeros(2, dtype=torch.float32, device=device)
            local_max = torch.ones(2, dtype=torch.float32, device=device)

        if ddp_is_initialized():
            dist.all_reduce(local_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)

        center = local_min
        scale = (local_max - local_min).clamp(min=eps)
        return center, scale

    if mode == "robust":
        loader = _make_stats_loader(train_ds, num_workers=0)
        ys = []
        for i, batch in enumerate(loader):
            if max_events > 0 and i >= max_events:
                break
            ys.append(get_y2(batch).float().view(1, 2))

        if len(ys) == 0:
            center = torch.zeros(2, dtype=torch.float32, device=device)
            scale = torch.ones(2, dtype=torch.float32, device=device)
        else:
            y = torch.cat(ys, dim=0).to(device)
            q25 = torch.quantile(y, 0.25, dim=0)
            q50 = torch.quantile(y, 0.50, dim=0)
            q75 = torch.quantile(y, 0.75, dim=0)
            center = q50
            scale = (q75 - q25).clamp(min=eps)
        return center, scale

    raise ValueError(f"Unknown target scaling mode: {mode}")


def build_scheduler(opt, args, steps_per_epoch: int):
    if args.lr_schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
            min_lr=args.lr_plateau_min_lr,
        )

    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    total_steps = max(1, int(args.epochs * steps_per_epoch))

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


def reset_optimizer_lr(opt, lr: float):
    for pg in opt.param_groups:
        pg["lr"] = lr


def load_best_checkpoint_into_model(model, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state"], strict=True)
    raw_model.to(device)


def _try_resume_from_checkpoint(
    *,
    ckpt_path: str,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: Optional[EMA],
    device: torch.device,
) -> Tuple[int, Optional[float], int, Optional[int], bool]:
    p = Path(ckpt_path)
    if not p.exists():
        return 1, None, 0, None, False

    ckpt = torch.load(str(p), map_location="cpu")
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state"], strict=True)
    raw_model.to(device)

    if "optimizer_state" in ckpt:
        opt.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt and scheduler is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            pass
    if "scaler_state" in ckpt and scaler is not None and scaler.is_enabled():
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except Exception:
            pass
    if ema is not None and "ema_shadow" in ckpt and isinstance(ckpt["ema_shadow"], dict):
        raw_state = raw_model.state_dict()
        fixed_shadow = {}
        for k, v in ckpt["ema_shadow"].items():
            if not torch.is_tensor(v):
                continue
            if k in raw_state:
                ref = raw_state[k]
                fixed_shadow[k] = v.detach().to(device=ref.device, dtype=ref.dtype).clone()
            else:
                fixed_shadow[k] = v.detach().to(device=device).clone()
        ema.shadow = fixed_shadow

    last_epoch = int(ckpt.get("epoch", 0))
    best_monitor = ckpt.get("best_monitor", None)
    bad_epochs = int(ckpt.get("bad_epochs", 0))
    best_ckpt_epoch = ckpt.get("best_ckpt_epoch", None)
    start_epoch = max(1, last_epoch + 1)
    return start_epoch, best_monitor, bad_epochs, best_ckpt_epoch, True


# ----------------------------
# Losses / metrics
# ----------------------------

def wrapped_angle_diff(pred_phi: torch.Tensor, target_phi: torch.Tensor, period: float = 2.0 * math.pi) -> torch.Tensor:
    half_period = 0.5 * float(period)
    return torch.remainder(pred_phi - target_phi + half_period, float(period)) - half_period


def angle_from_sincos(sin_cos: torch.Tensor) -> torch.Tensor:
    if sin_cos.ndim == 1:
        s = sin_cos[0]
        c = sin_cos[1]
    else:
        s = sin_cos[..., 0]
        c = sin_cos[..., 1]
    return torch.atan2(s, c)


def component_regression_loss(diff: torch.Tensor, loss_type: str = "smoothl1", beta: float = 1.0) -> torch.Tensor:
    if loss_type == "mse":
        return diff.pow(2)
    if loss_type == "l1":
        return diff.abs()
    if loss_type == "smoothl1":
        return F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=beta, reduction="none")
    raise ValueError(f"Unknown loss_type={loss_type}")


def parse_three_floats(text: str) -> Tuple[float, float, float]:
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != 3:
        raise ValueError("Expected three comma-separated floats.")
    return vals[0], vals[1], vals[2]


@torch.no_grad()
def compute_metric_components(pred_metric: torch.Tensor, target_metric: torch.Tensor, phi_period: float) -> Dict[str, torch.Tensor]:
    diff = pred_metric - target_metric
    diff = diff.clone()
    diff[2] = wrapped_angle_diff(pred_metric[2], target_metric[2], period=phi_period)

    abs_err = diff.abs()
    sq_err = diff.pow(2)

    pt_target = target_metric[0].abs().clamp(min=1e-6)
    pt_mape = 100.0 * abs_err[0] / pt_target

    smape = 200.0 * abs_err / (pred_metric.abs() + target_metric.abs()).clamp(min=1e-6)

    return {
        "abs_err": abs_err.to(torch.float64),
        "sq_err": sq_err.to(torch.float64),
        "smape": smape.to(torch.float64),
        "pt_mape": pt_mape.to(torch.float64),
    }


def build_train_targets(
    y: torch.Tensor,
    target_center_2: torch.Tensor,
    target_scale_2: torch.Tensor,
    phi_mode: str,
) -> Dict[str, torch.Tensor]:
    out = {
        "pt": (y[0] - target_center_2[0]) / target_scale_2[0],
        "eta": (y[1] - target_center_2[1]) / target_scale_2[1],
    }
    phi = y[2]
    if phi_mode == "sincos":
        out["phi_target"] = torch.stack([torch.sin(phi), torch.cos(phi)], dim=0)
    elif phi_mode == "scalar":
        out["phi_target"] = phi
    else:
        raise ValueError(f"Unsupported phi_mode={phi_mode}")
    return out


def decode_prediction_to_metric(
    pred_dict: Dict[str, torch.Tensor],
    target_center_2: torch.Tensor,
    target_scale_2: torch.Tensor,
    phi_mode: str,
) -> torch.Tensor:
    pt = pred_dict["pt"] * target_scale_2[0] + target_center_2[0]
    eta = pred_dict["eta"] * target_scale_2[1] + target_center_2[1]
    if phi_mode == "sincos":
        phi = angle_from_sincos(pred_dict["phi_raw"])
    else:
        phi = pred_dict["phi_raw"].reshape(-1)[0]
    return torch.stack([pt, eta, phi], dim=0)


def compute_total_loss(
    pred_dict: Dict[str, torch.Tensor],
    y: torch.Tensor,
    target_center_2: torch.Tensor,
    target_scale_2: torch.Tensor,
    *,
    loss_type: str,
    phi_mode: str,
    phi_vec_weight: float,
    target_weights: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    t = build_train_targets(y, target_center_2, target_scale_2, phi_mode)

    pt_loss = component_regression_loss(pred_dict["pt"] - t["pt"], loss_type=loss_type).mean()
    eta_loss = component_regression_loss(pred_dict["eta"] - t["eta"], loss_type=loss_type).mean()

    if phi_mode == "sincos":
        phi_pred = F.normalize(pred_dict["phi_raw"], dim=0)
        phi_loss_vec = component_regression_loss(phi_pred - t["phi_target"], loss_type=loss_type)
        phi_loss = phi_vec_weight * phi_loss_vec.mean()
    else:
        diff_phi = wrapped_angle_diff(pred_dict["phi_raw"].reshape(-1)[0], t["phi_target"], period=2.0 * math.pi)
        phi_loss = component_regression_loss(diff_phi, loss_type=loss_type).mean()

    total = target_weights[0] * pt_loss + target_weights[1] * eta_loss + target_weights[2] * phi_loss
    pieces = {
        "loss_pt": pt_loss.detach(),
        "loss_eta": eta_loss.detach(),
        "loss_phi": phi_loss.detach(),
    }
    return total, pieces


def infer_dims_from_subset(subset) -> Tuple[int, int, int]:
    sample = subset[0]
    xdim = int(sample["x"].shape[1])
    edim = int(sample["edge_attr"].shape[1])
    ydim = int(sample["y_track"].shape[0])
    return xdim, edim, ydim


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def quarantine_bad_batch(
    batch: Dict[str, Any],
    exc: Exception,
    *,
    where: str,
    quarantine_file: str,
    skip_bad_graphs: bool,
) -> bool:
    bid = _batch_id_string(batch)
    payload = {
        "graph_id": bid,
        "where": where,
        "rank": ddp_rank(),
        "pid": os.getpid(),
        "error": _jsonable_error(exc),
    }
    try:
        _append_line_atomic(quarantine_file, json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass

    print(f"[rank{ddp_rank()}] quarantined bad graph: {bid} @ {where}", flush=True)
    return bool(skip_bad_graphs)


def _format_rank_prefix() -> str:
    return f"[rank{ddp_rank()}]"


def fail_fast_bad_batch(batch: Dict[str, Any], exc: Exception, where: str) -> None:
    msg = (
        f"{_format_rank_prefix()} fatal batch error during {where}: {_batch_id_string(batch)}\n"
        f"{type(exc).__name__}: {exc}"
    )
    print(msg, flush=True)
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    raise RuntimeError(msg) from exc


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", required=True)
    ap.add_argument("--split-file", required=True)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-train-graphs", type=int, default=-1)

    ap.add_argument("--layer-type", default="mpnn", choices=["mpnn", "edge_residual", "sage_residual", "gat_residual"])
    ap.add_argument("--gat-heads", type=int, default=4)
    ap.add_argument("--sage-aggr", default="mean", choices=["mean", "sum", "max"])
    ap.add_argument("--edgeconv-aggr", default="mean", choices=["mean", "sum", "max"])
    ap.add_argument("--graph-pool", default="meanmax", choices=["mean", "max", "meanmax"])

    ap.add_argument("--loss-type", default="smoothl1", choices=["smoothl1", "mse", "l1"])
    ap.add_argument("--target-scale", choices=["none", "standard", "robust", "minmax"], default="robust")
    ap.add_argument("--target-scale-eps", type=float, default=1e-6)
    ap.add_argument("--target-stats-max-events", type=int, default=-1)

    ap.add_argument("--phi-mode", default="sincos", choices=["sincos", "scalar"], help="Use sin/cos head for phi by default.")
    ap.add_argument("--phi-period", type=float, default=(2.0 * math.pi))
    ap.add_argument("--phi-vec-weight", type=float, default=1.0, help="Extra multiplier for the sin/cos phi loss.")
    ap.add_argument("--target-loss-weights", default="1.0,1.0,1.0", help="Weights for ptq, eta, phi losses.")

    ap.add_argument("--save", default="track_graph_regressor_v2.pt")
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--time", action="store_true", default=True)
    ap.add_argument("--no-time", dest="time", action="store_false")

    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pin-memory", action="store_true", default=True)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=False)
    ap.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    ap.add_argument("--worker-start-method", type=str, default="spawn", choices=["fork", "forkserver", "spawn"])

    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="track_graph_regressor")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-dir", default=None)
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-key", default=None)

    ap.add_argument("--early-stop", dest="early_stop", action="store_true", default=True)
    ap.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    ap.add_argument("--early-stop-patience", type=int, default=30)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0)
    ap.add_argument("--early-stop-monitor", choices=["val_loss", "val_rmse_mean"], default="val_loss")

    ap.add_argument("--lr-schedule", choices=["plateau", "cosine"], default="plateau")
    ap.add_argument("--lr-plateau-factor", type=float, default=0.5)
    ap.add_argument("--lr-plateau-patience", type=int, default=5)
    ap.add_argument("--lr-plateau-min-lr", type=float, default=0.0)
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)

    ap.add_argument("--reload-best-half-patience", dest="reload_best_half_patience", action="store_true", default=False)

    ap.add_argument("--fourier", dest="fourier", action="store_true", default=True)
    ap.add_argument("--no-fourier", dest="fourier", action="store_false")
    ap.add_argument("--fourier-base", type=float, default=3.0)
    ap.add_argument("--fourier-min-exp", type=int, default=-6)
    ap.add_argument("--fourier-max-exp", type=int, default=6)

    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--no-decay-norm-bias", action="store_true", default=True)
    ap.add_argument("--decay-norm-bias", dest="no_decay_norm_bias", action="store_false")

    ap.add_argument("--edge-dropout", type=float, default=0.0)
    ap.add_argument("--feat-noise-std", type=float, default=0.0)

    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="ema", action="store_false")
    ap.add_argument("--ema-decay", type=float, default=0.999)

    ap.add_argument("--run-id", default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--resume", dest="resume", action="store_true", default=False)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--skip-bad-graphs", action="store_true", default=True)
    ap.add_argument("--no-skip-bad-graphs", dest="skip_bad_graphs", action="store_false")
    ap.add_argument("--bad-graphs-file", default="bad_graphs.jsonl")
    ap.add_argument("--code-version", default=None)

    args = ap.parse_args()

    try:
        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGBUS, all_threads=True, chain=True)
    except Exception:
        pass

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.set_num_interop_threads(int(os.environ.get("TORCH_INTEROP_THREADS", "1")))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    try:
        mp.set_start_method(args.worker_start_method, force=True)
    except RuntimeError:
        pass
    ctx = mp.get_context(args.worker_start_method)

    ddp_setup()

    if ddp_is_main():
        print(
            f"[final-dl] num_workers={args.num_workers} pin_memory={args.pin_memory} "
            f"persistent_workers={args.persistent_workers} prefetch_factor={args.prefetch_factor}",
            flush=True,
        )

    seed_all(args.seed + 1000 * ddp_rank())

    run_id = args.run_id
    if ddp_is_main() and run_id is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
    if ddp_is_initialized():
        obj = [run_id]
        dist.broadcast_object_list(obj, src=0)
        run_id = obj[0]

    paths = sorted(glob.glob(args.data_glob))
    if not paths:
        raise SystemExit(f"No H5 files matched: {args.data_glob}")

    index_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with timed_section("dataset_index", device=index_device, enabled=args.time) as tt:
        ds = H5TrackGraphDataset(paths)
    if ddp_is_main() and args.time:
        print(f"[time] dataset indexing: {tt['seconds']:.3f}s (rank0)")

    split = np.load(args.split_file, allow_pickle=True)
    train_idx = split["train_idx"].astype(np.int64)
    val_idx = split["val_idx"].astype(np.int64)

    if "h5_paths" in split.files:
        saved_paths = [str(p) for p in split["h5_paths"].tolist()]
        cur_paths = [str(Path(p).resolve()) for p in paths]
        if saved_paths != cur_paths:
            raise RuntimeError("Split file appears to be for different H5 files or ordering. Regenerate split.")

    n = len(ds)
    if train_idx.size == 0 or val_idx.size == 0:
        raise RuntimeError(f"Split file has empty train/val: train={train_idx.size} val={val_idx.size}")
    if train_idx.min() < 0 or train_idx.max() >= n or val_idx.min() < 0 or val_idx.max() >= n:
        raise RuntimeError("Split indices out of range for current dataset.")

    if args.max_train_graphs > 0:
        train_idx = train_idx[:args.max_train_graphs]

    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(ds, val_idx.tolist())

    train_sampler = DistributedSampler(
        train_ds, num_replicas=ddp_world_size(), rank=ddp_rank(), shuffle=True, drop_last=True
    ) if ddp_is_initialized() else None
    val_sampler = DistributedSampler(
        val_ds, num_replicas=ddp_world_size(), rank=ddp_rank(), shuffle=False, drop_last=False
    ) if ddp_is_initialized() else None

    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if torch.cuda.is_available() and "LOCAL_RANK" in os.environ
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if ddp_is_main():
        print(f"[i] ddp world_size={ddp_world_size()} device={device}")
        print(f"[i] graphs: train={len(train_ds)} val={len(val_ds)} total={len(ds)} split={args.split_file}")

    xdim, edim, ydim = infer_dims_from_subset(train_ds)
    ds._close_files()
    gc.collect()
    if ydim != 3:
        raise RuntimeError(f"Expected y_track dim=3, got {ydim}")
    if ddp_is_main():
        print(f"[i] xdim={xdim} edim={edim} ydim={ydim}")

    model = TrackGraphRegressorGNN(
        xdim=xdim,
        edim=edim,
        hdim=args.hidden_dim,
        n_layers=args.layers,
        dropout=args.dropout,
        layer_type=args.layer_type,
        gat_heads=args.gat_heads,
        sage_aggr=args.sage_aggr,
        edgeconv_aggr=args.edgeconv_aggr,
        use_fourier=args.fourier,
        fourier_base=args.fourier_base,
        fourier_min_exp=args.fourier_min_exp,
        fourier_max_exp=args.fourier_max_exp,
        graph_pool=args.graph_pool,
        phi_mode=args.phi_mode,
    ).to(device)

    if ddp_is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    raw_model = model.module if hasattr(model, "module") else model
    if args.no_decay_norm_bias:
        decay, no_decay = [], []
        for n, p in raw_model.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith(".bias") or ("norm" in n.lower()) or ("layernorm" in n.lower()):
                no_decay.append(p)
            else:
                decay.append(p)
        param_groups = [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    else:
        param_groups = [{"params": raw_model.parameters(), "weight_decay": args.weight_decay}]

    try:
        opt = torch.optim.AdamW(param_groups, lr=args.lr, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(param_groups, lr=args.lr)

    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    target_center_2, target_scale_2 = estimate_target_transform(
        train_ds,
        device=device,
        mode=args.target_scale,
        max_events=args.target_stats_max_events,
        eps=args.target_scale_eps,
    )
    ds._close_files()

    target_weights = torch.tensor(parse_three_floats(args.target_loss_weights), device=device, dtype=torch.float32)

    if ddp_is_main():
        print(f"[i] target_scale_mode={args.target_scale}", flush=True)
        print(f"[i] target_center_linear={target_center_2.detach().cpu().numpy()}  # [ptq, eta]", flush=True)
        print(f"[i] target_scale_linear ={target_scale_2.detach().cpu().numpy()}  # [ptq, eta]", flush=True)
        print(f"[i] phi_mode={args.phi_mode} phi_period={args.phi_period}", flush=True)
        print(f"[i] target_loss_weights={target_weights.detach().cpu().tolist()}", flush=True)

    loader_kwargs = {
        "collate_fn": collate_one,
        "num_workers": int(args.num_workers),
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            multiprocessing_context=ctx,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        **loader_kwargs,
    )

    if ddp_is_initialized():
        if len(train_loader) == 0:
            raise RuntimeError(
                f"DDP training has 0 batches per rank (train_ds={len(train_ds)} world_size={ddp_world_size()})."
            )
        if len(val_loader) == 0:
            raise RuntimeError(
                f"DDP validation has 0 batches per rank (val_ds={len(val_ds)} world_size={ddp_world_size()})."
            )

    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(opt, args, steps_per_epoch=steps_per_epoch)

    ema = EMA.create(model, decay=args.ema_decay) if args.ema else None

    wandb_run = None
    wandb_enabled = bool(args.wandb and ddp_is_main() and args.wandb_mode != "disabled")
    if wandb_enabled:
        if args.wandb_key is not None:
            os.environ["WANDB_API_KEY"] = args.wandb_key
        os.environ["WANDB_MODE"] = args.wandb_mode
        cfg = {
            **vars(args),
            "target_center_linear": target_center_2.detach().cpu().tolist(),
            "target_scale_linear": target_scale_2.detach().cpu().tolist(),
            "target_labels": ["pt_like", "eta", "phi"],
            "xdim": xdim,
            "edim": edim,
        }
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                dir=args.wandb_dir,
                config=cfg,
            )
        except Exception as e:
            print(f"[wandb] init failed ({type(e).__name__}: {e}). Falling back to offline.")
            os.environ["WANDB_MODE"] = "offline"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                dir=args.wandb_dir,
                config=cfg,
            )

    save_path = _build_save_path(args, run_id)
    if ddp_is_main():
        print(f"[i] checkpoint path: {save_path}")

    best_monitor: Optional[float] = None
    bad_epochs: int = 0
    best_ckpt_epoch: Optional[int] = None
    start_epoch: int = 1

    if ddp_is_main() and args.resume:
        start_epoch, best_monitor, bad_epochs, best_ckpt_epoch, resumed = _try_resume_from_checkpoint(
            ckpt_path=save_path,
            model=model,
            opt=opt,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            device=device,
        )
        if resumed:
            print(
                f"[resume] Resumed from {save_path}: start_epoch={start_epoch} "
                f"best_monitor={best_monitor} bad_epochs={bad_epochs} best_ckpt_epoch={best_ckpt_epoch}",
                flush=True,
            )
        else:
            print(f"[resume] No checkpoint found at {save_path}; starting fresh.", flush=True)

    if ddp_is_initialized():
        payload = [int(start_epoch), best_monitor, int(bad_epochs), (int(best_ckpt_epoch) if best_ckpt_epoch is not None else -1)]
        dist.broadcast_object_list(payload, src=0)
        start_epoch = int(payload[0])
        best_monitor = payload[1]
        bad_epochs = int(payload[2])
        best_ckpt_epoch = (int(payload[3]) if int(payload[3]) >= 0 else None)

    half_pat = max(1, args.early_stop_patience // 2) if (args.early_stop and args.reload_best_half_patience) else 0
    reloaded_this_plateau = False

    for epoch in range(int(start_epoch), args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = torch.tensor(0.0, device=device)
        train_steps = torch.tensor(0.0, device=device)
        train_loss_parts = torch.zeros(3, device=device, dtype=torch.float64)
        train_abs_err = torch.zeros(3, device=device, dtype=torch.float64)
        train_sq_err = torch.zeros(3, device=device, dtype=torch.float64)
        train_smape_sum = torch.zeros(3, device=device, dtype=torch.float64)
        train_pt_mape_sum = torch.tensor(0.0, device=device, dtype=torch.float64)

        with timed_section("train_epoch_total", device, enabled=args.time):
            for batch in train_loader:
                try:
                    validate_graph_batch(batch, where="train/cpu")
                    batch = move_batch_to_device(batch, device)
                    x = batch["x"]
                    edge_index = batch["edge_index"]
                    edge_attr = batch["edge_attr"]
                    y = batch["y_track"].float()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                except Exception as e:
                    if quarantine_bad_batch(batch, e, where="train/load_or_h2d", quarantine_file=args.bad_graphs_file, skip_bad_graphs=args.skip_bad_graphs):
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    fail_fast_bad_batch(batch, e, where="train/load_or_h2d")

                if args.feat_noise_std > 0.0 and model.training:
                    x = x + args.feat_noise_std * torch.randn_like(x)

                batch_id = _batch_id_string(batch)
                if not torch.isfinite(x).all():
                    e = RuntimeError(f"Non-finite x after augmentation for {batch_id}")
                    if quarantine_bad_batch(batch, e, where="train/post_aug", quarantine_file=args.bad_graphs_file, skip_bad_graphs=args.skip_bad_graphs):
                        continue
                    raise e

                opt.zero_grad(set_to_none=True)
                try:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        pred_dict = model(x, edge_index, edge_attr, edge_dropout_p=args.edge_dropout)
                        loss, pieces = compute_total_loss(
                            pred_dict,
                            y,
                            target_center_2,
                            target_scale_2,
                            loss_type=args.loss_type,
                            phi_mode=args.phi_mode,
                            phi_vec_weight=args.phi_vec_weight,
                            target_weights=target_weights,
                        )

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss for {batch_id}")
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                except Exception as e:
                    if quarantine_bad_batch(batch, e, where="train/forward", quarantine_file=args.bad_graphs_file, skip_bad_graphs=args.skip_bad_graphs):
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    fail_fast_bad_batch(batch, e, where="train/forward")

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    opt.step()

                if ema is not None:
                    ema.update(model)
                if args.lr_schedule == "cosine":
                    scheduler.step()

                train_loss += loss.detach()
                train_steps += 1.0
                train_loss_parts += torch.stack([pieces["loss_pt"], pieces["loss_eta"], pieces["loss_phi"]]).to(torch.float64)

                pred_metric = decode_prediction_to_metric(pred_dict, target_center_2, target_scale_2, args.phi_mode)
                metric_parts = compute_metric_components(pred_metric, y, args.phi_period)
                train_abs_err += metric_parts["abs_err"]
                train_sq_err += metric_parts["sq_err"]
                train_smape_sum += metric_parts["smape"]
                train_pt_mape_sum += metric_parts["pt_mape"]

                del x, edge_index, edge_attr, y, pred_dict, loss, pieces, pred_metric, metric_parts

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        ddp_all_reduce_sum(train_loss)
        ddp_all_reduce_sum(train_steps)
        ddp_all_reduce_sum(train_loss_parts)
        ddp_all_reduce_sum(train_abs_err)
        ddp_all_reduce_sum(train_sq_err)
        ddp_all_reduce_sum(train_smape_sum)
        ddp_all_reduce_sum(train_pt_mape_sum)

        train_loss_mean = (train_loss / torch.clamp(train_steps, min=1.0)).item()
        train_loss_parts_mean = (train_loss_parts / torch.clamp(train_steps, min=1.0)).cpu().numpy()
        train_mae = (train_abs_err / torch.clamp(train_steps, min=1.0)).cpu().numpy()
        train_rmse = torch.sqrt(train_sq_err / torch.clamp(train_steps, min=1.0)).cpu().numpy()
        train_smape = (train_smape_sum / torch.clamp(train_steps, min=1.0)).cpu().numpy()
        train_pt_mape = (train_pt_mape_sum / torch.clamp(train_steps, min=1.0)).item()
        train_rmse_mean = float(np.mean(train_rmse))

        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        val_steps = torch.tensor(0.0, device=device)
        val_loss_parts = torch.zeros(3, device=device, dtype=torch.float64)
        val_abs_err = torch.zeros(3, device=device, dtype=torch.float64)
        val_sq_err = torch.zeros(3, device=device, dtype=torch.float64)
        val_smape_sum = torch.zeros(3, device=device, dtype=torch.float64)
        val_pt_mape_sum = torch.tensor(0.0, device=device, dtype=torch.float64)

        eval_ctx = ema.apply_to(model) if ema is not None else nullcontext()
        with eval_ctx:
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        validate_graph_batch(batch, where="val/cpu")
                        batch = move_batch_to_device(batch, device)
                        x = batch["x"]
                        edge_index = batch["edge_index"]
                        edge_attr = batch["edge_attr"]
                        y = batch["y_track"].float()
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                    except Exception as e:
                        if quarantine_bad_batch(batch, e, where="val/load_or_h2d", quarantine_file=args.bad_graphs_file, skip_bad_graphs=args.skip_bad_graphs):
                            continue
                        fail_fast_bad_batch(batch, e, where="val/load_or_h2d")

                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        pred_dict = model(x, edge_index, edge_attr, edge_dropout_p=0.0)
                        loss, pieces = compute_total_loss(
                            pred_dict,
                            y,
                            target_center_2,
                            target_scale_2,
                            loss_type=args.loss_type,
                            phi_mode=args.phi_mode,
                            phi_vec_weight=args.phi_vec_weight,
                            target_weights=target_weights,
                        )

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite val loss for {_batch_id_string(batch)}")
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)

                    pred_metric = decode_prediction_to_metric(pred_dict, target_center_2, target_scale_2, args.phi_mode)
                    metric_parts = compute_metric_components(pred_metric, y, args.phi_period)

                    val_loss += loss.detach()
                    val_steps += 1.0
                    val_loss_parts += torch.stack([pieces["loss_pt"], pieces["loss_eta"], pieces["loss_phi"]]).to(torch.float64)
                    val_abs_err += metric_parts["abs_err"]
                    val_sq_err += metric_parts["sq_err"]
                    val_smape_sum += metric_parts["smape"]
                    val_pt_mape_sum += metric_parts["pt_mape"]

                    del x, edge_index, edge_attr, y, pred_dict, loss, pieces, pred_metric, metric_parts

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        ddp_all_reduce_sum(val_loss)
        ddp_all_reduce_sum(val_steps)
        ddp_all_reduce_sum(val_loss_parts)
        ddp_all_reduce_sum(val_abs_err)
        ddp_all_reduce_sum(val_sq_err)
        ddp_all_reduce_sum(val_smape_sum)
        ddp_all_reduce_sum(val_pt_mape_sum)

        val_loss_mean = (val_loss / torch.clamp(val_steps, min=1.0)).item()
        val_loss_parts_mean = (val_loss_parts / torch.clamp(val_steps, min=1.0)).cpu().numpy()
        val_mae = (val_abs_err / torch.clamp(val_steps, min=1.0)).cpu().numpy()
        val_rmse = torch.sqrt(val_sq_err / torch.clamp(val_steps, min=1.0)).cpu().numpy()
        val_smape = (val_smape_sum / torch.clamp(val_steps, min=1.0)).cpu().numpy()
        val_pt_mape = (val_pt_mape_sum / torch.clamp(val_steps, min=1.0)).item()
        val_rmse_mean = float(np.mean(val_rmse))

        if args.lr_schedule == "plateau":
            scheduler.step(val_loss_mean)
        current_lr = opt.param_groups[0]["lr"]

        monitor_val = val_loss_mean if args.early_stop_monitor == "val_loss" else val_rmse_mean
        improved = best_monitor is None or monitor_val < best_monitor - args.early_stop_min_delta

        if ddp_is_main():
            print(
                f"[epoch {epoch:03d}] "
                f"train loss={train_loss_mean:.5f} parts=({train_loss_parts_mean[0]:.5f},{train_loss_parts_mean[1]:.5f},{train_loss_parts_mean[2]:.5f}) "
                f"mae=({train_mae[0]:.4f},{train_mae[1]:.4f},{train_mae[2]:.4f}) "
                f"smape=({train_smape[0]:.2f}%,{train_smape[1]:.2f}%,{train_smape[2]:.2f}%) ptq_mape={train_pt_mape:.2f}% "
                f"rmse_mean={train_rmse_mean:.4f} | "
                f"val loss={val_loss_mean:.5f} parts=({val_loss_parts_mean[0]:.5f},{val_loss_parts_mean[1]:.5f},{val_loss_parts_mean[2]:.5f}) "
                f"mae=({val_mae[0]:.4f},{val_mae[1]:.4f},{val_mae[2]:.4f}) "
                f"smape=({val_smape[0]:.2f}%,{val_smape[1]:.2f}%,{val_smape[2]:.2f}%) ptq_mape={val_pt_mape:.2f}% "
                f"rmse_mean={val_rmse_mean:.4f} | "
                f"lr={current_lr:.3e} | "
                f"{args.early_stop_monitor}={monitor_val:.6f} {'(best)' if improved else ''}"
            )

        if wandb_run is not None and ddp_is_main():
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss_mean,
                    "train/loss_ptq": float(train_loss_parts_mean[0]),
                    "train/loss_eta": float(train_loss_parts_mean[1]),
                    "train/loss_phi": float(train_loss_parts_mean[2]),
                    "train/mae_ptq": float(train_mae[0]),
                    "train/mae_eta": float(train_mae[1]),
                    "train/mae_phi": float(train_mae[2]),
                    "train/smape_ptq": float(train_smape[0]),
                    "train/smape_eta": float(train_smape[1]),
                    "train/smape_phi": float(train_smape[2]),
                    "train/ptq_mape": float(train_pt_mape),
                    "train/rmse_ptq": float(train_rmse[0]),
                    "train/rmse_eta": float(train_rmse[1]),
                    "train/rmse_phi": float(train_rmse[2]),
                    "train/rmse_mean": train_rmse_mean,
                    "val/loss": val_loss_mean,
                    "val/loss_ptq": float(val_loss_parts_mean[0]),
                    "val/loss_eta": float(val_loss_parts_mean[1]),
                    "val/loss_phi": float(val_loss_parts_mean[2]),
                    "val/mae_ptq": float(val_mae[0]),
                    "val/mae_eta": float(val_mae[1]),
                    "val/mae_phi": float(val_mae[2]),
                    "val/smape_ptq": float(val_smape[0]),
                    "val/smape_eta": float(val_smape[1]),
                    "val/smape_phi": float(val_smape[2]),
                    "val/ptq_mape": float(val_pt_mape),
                    "val/rmse_ptq": float(val_rmse[0]),
                    "val/rmse_eta": float(val_rmse[1]),
                    "val/rmse_phi": float(val_rmse[2]),
                    "val/rmse_mean": val_rmse_mean,
                    "lr": current_lr,
                    args.early_stop_monitor: monitor_val,
                },
                step=epoch,
            )

        if ddp_is_main() and improved:
            best_monitor = monitor_val
            best_ckpt_epoch = epoch
            save_ctx = ema.apply_to(model) if ema is not None else nullcontext()
            _ensure_parent_dir(str(save_path))
            with save_ctx:
                raw_model = model.module if hasattr(model, "module") else model
                ckpt_obj = {
                    "model_state": raw_model.state_dict(),
                    "xdim": xdim,
                    "edim": edim,
                    "ydim": 3,
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "layer_type": args.layer_type,
                    "gat_heads": args.gat_heads,
                    "sage_aggr": args.sage_aggr,
                    "edgeconv_aggr": args.edgeconv_aggr,
                    "graph_pool": args.graph_pool,
                    "loss_type": args.loss_type,
                    "fourier": args.fourier,
                    "fourier_base": args.fourier_base,
                    "fourier_min_exp": args.fourier_min_exp,
                    "fourier_max_exp": args.fourier_max_exp,
                    "target_scale_mode": args.target_scale,
                    "target_scale_eps": args.target_scale_eps,
                    "target_center_linear": target_center_2.detach().cpu(),
                    "target_scale_linear": target_scale_2.detach().cpu(),
                    "phi_mode": args.phi_mode,
                    "phi_period": args.phi_period,
                    "target_loss_weights": target_weights.detach().cpu(),
                    "best_monitor": best_monitor,
                    "early_stop_monitor": args.early_stop_monitor,
                    "best_ckpt_epoch": best_ckpt_epoch,
                    "run_id": run_id,
                    "weight_decay": args.weight_decay,
                    "edge_dropout": args.edge_dropout,
                    "feat_noise_std": args.feat_noise_std,
                    "ema": bool(ema is not None),
                    "ema_decay": args.ema_decay,
                    "lr_schedule": args.lr_schedule,
                    "warmup_epochs": args.warmup_epochs,
                    "min_lr_ratio": args.min_lr_ratio,
                    "code_version": args.code_version,
                    "epoch": int(epoch),
                    "bad_epochs": int(bad_epochs),
                    "optimizer_state": opt.state_dict(),
                    "scheduler_state": (scheduler.state_dict() if scheduler is not None else None),
                    "scaler_state": (scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None),
                    "ema_shadow": (ema.shadow if ema is not None else None),
                }
                _torch_save_atomic_with_retries(ckpt_obj, save_path, retries=6, base_sleep_s=1.0)

            print(
                f"  [*] saved best checkpoint to {save_path} "
                f"({args.early_stop_monitor}={best_monitor:.6f}, epoch={epoch})"
            )
            reloaded_this_plateau = False

        stop_now = False
        reload_now = False

        if args.early_stop and ddp_is_main():
            if improved:
                bad_epochs = 0
                reloaded_this_plateau = False
            else:
                bad_epochs += 1

            if args.reload_best_half_patience and (half_pat > 0) and (bad_epochs >= half_pat) and (not reloaded_this_plateau):
                if os.path.exists(save_path):
                    reload_now = True
                    reloaded_this_plateau = True
                    print(
                        f"[reload-best] No improvement for {bad_epochs} epochs (half_pat={half_pat}). "
                        f"Reloading best checkpoint (epoch={best_ckpt_epoch}) and resetting lr -> {args.lr:.3e}."
                    )
                else:
                    print(f"[reload-best] Wanted to reload at half_pat={half_pat}, but {save_path} not found.")

            if bad_epochs >= args.early_stop_patience:
                stop_now = True
                print(f"[early-stop] Triggered at epoch {epoch} (monitor={args.early_stop_monitor}).")

        reload_t = torch.tensor([1 if reload_now else 0], device=device, dtype=torch.int32)
        if ddp_is_initialized():
            dist.broadcast(reload_t, src=0)
        reload_now_all = bool(reload_t.item())

        if reload_now_all:
            ddp_barrier()
            load_best_checkpoint_into_model(model, save_path, device)
            reset_optimizer_lr(opt, args.lr)
            scheduler = build_scheduler(opt, args, steps_per_epoch=len(train_loader))
            if ema is not None:
                ema = EMA.create(model, decay=args.ema_decay)
            if scaler.is_enabled():
                scaler = torch.amp.GradScaler("cuda", enabled=True)
            if args.early_stop:
                bad_epochs = 0
            ddp_barrier()

        stop_t = torch.tensor([1 if stop_now else 0], device=device, dtype=torch.int32)
        if ddp_is_initialized():
            dist.broadcast(stop_t, src=0)
        if bool(stop_t.item()):
            break

    if wandb_run is not None and ddp_is_main():
        wandb.finish(quiet=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        ddp_cleanup()
