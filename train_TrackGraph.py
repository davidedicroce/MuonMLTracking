#!/usr/bin/env python3
"""
DDP multi-GPU training for graph-level track regression GNN.

Targets:
  y_track = [pt_over_q, eta, phi]

Launch example:
torchrun --standalone --nproc_per_node=8 train_TrackGraph.py \
    --data-glob "./data/track_graphs_pu0_part*.h5" \
    --split-file "./data/split_track_graphs_pu0_seed12345.npz" \
    --epochs 200 --lr 2e-4 \
    --num-workers 4 --pin-memory \
    --wandb --wandb-project "TrackGraphRegressor" --wandb-name "trackreg01" \
    --early-stop --fourier \
    --weight-decay 0.02 \
    --edge-dropout 0.1 --feat-noise-std 0.01 \
    --ema --ema-decay 0.999 \
    --save "track_graph_regressor.pt"
"""

import argparse
import atexit
import faulthandler
import glob
import os
import random
import shutil
import signal
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("PYTHONUNBUFFERED", "1")

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
      /graphs/<id>/y_track    # [pt_over_q, eta, phi]
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
                f"Please update your converter to store y_track = [pt_over_q, eta, phi]."
            )
        y_track = torch.from_numpy(g["y_track"][...]).float()

        out = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y_track": y_track,
        }

        if "original_node_ids" in g:
            out["original_node_ids"] = torch.from_numpy(g["original_node_ids"][...]).long()

        return out


def collate_one(batch):
    assert len(batch) == 1
    return batch[0]


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

    def _ddp_touch_edge_mlp(self) -> torch.Tensor:
        z = None
        for p in self.edge_mlp.parameters():
            z = (p.sum() * 0.0) if z is None else (z + p.sum() * 0.0)
        return z if z is not None else torch.tensor(0.0)

    def forward(self, h, edge_index, edge_attr, edge_dropout_p: float = 0.0):
        E = edge_attr.size(0)

        if E == 0:
            agg = torch.zeros_like(h)
            h_upd = self.node_mlp(torch.cat([h, agg], dim=1))
            touch = self._ddp_touch_edge_mlp().to(h.device)
            return self.norm(h + h_upd) + touch

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
    ):
        super().__init__()
        self.fourier = None
        self.layer_type = layer_type
        self.gat_heads = gat_heads
        self.sage_aggr = sage_aggr
        self.edgeconv_aggr = edgeconv_aggr
        self.graph_pool = graph_pool

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
        self.graph_head = MLP(pooled_dim, 3, hidden_dim=hdim, n_layers=2, dropout=dropout)

    def _pool_graph(self, h: torch.Tensor) -> torch.Tensor:
        if h.size(0) == 0:
            return h.new_zeros((1, self.graph_head.net[0].in_features), dtype=h.dtype)

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
        out = self.graph_head(g).squeeze(0)   # (3,)
        return out


# ----------------------------
# Metrics / losses
# ----------------------------

def build_scheduler(opt, args, steps_per_epoch: int):
    if args.lr_schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
            min_lr=args.lr_plateau_min_lr,
            verbose=ddp_is_main(),
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
    ema: Optional["EMA"],
    device: torch.device,
) -> Tuple[int, Optional[float], int, Optional[int], bool]:
    p = Path(ckpt_path)
    if not p.exists():
        return 1, None, 0, None, False

    def _move_ema_shadow_to_model_device_dtype(
        ema_shadow: Dict[str, torch.Tensor],
        model_state: Dict[str, torch.Tensor],
        fallback_device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in ema_shadow.items():
            if not torch.is_tensor(v):
                continue
            if k in model_state:
                ref = model_state[k]
                out[k] = v.detach().to(device=ref.device, dtype=ref.dtype).clone()
            else:
                out[k] = v.detach().to(device=fallback_device).clone()
        return out

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
        ema.shadow = _move_ema_shadow_to_model_device_dtype(ckpt["ema_shadow"], raw_model.state_dict(), device)

    last_epoch = int(ckpt.get("epoch", 0))
    best_monitor = ckpt.get("best_monitor", None)
    bad_epochs = int(ckpt.get("bad_epochs", 0))
    best_ckpt_epoch = ckpt.get("best_ckpt_epoch", None)
    start_epoch = max(1, last_epoch + 1)
    return start_epoch, best_monitor, bad_epochs, best_ckpt_epoch, True


def regression_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = "smoothl1") -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "smoothl1":
        return F.smooth_l1_loss(pred, target, beta=1.0)
    raise ValueError(f"Unknown loss_type={loss_type}")


@torch.no_grad()
def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    abs_err = (pred - target).abs()
    sq_err = (pred - target) ** 2

    mae_ptq = float(abs_err[0].item())
    mae_eta = float(abs_err[1].item())
    mae_phi = float(abs_err[2].item())

    rmse_ptq = float(torch.sqrt(sq_err[0]).item())
    rmse_eta = float(torch.sqrt(sq_err[1]).item())
    rmse_phi = float(torch.sqrt(sq_err[2]).item())

    mae_mean = float(abs_err.mean().item())
    rmse_mean = float(torch.sqrt(sq_err.mean()).item())

    return {
        "mae_ptq": mae_ptq,
        "mae_eta": mae_eta,
        "mae_phi": mae_phi,
        "rmse_ptq": rmse_ptq,
        "rmse_eta": rmse_eta,
        "rmse_phi": rmse_phi,
        "mae_mean": mae_mean,
        "rmse_mean": rmse_mean,
    }


# ----------------------------
# Train
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

    ap.add_argument("--save", default="track_graph_regressor.pt")
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--time", action="store_true", default=True)
    ap.add_argument("--no-time", dest="time", action="store_false")

    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pin-memory", action="store_true", default=True)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=True)
    ap.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    ap.add_argument("--worker-start-method", type=str, default="fork", choices=["fork", "forkserver", "spawn"])

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
        val_ds, num_replicas=ddp_world_size(), rank=ddp_rank(), shuffle=False, drop_last=True
    ) if ddp_is_initialized() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_one,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        multiprocessing_context=ctx,
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory_device=f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_one,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        multiprocessing_context=ctx,
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory_device=f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "",
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

    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if torch.cuda.is_available() and "LOCAL_RANK" in os.environ
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if ddp_is_main():
        print(f"[i] ddp world_size={ddp_world_size()} device={device}")
        print(f"[i] graphs: train={len(train_ds)} val={len(val_ds)} total={len(ds)} split={args.split_file}")

    sample = next(iter(train_loader))
    xdim = sample["x"].shape[1]
    edim = sample["edge_attr"].shape[1]
    ydim = sample["y_track"].shape[0]
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

    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(opt, args, steps_per_epoch=steps_per_epoch)

    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    ema = EMA.create(model, decay=args.ema_decay) if args.ema else None

    wandb_run = None
    wandb_enabled = bool(args.wandb and ddp_is_main() and args.wandb_mode != "disabled")
    if wandb_enabled:
        if args.wandb_key is not None:
            os.environ["WANDB_API_KEY"] = args.wandb_key
        os.environ["WANDB_MODE"] = args.wandb_mode
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                dir=args.wandb_dir,
                config=vars(args),
            )
        except Exception as e:
            print(f"[wandb] init failed ({type(e).__name__}: {e}). Falling back to offline.")
            os.environ["WANDB_MODE"] = "offline"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                dir=args.wandb_dir,
                config=vars(args),
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

        # ---- train ----
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        train_steps = torch.tensor(0.0, device=device)

        train_abs_err = torch.zeros(3, device=device, dtype=torch.float64)
        train_sq_err = torch.zeros(3, device=device, dtype=torch.float64)

        with timed_section("train_epoch_total", device, enabled=args.time):
            for batch in train_loader:
                x = batch["x"].to(device, non_blocking=True)
                edge_index = batch["edge_index"].to(device, non_blocking=True)
                edge_attr = batch["edge_attr"].to(device, non_blocking=True)
                y = batch["y_track"].to(device, non_blocking=True).float()

                if args.feat_noise_std > 0.0 and model.training:
                    x = x + args.feat_noise_std * torch.randn_like(x)

                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    pred = model(x, edge_index, edge_attr, edge_dropout_p=args.edge_dropout)
                    loss = regression_loss(pred, y, loss_type=args.loss_type)

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

                abs_err = (pred.detach() - y).abs().to(torch.float64)
                sq_err = ((pred.detach() - y) ** 2).to(torch.float64)
                train_abs_err += abs_err
                train_sq_err += sq_err

        ddp_all_reduce_sum(train_loss)
        ddp_all_reduce_sum(train_steps)
        ddp_all_reduce_sum(train_abs_err)
        ddp_all_reduce_sum(train_sq_err)

        train_loss_mean = (train_loss / torch.clamp(train_steps, min=1.0)).item()
        train_mae = (train_abs_err / torch.clamp(train_steps, min=1.0)).cpu().numpy()
        train_rmse = torch.sqrt(train_sq_err / torch.clamp(train_steps, min=1.0)).cpu().numpy()
        train_mae_mean = float(np.mean(train_mae))
        train_rmse_mean = float(np.mean(train_rmse))

        # ---- val ----
        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        val_steps = torch.tensor(0.0, device=device)
        val_abs_err = torch.zeros(3, device=device, dtype=torch.float64)
        val_sq_err = torch.zeros(3, device=device, dtype=torch.float64)

        eval_ctx = ema.apply_to(model) if ema is not None else nullcontext()
        with eval_ctx:
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device, non_blocking=True)
                    edge_index = batch["edge_index"].to(device, non_blocking=True)
                    edge_attr = batch["edge_attr"].to(device, non_blocking=True)
                    y = batch["y_track"].to(device, non_blocking=True).float()

                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        pred = model(x, edge_index, edge_attr, edge_dropout_p=0.0)
                        loss = regression_loss(pred, y, loss_type=args.loss_type)

                    val_loss += loss
                    val_steps += 1.0

                    abs_err = (pred - y).abs().to(torch.float64)
                    sq_err = ((pred - y) ** 2).to(torch.float64)
                    val_abs_err += abs_err
                    val_sq_err += sq_err

        ddp_all_reduce_sum(val_loss)
        ddp_all_reduce_sum(val_steps)
        ddp_all_reduce_sum(val_abs_err)
        ddp_all_reduce_sum(val_sq_err)

        val_loss_mean = (val_loss / torch.clamp(val_steps, min=1.0)).item()
        val_mae = (val_abs_err / torch.clamp(val_steps, min=1.0)).cpu().numpy()
        val_rmse = torch.sqrt(val_sq_err / torch.clamp(val_steps, min=1.0)).cpu().numpy()
        val_mae_mean = float(np.mean(val_mae))
        val_rmse_mean = float(np.mean(val_rmse))

        if args.lr_schedule == "plateau":
            scheduler.step(val_loss_mean)
        current_lr = opt.param_groups[0]["lr"]

        monitor_val = val_loss_mean if args.early_stop_monitor == "val_loss" else val_rmse_mean
        improved = (
            best_monitor is None or
            monitor_val < best_monitor - args.early_stop_min_delta
        )

        if ddp_is_main():
            print(
                f"[epoch {epoch:03d}] "
                f"train loss={train_loss_mean:.5f} mae=({train_mae[0]:.4f},{train_mae[1]:.4f},{train_mae[2]:.4f}) "
                f"rmse_mean={train_rmse_mean:.4f} | "
                f"val loss={val_loss_mean:.5f} mae=({val_mae[0]:.4f},{val_mae[1]:.4f},{val_mae[2]:.4f}) "
                f"rmse_mean={val_rmse_mean:.4f} | "
                f"lr={current_lr:.3e} | "
                f"{args.early_stop_monitor}={monitor_val:.6f} {'(best)' if improved else ''}"
            )

        if wandb_run is not None and ddp_is_main():
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss_mean,
                    "train/mae_ptq": float(train_mae[0]),
                    "train/mae_eta": float(train_mae[1]),
                    "train/mae_phi": float(train_mae[2]),
                    "train/mae_mean": train_mae_mean,
                    "train/rmse_ptq": float(train_rmse[0]),
                    "train/rmse_eta": float(train_rmse[1]),
                    "train/rmse_phi": float(train_rmse[2]),
                    "train/rmse_mean": train_rmse_mean,
                    "val/loss": val_loss_mean,
                    "val/mae_ptq": float(val_mae[0]),
                    "val/mae_eta": float(val_mae[1]),
                    "val/mae_phi": float(val_mae[2]),
                    "val/mae_mean": val_mae_mean,
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
                scaler = torch.cuda.amp.GradScaler(enabled=True)
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