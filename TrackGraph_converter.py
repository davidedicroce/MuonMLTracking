#!/usr/bin/env python3

"""
Build per-track branch graphs for regression of (pt, eta, phi).

Pipeline
--------
1. Read ROOT segment trees event-by-event.
2. Build the same candidate graph used for edge classification:
   - node features: [pos(3), dir(3), bucket(4)]  -> 10 dims
   - candidate edges from sector window + delta-theta filter
   - edge features: [dpos(3), dist(1), cosang(1), same_chamber(1), same_sector(1)] -> 7 dims
3. Run the trained edge-classifier ONNX model on the candidate graph.
4. Threshold predicted edge scores and build undirected connected components ("branches").
5. For each branch:
   - determine the majority segmentTruthPart among truth-bearing nodes
   - determine muon charge from segmentTruthPDGId
   - set regression target = (segmentTruthPt / q, segmentTruthEta, segmentTruthPhi)
     from that majority truth particle, where q = -1 for PDGId=13 and q = +1 for PDGId=-13
6. If multiple branches in the same event have the same majority segmentTruthPart,
   keep only the branch with the largest number of nodes carrying that truth id
   and discard the others.
7. Write one H5 graph per surviving branch.

Example
-------
python -u TrackGraph_converter.py \
  --input-dir /eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/MuonBucketClassifier/data_pu0_gsegments \
  --pattern "MuonSegmentDump_*.root" \
  --onnx-path ../MuonSegmentClassifier/onnx_eval_mpnn/best_segment_classifier_mpnn.onnx \
  --output-dir ./data \
  --output-name track_graphs_pu0 \
  --edge-score-threshold 0.5
"""

import argparse
import os
import re
import glob
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import uproot
import h5py
import onnxruntime as ort


REQUIRED_SEGMENT_BRANCHES = [
    "segmentDirectionX",
    "segmentDirectionY",
    "segmentDirectionZ",
    "segmentPositionX",
    "segmentPositionY",
    "segmentPositionZ",
    "CommonEventHash",
    "segmentHasTruth",
    "segmentTruthPart",
    "segmentTruthPt",
    "segmentTruthEta",
    "segmentTruthPhi",
    "segmentTruthPDGId",
    "bucket_chamberIndex",
    "bucket_layers",
    "bucket_sector",
    "bucket_segments",
]


# ----------------------------
# ROOT reading
# ----------------------------

def _normalize_keys(keys):
    return {re.sub(r"\[.*\]/[A-Za-z]$", "", k): k for k in keys}


def _open_segment_tree_auto(root_file: str, preferred_tree_names=("MuonSegmentDump",)):
    if ":" in root_file and not root_file.strip().endswith(":"):
        obj = uproot.open(root_file)
        try:
            _ = obj.num_entries
        except Exception as e:
            raise ValueError(f"Object selected by '{root_file}' is not a TTree.") from e
        return obj

    f = uproot.open(root_file)
    trees = []
    for name, obj in f.items():
        short = name.split(";")[0]
        try:
            _ = obj.num_entries
            trees.append((short, obj))
        except Exception:
            continue

    if not trees:
        raise ValueError(f"No TTrees found in '{root_file}'.")

    def has_required(tobj):
        clean = _normalize_keys(list(tobj.keys()))
        return all(b in clean for b in REQUIRED_SEGMENT_BRANCHES)

    for pref in preferred_tree_names:
        for tname, tobj in trees:
            if tname == pref and has_required(tobj):
                return tobj

    for tname, tobj in trees:
        if has_required(tobj):
            return tobj

    raise ValueError("Could not find a TTree with required segment branches.")


def _read_segment_tree(root_file: str):
    tree = _open_segment_tree_auto(root_file, preferred_tree_names=("MuonSegmentDump",))
    clean = _normalize_keys(list(tree.keys()))
    missing = [k for k in REQUIRED_SEGMENT_BRANCHES if k not in clean]
    if missing:
        raise ValueError(f"Missing segment branches: {missing}")

    arrays = tree.arrays([clean[k] for k in REQUIRED_SEGMENT_BRANCHES], library="np")
    td = {k: arrays[clean[k]] for k in REQUIRED_SEGMENT_BRANCHES}

    evh_arr = td["CommonEventHash"]
    if isinstance(evh_arr, np.ndarray) and evh_arr.ndim == 2 and evh_arr.shape[1] == 1:
        event_hashes = evh_arr[:, 0].astype(np.int64)
    else:
        event_hashes = np.asarray([int(np.ravel(x)[0]) for x in evh_arr], dtype=np.int64)

    seen = set()
    unique_hashes = []
    for h in event_hashes:
        if h not in seen:
            seen.add(h)
            unique_hashes.append(h)
    unique_hashes = np.array(unique_hashes, dtype=np.int64)

    ev_to_idx = defaultdict(list)
    for i, h in enumerate(event_hashes):
        ev_to_idx[int(h)].append(i)

    return td, ev_to_idx, unique_hashes


def _first_scalar(x):
    a = np.ravel(x)
    if a.size == 0:
        return None
    return a[0]


def _build_event_nodes(td, idxs):
    """
    Returns:
      pos_m:     (N,3) float32
      dir_u:     (N,3) float32
      bucket:    (N,4) int64
      has_truth: (N,)  int64
      truth_id:  (N,)  int64
      truth_pt:  (N,)  float32
      truth_eta: (N,)  float32
      truth_phi: (N,)  float32
      truth_pdgid:(N,) int64
    """
    px, py, pz = [], [], []
    dx, dy, dz = [], [], []
    b_ch, b_lay, b_sec, b_seg = [], [], [], []
    has_t, tid = [], []
    tpt, teta, tphi, tpdgid = [], [], [], []

    for i in idxs:
        pxi = _first_scalar(td["segmentPositionX"][i])
        pyi = _first_scalar(td["segmentPositionY"][i])
        pzi = _first_scalar(td["segmentPositionZ"][i])
        dxi = _first_scalar(td["segmentDirectionX"][i])
        dyi = _first_scalar(td["segmentDirectionY"][i])
        dzi = _first_scalar(td["segmentDirectionZ"][i])
        if None in (pxi, pyi, pzi, dxi, dyi, dzi):
            continue

        bchi = _first_scalar(td["bucket_chamberIndex"][i])
        blayi = _first_scalar(td["bucket_layers"][i])
        bseci = _first_scalar(td["bucket_sector"][i])
        bsegi = _first_scalar(td["bucket_segments"][i])
        if None in (bchi, blayi, bseci, bsegi):
            bchi, blayi, bseci, bsegi = -1, -1, -1, -1

        hti = _first_scalar(td["segmentHasTruth"][i])
        ht = 0 if hti is None else int(hti)

        tidi = _first_scalar(td["segmentTruthPart"][i])
        pti = _first_scalar(td["segmentTruthPt"][i])
        etai = _first_scalar(td["segmentTruthEta"][i])
        phii = _first_scalar(td["segmentTruthPhi"][i])
        pdgidi = _first_scalar(td["segmentTruthPDGId"][i])

        if ht == 0 or tidi is None:
            t = -1
        else:
            t = int(tidi)

        px.append(float(pxi)); py.append(float(pyi)); pz.append(float(pzi))
        dx.append(float(dxi)); dy.append(float(dyi)); dz.append(float(dzi))
        b_ch.append(int(bchi)); b_lay.append(int(blayi)); b_sec.append(int(bseci)); b_seg.append(int(bsegi))
        has_t.append(ht); tid.append(t)

        tpt.append(np.nan if pti is None else float(pti))
        teta.append(np.nan if etai is None else float(etai))
        tphi.append(np.nan if phii is None else float(phii))
        tpdgid.append(0 if pdgidi is None else int(pdgidi))

    if len(px) == 0:
        return None

    pos_mm = np.stack([px, py, pz], axis=1).astype(np.float32)
    pos_m = pos_mm / 1000.0

    dir_vec = np.stack([dx, dy, dz], axis=1).astype(np.float32)
    n = np.linalg.norm(dir_vec, axis=1, keepdims=True)
    n[n == 0] = 1.0
    dir_u = dir_vec / n

    bucket = np.stack([b_ch, b_lay, b_sec, b_seg], axis=1).astype(np.int64)
    has_truth = np.asarray(has_t, dtype=np.int64)
    truth_id = np.asarray(tid, dtype=np.int64)
    truth_pt = np.asarray(tpt, dtype=np.float32)
    truth_eta = np.asarray(teta, dtype=np.float32)
    truth_phi = np.asarray(tphi, dtype=np.float32)
    truth_pdgid = np.asarray(tpdgid, dtype=np.int64)

    return pos_m, dir_u, bucket, has_truth, truth_id, truth_pt, truth_eta, truth_phi, truth_pdgid


# ----------------------------
# Candidate graph construction
# ----------------------------

def build_edges_sector_window(bucket_sector: np.ndarray, max_delta: int = 1, sector_mod: int = -1):
    sec = np.asarray(bucket_sector, dtype=np.int64)
    n = sec.shape[0]
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64)

    sec_to_nodes = defaultdict(list)
    for i, s in enumerate(sec):
        sec_to_nodes[int(s)].append(i)

    def sec_dist(a, b):
        d = abs(a - b)
        if sector_mod and sector_mod > 0:
            d = min(d, sector_mod - d)
        return d

    src_list = []
    dst_list = []

    for s, nodes_s in sec_to_nodes.items():
        for delta in range(-max_delta, max_delta + 1):
            t = s + delta
            if sector_mod and sector_mod > 0:
                t %= sector_mod

            if t not in sec_to_nodes:
                continue
            if sec_dist(s, t) > max_delta:
                continue

            nodes_t = sec_to_nodes[t]

            if s == t:
                for i in nodes_s:
                    for j in nodes_s:
                        if i != j:
                            src_list.append(i)
                            dst_list.append(j)
            else:
                for i in nodes_s:
                    for j in nodes_t:
                        src_list.append(i)
                        dst_list.append(j)

    if len(src_list) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    return np.stack(
        [np.asarray(src_list, dtype=np.int64), np.asarray(dst_list, dtype=np.int64)],
        axis=0,
    )


def filter_edges_by_delta_theta(edge_index: np.ndarray, dir_u: np.ndarray, max_delta_theta_deg: float):
    if edge_index.shape[1] == 0:
        return edge_index

    src = edge_index[0]
    dst = edge_index[1]

    cosang = np.sum(dir_u[src] * dir_u[dst], axis=1)
    cosang = np.clip(cosang, -1.0, 1.0)

    theta_deg = np.degrees(np.arccos(cosang))
    keep = np.abs(theta_deg) <= max_delta_theta_deg
    return edge_index[:, keep]


def edge_features(pos_m, dir_u, bucket, edge_index):
    if edge_index.shape[1] == 0:
        return np.zeros((0, 7), dtype=np.float32)

    src = edge_index[0]
    dst = edge_index[1]

    dpos = (pos_m[dst] - pos_m[src]).astype(np.float32)
    dist = np.linalg.norm(dpos, axis=1, keepdims=True).astype(np.float32)
    cosang = np.sum(dir_u[src] * dir_u[dst], axis=1, keepdims=True).astype(np.float32)

    same_ch = (bucket[src, 0] == bucket[dst, 0]).astype(np.float32).reshape(-1, 1)
    same_sec = (bucket[src, 2] == bucket[dst, 2]).astype(np.float32).reshape(-1, 1)

    return np.concatenate([dpos, dist, cosang, same_ch, same_sec], axis=1).astype(np.float32)


# ----------------------------
# Edge-classifier ONNX inference
# ----------------------------

def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def make_onnx_session(onnx_path: str):
    return ort.InferenceSession(onnx_path)


def run_edge_onnx(sess, x, edge_index, edge_attr, output_is_probability: bool = False):
    out = sess.run(
        None,
        {
            "x": x.astype(np.float32),
            "edge_index": edge_index.astype(np.int64),
            "edge_attr": edge_attr.astype(np.float32),
        },
    )[0].reshape(-1)

    if output_is_probability:
        return out.astype(np.float32)
    return sigmoid(out).astype(np.float32)


# ----------------------------
# Branch building from predicted edges
# ----------------------------

def build_graph_from_selected_edges(edge_index: np.ndarray, edge_scores: np.ndarray, threshold: float):
    adj = defaultdict(set)
    if edge_index.shape[1] == 0:
        return adj

    for (i, j), s in zip(edge_index.T, edge_scores):
        if s >= threshold:
            adj[int(i)].add(int(j))
            adj[int(j)].add(int(i))
    return adj


def connected_components(adj, n_nodes: int):
    visited = set()
    comps = []

    for i in range(n_nodes):
        if i in visited:
            continue
        stack = [i]
        comp = []
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.append(u)
            stack.extend(adj[u])
        comps.append(sorted(comp))

    return comps


def subgraph_from_component(node_ids, x, pos_m, dir_u, bucket, edge_index, edge_attr):
    """
    Build branch subgraph with remapped node indices.
    Keeps only edges whose src and dst are both in node_ids.
    """
    node_ids = np.asarray(sorted(node_ids), dtype=np.int64)
    old_to_new = {int(old): new for new, old in enumerate(node_ids)}

    x_sub = x[node_ids]
    pos_sub = pos_m[node_ids]
    dir_sub = dir_u[node_ids]
    bucket_sub = bucket[node_ids]

    if edge_index.shape[1] == 0:
        edge_index_sub = np.zeros((2, 0), dtype=np.int64)
        edge_attr_sub = np.zeros((0, edge_attr.shape[1]), dtype=np.float32)
        return node_ids, x_sub, pos_sub, dir_sub, bucket_sub, edge_index_sub, edge_attr_sub

    keep = np.isin(edge_index[0], node_ids) & np.isin(edge_index[1], node_ids)
    edge_index_kept = edge_index[:, keep]
    edge_attr_kept = edge_attr[keep]

    if edge_index_kept.shape[1] == 0:
        edge_index_sub = np.zeros((2, 0), dtype=np.int64)
        edge_attr_sub = np.zeros((0, edge_attr.shape[1]), dtype=np.float32)
        return node_ids, x_sub, pos_sub, dir_sub, bucket_sub, edge_index_sub, edge_attr_sub

    remapped_src = np.asarray([old_to_new[int(i)] for i in edge_index_kept[0]], dtype=np.int64)
    remapped_dst = np.asarray([old_to_new[int(i)] for i in edge_index_kept[1]], dtype=np.int64)
    edge_index_sub = np.stack([remapped_src, remapped_dst], axis=0)

    return node_ids, x_sub, pos_sub, dir_sub, bucket_sub, edge_index_sub, edge_attr_kept


def majority_truth_for_branch(node_ids, has_truth, truth_id, truth_pt, truth_eta, truth_phi, truth_pdgid):
    """
    Returns dict with:
      majority_truth_id
      majority_count
      branch_size
      target (pt_over_q, eta, phi)

    Majority is computed only over nodes with has_truth != 0 and truth_id >= 0.
    For target values, take the median over nodes belonging to the majority truth_id.
    Charge convention from PDGId:
      mu-      :  PDGId =  13 -> q = -1
      mu+      :  PDGId = -13 -> q = +1
    Therefore:
      pt_over_q = pt / q
    """
    node_ids = np.asarray(node_ids, dtype=np.int64)
    valid = (has_truth[node_ids] != 0) & (truth_id[node_ids] >= 0)
    valid_nodes = node_ids[valid]

    if valid_nodes.size == 0:
        return None

    tids = truth_id[valid_nodes]
    counts = Counter(int(t) for t in tids)
    majority_tid, majority_count = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))

    maj_nodes = valid_nodes[truth_id[valid_nodes] == majority_tid]
    if maj_nodes.size == 0:
        return None

    pt_vals = truth_pt[maj_nodes]
    eta_vals = truth_eta[maj_nodes]
    phi_vals = truth_phi[maj_nodes]
    pdgid_vals = truth_pdgid[maj_nodes]

    pt_vals = pt_vals[np.isfinite(pt_vals)]
    eta_vals = eta_vals[np.isfinite(eta_vals)]
    phi_vals = phi_vals[np.isfinite(phi_vals)]

    if pt_vals.size == 0 or eta_vals.size == 0 or phi_vals.size == 0 or pdgid_vals.size == 0:
        return None

    # Determine the majority PDGId among the majority-truth nodes.
    # Accept only the expected muon/anti-muon PDG ids.
    pdgid_counter = Counter(int(x) for x in pdgid_vals if int(x) in (13, -13))
    if len(pdgid_counter) == 0:
        return None

    majority_pdgid, _ = max(pdgid_counter.items(), key=lambda kv: (kv[1], -abs(kv[0]), -kv[0]))

    if majority_pdgid == 13:
        charge = -1.0
    elif majority_pdgid == -13:
        charge = +1.0
    else:
        return None
    
    pt_over_q_vals = pt_vals / charge

    target = np.asarray(
        [
            np.median(pt_over_q_vals).astype(np.float32),
            np.median(eta_vals).astype(np.float32),
            np.median(phi_vals).astype(np.float32),
        ],
        dtype=np.float32,
    )

    return {
        "majority_truth_id": int(majority_tid),
        "majority_count": int(majority_count),
        "branch_size": int(node_ids.size),
        "majority_pdgid": int(majority_pdgid),
        "charge": np.float32(charge),
        "target": target,
    }


def deduplicate_branches_by_majority_truth(branch_records):
    """
    If multiple branches have the same majority_truth_id, keep only the one with:
      1) larger majority_count
      2) then larger branch_size
      3) then lower original branch_idx
    """
    best_by_tid = {}

    for rec in branch_records:
        tid = rec["majority_truth_id"]
        if tid not in best_by_tid:
            best_by_tid[tid] = rec
            continue

        old = best_by_tid[tid]
        new_key = (rec["majority_count"], rec["branch_size"], -rec["branch_idx"])
        old_key = (old["majority_count"], old["branch_size"], -old["branch_idx"])
        if new_key > old_key:
            best_by_tid[tid] = rec

    kept = list(best_by_tid.values())
    kept.sort(key=lambda r: r["branch_idx"])
    return kept


# ----------------------------
# H5 writing
# ----------------------------

def _write_branch_group(
    g,
    *,
    event_hash: int,
    branch_idx: int,
    original_node_ids: np.ndarray,
    x: np.ndarray,
    pos_m: np.ndarray,
    dir_u: np.ndarray,
    bucket: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    y_track: np.ndarray,
    majority_truth_id: int,
    majority_count: int,
    branch_size: int,
    majority_pdgid: int,
    charge: float,
):
    g.attrs["event_hash"] = int(event_hash)
    g.attrs["branch_idx"] = int(branch_idx)
    g.attrs["majority_truth_id"] = int(majority_truth_id)
    g.attrs["majority_count"] = int(majority_count)
    g.attrs["branch_size"] = int(branch_size)
    g.attrs["majority_pdgid"] = int(majority_pdgid)
    g.attrs["charge"] = float(charge)

    g.create_dataset("original_node_ids", data=original_node_ids.astype(np.int64), compression="gzip", compression_opts=4)
    g.create_dataset("x", data=x.astype(np.float32), compression="gzip", compression_opts=4)
    g.create_dataset("pos_m", data=pos_m.astype(np.float32), compression="gzip", compression_opts=4)
    g.create_dataset("dir_u", data=dir_u.astype(np.float32), compression="gzip", compression_opts=4)
    g.create_dataset("bucket", data=bucket.astype(np.int64), compression="gzip", compression_opts=4)
    g.create_dataset("edge_index", data=edge_index.astype(np.int64), compression="gzip", compression_opts=4)
    g.create_dataset("edge_attr", data=edge_attr.astype(np.float32), compression="gzip", compression_opts=4)
    # y_track = [pt_over_q, eta, phi]
    g.create_dataset("y_track", data=y_track.astype(np.float32), compression="gzip", compression_opts=4)


def _open_new_part(output_dir: Path, output_name: str, part_idx: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_name}_part{part_idx:04d}.h5"
    h5 = h5py.File(out_path, "w")
    h5.attrs["n_graphs_written"] = 0
    h5.create_group("graphs")
    return h5, out_path


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory with ROOT files")
    ap.add_argument("--pattern", default="*.root", help="Glob pattern")
    ap.add_argument("--onnx-path", required=True, help="Edge-classifier ONNX path")
    ap.add_argument("--output-dir", required=True, help="Directory for output H5 files")
    ap.add_argument("--output-name", required=True, help="Base name for output H5 files")
    ap.add_argument("--max-events", type=int, default=-1, help="Global cap across all ROOT files (-1 = all)")
    ap.add_argument("--graphs-per-part", type=int, default=10000, help="Max branch-graphs per H5 part")

    ap.add_argument("--edge-score-threshold", type=float, default=0.5, help="Threshold on edge-classifier score")
    ap.add_argument("--onnx-output-probability", action="store_true",
                    help="Set if ONNX already outputs probabilities instead of logits")

    ap.add_argument("--max-delta-theta-deg", type=float, default=35.0,
                    help="Keep candidate edges only if delta-theta <= this")
    ap.add_argument("--max-delta-sector", type=int, default=1,
                    help="Keep candidate edges only if |sector_i - sector_j| <= this")
    ap.add_argument("--sector-mod", type=int, default=16,
                    help="Optional wrap-around for sectors; -1 disables")
    ap.add_argument("--min-branch-size", type=int, default=2,
                    help="Discard predicted branches smaller than this number of nodes")

    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No ROOT files matched: {os.path.join(args.input_dir, args.pattern)}")

    sess = make_onnx_session(args.onnx_path)

    output_dir = Path(args.output_dir)
    output_name = args.output_name
    part_idx = 1
    h5, out_path = _open_new_part(output_dir, output_name, part_idx)

    graphs_grp = h5["graphs"]
    print(f"[i] writing {out_path}")

    total_graphs_written = 0
    written_in_part = 0
    total_events_seen = 0
    skipped_events = 0
    skipped_branches = 0

    for root_path in files:
        print(f"[i] reading {root_path}")
        try:
            td, ev_to_idx, uniq = _read_segment_tree(root_path)
        except Exception as e:
            print(f"[!] skip file (failed to read tree): {root_path} :: {e}")
            continue

        for evh in uniq:
            if args.max_events > 0 and total_events_seen >= args.max_events:
                h5.attrs["skipped_events"] = skipped_events
                h5.attrs["skipped_branches"] = skipped_branches
                h5.close()
                print(f"[done] reached --max-events={args.max_events}; wrote {total_graphs_written} branch-graphs")
                return

            total_events_seen += 1

            idxs = np.asarray(ev_to_idx[int(evh)], dtype=np.int64)
            nodes = _build_event_nodes(td, idxs)
            if nodes is None:
                skipped_events += 1
                continue

            pos_m, dir_u, bucket, has_truth, truth_id, truth_pt, truth_eta, truth_phi, truth_pdgid = nodes
            n_nodes = pos_m.shape[0]
            if n_nodes < 2:
                skipped_events += 1
                continue

            x = np.concatenate([pos_m, dir_u, bucket.astype(np.float32)], axis=1).astype(np.float32)

            edge_index = build_edges_sector_window(
                bucket_sector=bucket[:, 2],
                max_delta=args.max_delta_sector,
                sector_mod=(args.sector_mod if args.sector_mod > 0 else -1),
            )
            edge_index = filter_edges_by_delta_theta(
                edge_index=edge_index,
                dir_u=dir_u,
                max_delta_theta_deg=args.max_delta_theta_deg,
            )

            if edge_index.shape[1] == 0:
                skipped_events += 1
                continue

            edge_attr = edge_features(pos_m, dir_u, bucket, edge_index)

            try:
                edge_scores = run_edge_onnx(
                    sess,
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    output_is_probability=args.onnx_output_probability,
                )
            except Exception as e:
                print(f"[!] ONNX failed for event {int(evh)} :: {e}")
                skipped_events += 1
                continue

            adj = build_graph_from_selected_edges(
                edge_index=edge_index,
                edge_scores=edge_scores,
                threshold=args.edge_score_threshold,
            )
            components = connected_components(adj, n_nodes)
            components = [c for c in components if len(c) >= args.min_branch_size]

            if len(components) == 0:
                skipped_events += 1
                continue

            branch_records = []
            selected_edge_mask = edge_scores >= args.edge_score_threshold
            selected_edge_index = edge_index[:, selected_edge_mask]
            selected_edge_attr = edge_attr[selected_edge_mask]

            for branch_idx, comp_nodes in enumerate(components):
                maj = majority_truth_for_branch(
                    node_ids=comp_nodes,
                    has_truth=has_truth,
                    truth_id=truth_id,
                    truth_pt=truth_pt,
                    truth_eta=truth_eta,
                    truth_phi=truth_phi,
                    truth_pdgid=truth_pdgid,
                )
                if maj is None:
                    skipped_branches += 1
                    continue

                (
                    original_node_ids,
                    x_sub,
                    pos_sub,
                    dir_sub,
                    bucket_sub,
                    edge_index_sub,
                    edge_attr_sub,
                ) = subgraph_from_component(
                    node_ids=comp_nodes,
                    x=x,
                    pos_m=pos_m,
                    dir_u=dir_u,
                    bucket=bucket,
                    edge_index=selected_edge_index,
                    edge_attr=selected_edge_attr,
                )

                if x_sub.shape[0] < args.min_branch_size:
                    skipped_branches += 1
                    continue

                branch_records.append(
                    {
                        "branch_idx": branch_idx,
                        "original_node_ids": original_node_ids,
                        "x": x_sub,
                        "pos_m": pos_sub,
                        "dir_u": dir_sub,
                        "bucket": bucket_sub,
                        "edge_index": edge_index_sub,
                        "edge_attr": edge_attr_sub,
                        "y_track": maj["target"],
                        "majority_truth_id": maj["majority_truth_id"],
                        "majority_count": maj["majority_count"],
                        "branch_size": maj["branch_size"],
                        "majority_pdgid": maj["majority_pdgid"],
                        "charge": maj["charge"],
                    }
                )

            if len(branch_records) == 0:
                skipped_events += 1
                continue

            branch_records = deduplicate_branches_by_majority_truth(branch_records)

            for rec in branch_records:
                if written_in_part >= args.graphs_per_part:
                    h5.attrs["skipped_events"] = skipped_events
                    h5.attrs["skipped_branches"] = skipped_branches
                    h5.close()
                    part_idx += 1
                    h5, out_path = _open_new_part(output_dir, output_name, part_idx)
                    graphs_grp = h5["graphs"]
                    print(f"[i] writing {out_path}")
                    written_in_part = 0

                g = graphs_grp.create_group(f"{total_graphs_written:07d}")
                _write_branch_group(
                    g,
                    event_hash=int(evh),
                    branch_idx=int(rec["branch_idx"]),
                    original_node_ids=rec["original_node_ids"],
                    x=rec["x"],
                    pos_m=rec["pos_m"],
                    dir_u=rec["dir_u"],
                    bucket=rec["bucket"],
                    edge_index=rec["edge_index"],
                    edge_attr=rec["edge_attr"],
                    y_track=rec["y_track"],
                    majority_truth_id=int(rec["majority_truth_id"]),
                    majority_count=int(rec["majority_count"]),
                    branch_size=int(rec["branch_size"]),
                    majority_pdgid=int(rec["majority_pdgid"]),
                    charge=float(rec["charge"]),
                )

                total_graphs_written += 1
                written_in_part += 1
                h5.attrs["n_graphs_written"] = int(h5.attrs["n_graphs_written"]) + 1

    h5.attrs["skipped_events"] = skipped_events
    h5.attrs["skipped_branches"] = skipped_branches
    h5.close()
    print(f"[done] wrote {total_graphs_written} branch-graphs across all files")


if __name__ == "__main__":
    main()