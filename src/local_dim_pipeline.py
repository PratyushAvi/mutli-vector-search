"""
Local effective dimension vs out-degree as a batch job. effective_dim.ipynb ("Local effective dimension vs
out-degree") reads what this writes and makes the tables and plots.

For each dataset folder (holding vectors.npy) and its graph, GRAPH_DIR/adj-list-<name>.txt and
GRAPH_DIR/<name>-computed.txt with <name> the folder name (line i of the adjacency list is the
out-neighbourhood of the point on line i of the computed file):
  1. sources = the points with a computed out-neighbourhood; out-degree = neighbours on their line
  2. one sequential pass over vectors.npy per group of sources: the exact K = max(k) nearest neighbours of
     each source (squared Euclidean, itself excluded), keeping their vectors rather than their ids
  3. local PR of the k nearest, for every k in the grid and both centrings ("mean", "v"; see localdim.py)

Sizes follow the memory available (printed at the start of each dataset):
  host RAM (the smallest of MemAvailable, the cgroup / SLURM limit) x --host-mem-fraction: two read
      buffers of `chunk` rows (the next block is read while the GPU works on the current one)
  free GPU memory x --gpu-mem-fraction: the group of sources whose running top K (vectors + distances)
      stays on the GPU during a pass (at most half of it), then the query batch and `chunk` that fit in the
      rest. More sources than one group holds means one pass over the file per group.

Writes OUT_DIR/<name>/ (resumes: groups with a part file are skipped):
  meta.json            source ("npy"), dataset, graph files, n, d, dtype, rows, ks, estimator, sources, groups, sizes
  sources.npy          source row ids (sorted); out_degree.npy: their out-degrees
  source_vectors.npy   (q, d) the sources' vectors, in the file's dtype
  nbr_vectors.npy      (q, K, d) the K nearest neighbours' vectors, nearest first, in the file's dtype
                       (with source_vectors.npy, not written with --no-save-vectors)
  nbr_sq_dists.npy     (q, K) float32 squared distances to them
  part-<g>.npz         local PR of group g; local_pr.csv: everything, one row per source (when all are done)

With --local-prune, also (the k-NN pass then keeps neighbour ids; groups done without them are run again):
  nbr_ids.npy          (q, K) int64 ids of the K nearest neighbours, nearest first
  local_prune_k<k>.txt robust prune from v over N_k(v) alone (alpha --prune-alpha, as utils.py's prune), one line
                       per source: "source [(neighbour id, points of N_k(v) still uncovered after it), ...]";
                       divide by k for the uncovered fraction
  local_pr.csv columns out_in_knn_k<k> (fraction of v's out-neighbourhood inside N_k(v)) and lp_degree_k<k>
                       (edges the local prune needs to cover N_k(v)); part-<g>-prune.npz per group

Usage:
  python src/local_dim_pipeline.py --graph-dir $SCRATCH/ANN-Search/graphs \\
      --dataset $SCRATCH/ANN-Search/datasets/bigann --dataset $SCRATCH/ANN-Search/datasets/spacev1b
  python src/local_dim_pipeline.py --graph-dir ... --dataset .../bigann --rows bigann=10000000 --max-sources 5000

local_dim_vibe.py does the same for VIBE datasets (robust-prune graphs from vibe_pipeline.py), through compute().
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

import localdim
from localdim import log
from paths import RESULTS_DIR

K_BASE = [16, 32, 64, 128, 256, 512, 1024, 2048]


def gib(b):
    return f"{b / 2**30:.1f} GiB"


def plan(nq, K, d, itemsize, rows, device, host_frac, gpu_frac, max_chunk, pr_itemsize, local_prune=False):
    """Group size, query batch, chunk rows and PR batch from the memory available now."""
    host = localdim.host_available_bytes() * host_frac
    on_gpu = str(device).startswith("cuda")
    dev = localdim.device_free_bytes(device) * gpu_frac if on_gpu else host * 0.75
    buf_budget = host * 0.25 if not on_gpu else host           # on CPU the "device" shares host RAM
    per_src = K * (4 + d * itemsize) + d * (4 + itemsize) + 8   # running top K + the query itself
    if local_prune:
        per_src += K * 8                                         # and the neighbours' ids
    group = int(min(nq, (dev * 0.5) // per_src))
    if group < 1:
        raise MemoryError(f"one source needs {gib(per_src)} of device memory for K={K}; have {gib(dev)}")
    work = dev - group * per_src
    per_merge = K * (2 * d * itemsize + 4 * 3 + 8 * 4)         # kept + gathered vectors, distances, indices
    qb = min(group, 4096)
    while True:
        merge_batch = int(max(1, min(qb, (work * 0.25) // per_merge)))
        per_row = d * (itemsize + 4) + 4 + qb * 4 * 2           # raw + float block, norms, distances + topk
        fixed = merge_batch * per_merge + qb * K * 12 * 2        # merge + the block's top K (values, indices)
        # half of the rest for the block and its distances; the other half is headroom for topk's workspace
        chunk = int(min((work - fixed) * 0.5 // per_row, buf_budget // (2 * d * itemsize), max_chunk, rows))
        if chunk >= 8192 or qb == 1:
            break
        qb //= 2
    if chunk < 1:
        raise MemoryError(f"no room for a block: device {gib(dev)}, host {gib(host)}")
    pr_per_src = 3 * K * d * pr_itemsize + 2 * min(K, d) ** 2 * pr_itemsize
    pr_batch = int(max(1, min(group, (dev - group * per_src) // pr_per_src)))
    prune_per_src = K * K * 4 * 2 + K * d * 4 + K * 64          # pairwise distances (+ a gathered slice), floats
    prune_batch = int(max(1, min(group, (dev - group * per_src) * 0.5 // prune_per_src)))
    return {"host budget": host, "device budget": dev, "group": group, "query batch": qb, "chunk": chunk,
            "merge batch": merge_batch, "pr batch": pr_batch, "prune batch": prune_batch, "per source": per_src}


def fingerprint(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def run_dataset(folder, args):
    folder = Path(folder)
    name = folder.name
    adj, computed = args.graph_dir / f"adj-list-{name}.txt", args.graph_dir / f"{name}-computed.txt"
    missing = [str(p) for p in (adj, computed) if not p.exists()]
    if missing:
        msg = f"{name}: missing {', '.join(missing)}"
        if args.skip_missing:
            log(msg + "; skipped")
            return
        raise FileNotFoundError(msg)
    f = localdim.NpyFile(folder / "vectors.npy")
    n = f.shape[0]
    rows = args.rows.get(name, n)
    log(f"{name}: reading graph files {adj} and {computed}")
    if args.local_prune:
        sources, out_deg, out_nbrs = localdim.computed_out_neighbours(adj, computed)
    else:
        (sources, out_deg), out_nbrs = localdim.computed_out_degrees(adj, computed), None
    info = {"source": "npy", "dataset": str(folder), "adj": str(adj), "computed": str(computed)}
    compute(name + (f"-n{rows}" if rows != n else ""), f, rows, sources, out_deg, info, args, out_nbrs)


def compute(name, f, rows, sources, out_deg, info, args, out_nbrs=None):
    """Steps 2-3 for one point set and its graph, written to OUT_DIR/<name>/. f: NpyFile or ArrayFile; rows: the
    graph was built on f's first `rows` points; sources (sorted) and out_deg: the points with a computed
    out-neighbourhood; info: extra meta.json entries (e.g. where the data and graph came from); out_nbrs:
    {source: out-neighbour ids}, needed with --local-prune. Shared with local_dim_vibe.py."""
    if args.local_prune and out_nbrs is None:
        raise ValueError("--local-prune needs the out-neighbour lists")
    t0 = time.time()
    n, d = f.shape
    out = args.out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    log(f"{name}: {len(sources):,} sources with a computed out-neighbourhood")
    assert len(sources), f"{name}: no sources with a computed out-neighbourhood"
    assert sources[-1] < rows, f"{name}: source {sources[-1]} is outside the first {rows:,} rows"
    if args.max_sources is not None and args.max_sources < len(sources):
        keep = np.sort(np.random.default_rng(args.seed).choice(len(sources), args.max_sources, replace=False))
        sources, out_deg = sources[keep], out_deg[keep]
    ks = localdim.k_grid(args.k, d, args.k_d_relative, rows)
    K, q = ks[-1], len(sources)
    device = localdim.resolve_device(args.device)
    p = plan(q, K, d, f.dtype.itemsize, rows, device, args.host_mem_fraction, args.gpu_mem_fraction,
             args.max_chunk, np.dtype(args.pr_dtype).itemsize, args.local_prune)
    meta_path = out / "meta.json"
    old = json.loads(meta_path.read_text()) if meta_path.exists() and not args.overwrite else None
    if old is not None and old.get("group") and old["group"] != p["group"]:
        # resuming: part files are per group, so keep the earlier grouping
        log(f"{name}: resuming with the earlier group size {old['group']:,} (planned {p['group']:,} now)")
        p["group"] = old["group"]
        p["pr batch"], p["prune batch"] = min(p["pr batch"], p["group"]), min(p["prune batch"], p["group"])
    groups = [(a, min(a + p["group"], q)) for a in range(0, q, p["group"])]
    log(f"{name}: {q:,} sources, {rows:,} x {d} {f.dtype} points, k = {ks}, device {device}\n"
          f"  budgets: host {gib(p['host budget'])}, device {gib(p['device budget'])}; "
          f"{gib(p['per source'])} per source\n"
          f"  {len(groups)} pass(es) of up to {p['group']:,} sources; chunk {p['chunk']:,} rows "
          f"({gib(p['chunk'] * d * f.dtype.itemsize)}), query batch {p['query batch']:,}, merge batch {p['merge batch']:,}, PR batch {p['pr batch']:,}"
          + (f", local-prune batch {p['prune batch']:,} (alpha {args.prune_alpha:g})" if args.local_prune else ""))

    meta = info | {"name": name, "n": n, "d": d,
            "dtype": str(f.dtype), "rows": rows, "ks": ks, "K": K, "centres": list(localdim.CENTRES),
            "estimator": args.estimator, "pr dtype": args.pr_dtype, "sources": q, "sources sha1": fingerprint(sources),
            "max sources": args.max_sources, "seed": args.seed, "group": p["group"], "groups": len(groups),
            "local prune": bool(args.local_prune or (old or {}).get("local prune")),
            "prune alpha": args.prune_alpha if args.local_prune else (old or {}).get("prune alpha")}
    if old is not None:
        keys = ("rows", "ks", "estimator", "pr dtype", "sources sha1")
        if any(old.get(k) != meta[k] for k in keys):
            raise SystemExit(f"{out} holds a run with different settings ({', '.join(k for k in keys if old.get(k) != meta[k])}); "
                             "pass --overwrite or a different --out-dir")
    else:
        for part in out.glob("part-*.npz"):
            part.unlink()
    meta_path.write_text(json.dumps(meta, indent=2))
    np.save(out / "sources.npy", sources)
    np.save(out / "out_degree.npy", out_deg)

    def memmap(fname, shape, dtype):
        path = out / fname
        if path.exists() and not args.overwrite:
            m = np.load(path, mmap_mode="r+")
            if m.shape == shape and m.dtype == dtype:
                return m
        return open_memmap(path, mode="w+", shape=shape, dtype=dtype)

    if args.save_vectors:
        log(f"{name}: neighbour vectors take {gib(q * K * d * f.dtype.itemsize)} on disk")
        src_v = memmap("source_vectors.npy", (q, d), f.dtype)
        nbr_v = memmap("nbr_vectors.npy", (q, K, d), f.dtype)
    nbr_d = memmap("nbr_sq_dists.npy", (q, K), np.float32)
    if args.local_prune:
        nbr_i = memmap("nbr_ids.npy", (q, K), np.int64)

    import torch
    for g, (a, b) in enumerate(groups):
        part, prune_part = out / f"part-{g}.npz", out / f"part-{g}-prune.npz"
        if part.exists() and (prune_part.exists() or not args.local_prune):
            log(f"{name}: group {g + 1}/{len(groups)} already done, skipped")
            continue
        log(f"{name}: group {g + 1}/{len(groups)}: reading {b - a:,} source vectors")
        Q_raw = f.rows(sources[a:b])
        res = localdim.knn_vectors(f, rows, sources[a:b], Q_raw, K, device, p["chunk"], p["query batch"],
                                   mininterval=args.progress_every, merge_batch=p["merge batch"],
                                   with_ids=args.local_prune)
        best_v, best_d = res[:2]
        nbr_d[a:b] = best_d.cpu().numpy()
        nbr_d.flush()
        if args.local_prune:
            knn_ids = res[2].cpu().numpy()
            nbr_i[a:b] = knn_ids
            nbr_i.flush()
        if args.save_vectors:
            log(f"{name}: group {g + 1}/{len(groups)}: writing neighbour vectors to {out / 'nbr_vectors.npy'}")
            src_v[a:b] = Q_raw
            step = max(1, (1 << 30) // max(1, K * d * f.dtype.itemsize))   # copy back ~1 GiB at a time
            for s in range(0, b - a, step):
                e = min(s + step, b - a)
                nbr_v[a + s : a + e] = best_v[s:e].cpu().numpy()
            nbr_v.flush(), src_v.flush()

        log(f"{name}: group {g + 1}/{len(groups)}: local PR for {len(ks)} k values x {len(localdim.CENTRES)} centrings")
        Qt = torch.as_tensor(Q_raw, device=device)
        pr = {key: np.empty(b - a) for key in ((c, k) for c in localdim.CENTRES for k in ks)}
        for s in range(0, b - a, p["pr batch"]):
            e = min(s + p["pr batch"], b - a)
            for key, val in localdim.local_pr_batch(best_v[s:e], Qt[s:e], ks, localdim.CENTRES,
                                                    args.estimator, args.pr_dtype).items():
                pr[key][s:e] = val
        if args.local_prune:
            log(f"{name}: group {g + 1}/{len(groups)}: out-neighbours in the kNN and local robust prune "
                f"(alpha {args.prune_alpha:g}) for {len(ks)} k values")
            lp = {"out_in_knn": localdim.out_in_knn([out_nbrs[int(v)] for v in sources[a:b]], knn_ids, ks)}
            deg = {k: np.empty(b - a, dtype=np.int64) for k in ks}
            nbr = {k: [] for k in ks}
            unc = {k: [] for k in ks}
            for s in localdim.track(range(0, b - a, p["prune batch"]), -(-(b - a) // p["prune batch"]),
                                    "local prune", "batch", args.progress_every):
                e = min(s + p["prune batch"], b - a)
                for k, (edges, left, dg) in localdim.local_robust_prune(best_v[s:e], best_d[s:e], ks,
                                                                        args.prune_alpha).items():
                    deg[k][s:e] = dg
                    valid = edges >= 0                     # row-major: each source's edges in order
                    nbr[k].append(knn_ids[s:e][np.nonzero(valid)[0], edges[valid]])
                    unc[k].append(left[valid])
            np.savez(prune_part, **{f"out_in_knn_k{k}": lp["out_in_knn"][k] for k in ks},
                     **{f"lp_degree_k{k}": deg[k] for k in ks},
                     **{f"lp_nbrs_k{k}": np.concatenate(nbr[k]) for k in ks},
                     **{f"lp_uncovered_k{k}": np.concatenate(unc[k]) for k in ks})
        np.savez(part, **{f"pr_{c}_k{k}": v for (c, k), v in pr.items()})   # written last: marks the group done
        del best_v, best_d, Qt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        log(f"{name}: group {g + 1}/{len(groups)} done ({localdim._hms(time.time() - t0)} so far)")

    parts = [np.load(out / f"part-{g}.npz") for g in range(len(groups))]
    cols = {key: np.concatenate([z[key] for z in parts]) for key in parts[0].files}
    if args.local_prune:
        lps = [np.load(out / f"part-{g}-prune.npz") for g in range(len(groups))]
        for k in ks:
            cols[f"out_in_knn_k{k}"] = np.concatenate([z[f"out_in_knn_k{k}"] for z in lps])
            cols[f"lp_degree_k{k}"] = np.concatenate([z[f"lp_degree_k{k}"] for z in lps])
            write_local_prune(out / f"local_prune_k{k}.txt", sources, cols[f"lp_degree_k{k}"],
                              np.concatenate([z[f"lp_nbrs_k{k}"] for z in lps]),
                              np.concatenate([z[f"lp_uncovered_k{k}"] for z in lps]))
        log(f"{name}: wrote {out}/local_prune_k<k>.txt for k = {ks}")
    df = pd.DataFrame({"source": sources, "out_degree": out_deg} | cols)
    df.to_csv(out / "local_pr.csv", index=False)
    log(f"{name}: wrote {out / 'local_pr.csv'} ({localdim._hms(time.time() - t0)})")


def write_local_prune(path, sources, degree, nbrs, uncovered):
    """Local-prune neighbourhoods in the vibe.py adjacency format, one line per source:
    "source [(neighbour id, points of N_k(v) still uncovered after this edge), ...]"."""
    ends = np.cumsum(degree)
    with open(path, "w") as fh:
        for v, e, dg in zip(sources, ends, degree):
            fh.write(f"{v} [" + ", ".join(f"({n}, {u})" for n, u in zip(nbrs[e - dg : e], uncovered[e - dg : e]))
                     + "]\n")


def add_compute_args(p):
    """Arguments shared with local_dim_vibe.py: output, sources, k grid, estimator, device and memory."""
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "local_dim",
                   help="results go to OUT_DIR/<name>/ (default: results/local_dim, as in the notebook)")
    p.add_argument("--max-sources", type=int, help="a seeded random subset of this many sources")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, nargs="+", default=K_BASE, help=f"k grid (default {K_BASE})")
    p.add_argument("--k-d-relative", type=float, nargs="*", default=[0.5, 1, 2],
                   help="also k = round(f * d) for each f (default 0.5 1 2)")
    p.add_argument("--estimator", choices=["plugin", "u"], default="plugin")
    p.add_argument("--pr-dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--device", default="auto", help='"auto" (CUDA if available), "cpu", "cuda:1", ...')
    p.add_argument("--host-mem-fraction", type=float, default=0.5, help="share of available RAM to use")
    p.add_argument("--gpu-mem-fraction", type=float, default=0.85, help="share of free GPU memory to use")
    p.add_argument("--max-chunk", type=int, default=1 << 22, help="upper bound on rows per block")
    p.add_argument("--skip-missing", action="store_true", help="skip datasets without data or graph files")
    p.add_argument("--no-save-vectors", dest="save_vectors", action="store_false",
                   help="do not write source_vectors.npy / nbr_vectors.npy (q x K x d; can be terabytes)")
    p.add_argument("--local-prune", action="store_true",
                   help="also: the fraction of each source's out-neighbourhood inside N_k(v), and robust prune from v "
                        "over N_k(v) alone (local_prune_k<k>.txt per k; out_in_knn_k<k> and lp_degree_k<k> columns)")
    p.add_argument("--prune-alpha", type=float, default=1.0, help="alpha of the local robust prune (default 1)")
    p.add_argument("--overwrite", action="store_true", help="discard earlier results in OUT_DIR/<name>/")
    p.add_argument("--progress-every", type=float, default=30, help="seconds between progress updates")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", required=True, type=Path,
                   help="dataset folder holding vectors.npy (repeat for several)")
    p.add_argument("--graph-dir", required=True, type=Path,
                   help="folder with adj-list-<name>.txt and <name>-computed.txt for each dataset")
    p.add_argument("--rows", action="append", default=[], metavar="NAME=N",
                   help="the graph of dataset NAME was built on its first N points (repeatable)")
    add_compute_args(p)
    args = p.parse_args()
    try:
        args.rows = {k: int(v) for k, v in (r.split("=", 1) for r in args.rows)}
    except ValueError:
        p.error("--rows takes NAME=N, e.g. --rows bigann=10000000")
    for folder in args.dataset:
        run_dataset(folder, args)


if __name__ == "__main__":
    main()
