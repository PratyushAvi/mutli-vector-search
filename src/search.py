"""
Exhaustive multi-vector search: score every query against every corpus item with a
ranker and write the top-k per query.

Reads   datasets/<name>/vec_{queries,corpus}/<encoder>/<tag>.*   (written by encode.py)
Writes  results/<name>/<encoder>/<ranker>__<query tag>__<corpus tag>.trec
        results/<name>/<encoder>/<ranker>__<query tag>__<corpus tag>.meta.json

The .trec file is a standard TREC run (`qid Q0 docid rank score run`), readable by
trec_eval / pytrec_eval / ir_measures.

The corpus is streamed once in length-sorted chunks (bounded padding and memory);
every chunk is scored against all queries and merged into a running top-k.

Usage:
  python src/search.py --ranker chamfer --encoder colbert --datasets arguana
  python src/search.py --ranker sw --encoder bert --datasets scidocs --k 100 --projections 128

From Python / a notebook:
  from search import run_search
  out = run_search("arguana", "chamfer", ignore_identical_ids=True)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from encode import DATASETS_DIR, ROOT, load_vectors, pick_device
from methods import ChamferRanker, SlicedWassersteinRanker

RESULTS_DIR = ROOT / "results"


def find_prefix(folder, tag):
    """Resolve `<folder>/<tag>` or the only vector set in `folder`."""
    if tag:
        return folder / tag
    tags = sorted(p.name.removesuffix(".meta.json") for p in folder.glob("*.meta.json"))
    if len(tags) != 1:
        raise SystemExit(f"{folder}: found {tags or 'no vector sets'}; pick one with --query-tag/--corpus-tag")
    return folder / tags[0]


def padded(vectors, offsets, items, device, dtype):
    """Gather items into a padded tensor [n, L, dim] and boolean mask [n, L]."""
    starts, counts = offsets[items], offsets[items + 1] - offsets[items]
    ar = np.arange(counts.max())
    mask = ar[None, :] < counts[:, None]
    idx = starts[:, None] + np.minimum(ar[None, :], counts[:, None] - 1)
    X = np.asarray(vectors[idx.ravel()]).reshape(len(items), len(ar), -1)
    return torch.from_numpy(X).to(device, dtype), torch.from_numpy(mask).to(device)


def length_chunks(counts, items, budget):
    """Split items (sorted by length) into chunks of at most ~budget padded vectors."""
    items = items[np.argsort(counts[items], kind="stable")]
    chunk, longest = [], 0
    for i in items:
        longest = max(longest, counts[i])
        if chunk and (len(chunk) + 1) * longest > budget:
            yield np.array(chunk)
            chunk, longest = [], counts[i]
        chunk.append(i)
    if chunk:
        yield np.array(chunk)


def search(ranker, queries, corpus, k, device, dtype, query_block, chunk_vectors):
    (QV, qoff), (DV, doff) = queries, corpus
    nq = len(qoff) - 1
    dcounts = np.diff(doff)
    docs = np.flatnonzero(dcounts > 0)  # empty documents can never match
    k = min(k, len(docs))

    # Queries are small: encode them all once, in blocks.
    blocks = [np.arange(s, min(s + query_block, nq)) for s in range(0, nq, query_block)]
    qreps = [ranker.encode_queries(*padded(QV, qoff, b, device, dtype)) for b in blocks]

    top_s = torch.full((nq, k), -float("inf"), device=device)
    top_i = torch.full((nq, k), -1, dtype=torch.long, device=device)
    t0, done = time.time(), 0
    for chunk in length_chunks(dcounts, docs, chunk_vectors):
        drep = ranker.encode_docs(*padded(DV, doff, chunk, device, dtype))
        cidx = torch.from_numpy(chunk).to(device)
        for b, qrep in zip(blocks, qreps):
            s = ranker.score(qrep, drep).float()
            rows = slice(b[0], b[-1] + 1)
            merged_s = torch.cat([top_s[rows], s], dim=1)
            merged_i = torch.cat([top_i[rows], cidx.expand(len(b), -1)], dim=1)
            top_s[rows], pos = merged_s.topk(k, dim=1)
            top_i[rows] = merged_i.gather(1, pos)
        done += len(chunk)
        print(f"\r  {done}/{len(docs)} docs  ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    return top_s.cpu().numpy(), top_i.cpu().numpy()


def make_ranker(name, dim, device, symmetric=False, projections=64, quantiles=32, p=2, seed=0):
    if name == "chamfer":
        return ChamferRanker(symmetric=symmetric)
    if name == "sw":
        return SlicedWassersteinRanker(dim, projections, quantiles, p, seed, device)
    raise ValueError(f"unknown ranker {name!r}")


def run_search(dataset, ranker="chamfer", encoder="colbert", query_tag=None, corpus_tag=None,
               k=100, ignore_identical_ids=False, device="auto", dtype="auto",
               query_block=32, chunk_vectors=131072, write=True, **ranker_opts):
    """Search one dataset. `ranker` is "chamfer", "sw", or a ranker instance; extra keyword
    arguments go to make_ranker (symmetric, projections, quantiles, p, seed).

    Returns a dict with `results` ({qid: [(docid, score), ...]}), `meta`, and `path` (or None).
    """
    device = pick_device(device)
    dtype = {"float16": torch.float16, "float32": torch.float32}.get(
        dtype, torch.float32 if device == "cpu" else torch.float16)

    qprefix = find_prefix(DATASETS_DIR / dataset / "vec_queries" / encoder, query_tag)
    dprefix = find_prefix(DATASETS_DIR / dataset / "vec_corpus" / encoder, corpus_tag)
    QV, qoff, qids, qmeta = load_vectors(qprefix)
    DV, doff, dids, dmeta = load_vectors(dprefix)
    if qmeta["dim"] != dmeta["dim"]:
        raise SystemExit(f"dimension mismatch: {qprefix.name} vs {dprefix.name}")
    if isinstance(ranker, str):
        ranker = make_ranker(ranker, dmeta["dim"], device, **ranker_opts)

    print(f"[{dataset}] {ranker.tag}: {len(qids)} queries x {len(dids)} docs  device={device} dtype={dtype}")
    extra = 1 if ignore_identical_ids else 0
    t0 = time.time()
    scores, idx = search(ranker, (QV, qoff), (DV, doff), k + extra, device, dtype, query_block, chunk_vectors)
    elapsed = time.time() - t0

    results = {}
    for qi, qid in enumerate(qids):
        hits = [(dids[j], float(s)) for j, s in zip(idx[qi], scores[qi])
                if j >= 0 and not (ignore_identical_ids and dids[j] == qid)]
        results[qid] = hits[:k]
    meta = {
        "dataset": dataset, "encoder": encoder, **ranker.config(),
        "queries": qprefix.name, "corpus": dprefix.name, "k": k,
        "ignore_identical_ids": ignore_identical_ids,
        "num_queries": len(qids), "num_docs": len(dids),
        "device": device, "dtype": str(dtype).removeprefix("torch."),
        "seconds": round(elapsed, 1),
    }

    path = None
    if write:
        out_dir = RESULTS_DIR / dataset / encoder
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"{ranker.tag}__{qprefix.name}__{dprefix.name}"
        with open(f"{stem}.trec", "w") as f:
            for qid, hits in results.items():
                for rank, (did, s) in enumerate(hits, 1):
                    f.write(f"{qid} Q0 {did} {rank} {s:.6f} {ranker.tag}\n")
        with open(f"{stem}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        path = Path(f"{stem}.trec")
        print(f"  wrote {path.relative_to(ROOT)}  ({elapsed:.0f}s)")
    return {"results": results, "meta": meta, "path": path}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ranker", choices=["chamfer", "sw"], required=True)
    p.add_argument("--encoder", choices=["colbert", "bert"], default="colbert")
    p.add_argument("--datasets", nargs="*", help="dataset names (default: all with vectors for --encoder)")
    p.add_argument("--query-tag", help="e.g. colbertv2.0_d128_nv32 (default: the only one present)")
    p.add_argument("--corpus-tag", help="e.g. colbertv2.0_d128_nv180 (default: the only one present)")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--ignore-identical-ids", action="store_true",
                   help="drop results whose doc id equals the query id (BEIR convention for ArguAna)")
    p.add_argument("--symmetric", action="store_true", help="chamfer: symmetric instead of query->doc")
    p.add_argument("--projections", type=int, default=64, help="sw: number of random directions L")
    p.add_argument("--quantiles", type=int, default=32, help="sw: quantile grid size K")
    p.add_argument("--p", type=int, choices=[1, 2], default=2, help="sw: Wasserstein order")
    p.add_argument("--seed", type=int, default=0, help="sw: projection seed")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto",
                   help="compute dtype for vectors (auto: float16 on GPU, float32 on CPU)")
    p.add_argument("--query-block", type=int, default=32)
    p.add_argument("--chunk-vectors", type=int, default=131072, help="padded doc vectors per chunk")
    args = p.parse_args()

    names = args.datasets or sorted(
        d.name for d in DATASETS_DIR.iterdir() if (d / "vec_corpus" / args.encoder).is_dir())
    ranker_opts = {"chamfer": {"symmetric": args.symmetric},
                   "sw": {"projections": args.projections, "quantiles": args.quantiles,
                          "p": args.p, "seed": args.seed}}[args.ranker]
    for name in names:
        run_search(name, args.ranker, args.encoder, args.query_tag, args.corpus_tag, args.k,
                   args.ignore_identical_ids, args.device, args.dtype, args.query_block,
                   args.chunk_vectors, **ranker_opts)


if __name__ == "__main__":
    main()
