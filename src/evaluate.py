"""
Evaluate search runs against BEIR relevance judgements.

Qrels are read from datasets/<name>/qrels/<split>.tsv (BEIR format: query-id, corpus-id,
score; a .parquet with the same columns also works). Only rows with score > 0 count as
relevant, so explicit negatives (e.g. SciDocs' score-0 rows) are treated like unjudged docs.
Every query with at least one relevant doc is evaluated; queries missing from the run
score 0. Metrics follow pytrec_eval / BEIR conventions (nDCG uses linear gain = score).

Usage:
  python src/evaluate.py --datasets arguana                 # all runs in results/arguana/*/
  python src/evaluate.py --runs results/arguana/colbert/chamfer__*.trec
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from paths import DATASETS_DIR, RESULTS_DIR
DEFAULT_METRICS = ("ndcg@10", "recall@100", "mrr@10")


def load_qrels(dataset, split="test"):
    """Returns {query_id: {doc_id: relevance}} with only relevant (score > 0) pairs."""
    folder = DATASETS_DIR / dataset / "qrels"
    files = sorted(folder.glob(f"{split}*.tsv")) + sorted(folder.glob(f"{split}*.parquet"))
    if not files:
        raise FileNotFoundError(f"no qrels for split {split!r} in {folder}")
    df = pd.concat(pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, sep="\t", dtype=str)
                   for f in files)
    df.columns = [c.replace("_", "-") for c in df.columns]
    df["score"] = df["score"].astype(int)
    qrels = defaultdict(dict)
    for q, d, s in df.loc[df["score"] > 0, ["query-id", "corpus-id", "score"]].itertuples(index=False):
        qrels[str(q)][str(d)] = s
    return dict(qrels)


def load_run(path):
    """Read a TREC run file into {query_id: [(doc_id, score), ...]} in rank order.

    Lines are tab-separated (search.py); older space-separated runs are also read, taking the
    fixed fields from both ends so a doc id containing spaces stays intact.
    """
    run = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 6:
                qid, _, did, rank, score, _ = fields
            else:
                qid, _, rest = line.split(maxsplit=2)
                did, rank, score, _ = rest.rsplit(maxsplit=3)
            run[qid].append((int(rank), did, float(score)))
    return {q: [(d, s) for _, d, s in sorted(hits)] for q, hits in run.items()}


def evaluate(results, qrels, metrics=DEFAULT_METRICS):
    """Mean of each metric ("ndcg@k", "recall@k", "mrr@k", "precision@k") over judged queries."""
    parsed = [(m, m.split("@")[0], int(m.split("@")[1])) for m in metrics]
    totals = dict.fromkeys(metrics, 0.0)
    for qid, rel in qrels.items():
        ranked = [d for d, _ in results.get(qid, [])]
        for name, kind, k in parsed:
            top = ranked[:k]
            if kind == "ndcg":
                dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(top))
                ideal = sorted(rel.values(), reverse=True)[:k]
                totals[name] += dcg / sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
            elif kind == "recall":
                totals[name] += sum(d in rel for d in top) / len(rel)
            elif kind == "precision":
                totals[name] += sum(d in rel for d in top) / k
            elif kind == "mrr":
                totals[name] += next((1 / (i + 1) for i, d in enumerate(top) if d in rel), 0.0)
            else:
                raise ValueError(f"unknown metric {name!r}")
    return {m: round(v / len(qrels), 4) for m, v in totals.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="*", help="evaluate every .trec under results/<dataset>/")
    p.add_argument("--runs", nargs="*", type=Path, help="specific .trec files (results/<dataset>/<encoder>/*.trec)")
    p.add_argument("--split", default="test")
    p.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    p.add_argument("--save", type=Path, help="also write the table to this CSV")
    args = p.parse_args()

    runs = list(args.runs or [])
    for name in args.datasets or ([] if runs else sorted(d.name for d in RESULTS_DIR.iterdir() if d.is_dir())):
        runs += sorted((RESULTS_DIR / name).glob("*/*.trec"))

    rows, qrels_cache = [], {}
    for path in runs:
        dataset, encoder = path.resolve().parent.parent.name, path.resolve().parent.name
        if dataset not in qrels_cache:
            qrels_cache[dataset] = load_qrels(dataset, args.split)
        ranker, qtag, dtag = path.stem.split("__")
        scores = evaluate(load_run(path), qrels_cache[dataset], args.metrics)
        rows.append({"dataset": dataset, "encoder": encoder, "ranker": ranker,
                     "queries": qtag, "corpus": dtag, **scores})

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if args.save:
        table.to_csv(args.save, index=False)


if __name__ == "__main__":
    main()
