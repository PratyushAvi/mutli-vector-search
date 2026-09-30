"""
Download OBLIQ-Bench (https://huggingface.co/datasets/dianetc/OBLIQ-Bench) and convert each
task into the BEIR layout used by the rest of the codebase:

    datasets/obliq-<task>/corpus/corpus-00000-of-00001.parquet     _id, title, text
    datasets/obliq-<task>/queries/queries-00000-of-00001.parquet   _id, title, text
    datasets/obliq-<task>/qrels/test.tsv          gold judgments (qrels.tsv)
    datasets/obliq-<task>/qrels/pool.tsv          pooled judgments (qrels_pool.tsv), if provided
    datasets/obliq-<task>/excluded_ids.json       {query_id: [doc_id, ...]} to drop from results,
                                                  if provided (math, writing); search.py applies it
    datasets/obliq-<task>/README.md               source, license, counts

Evaluate against pooled judgments with qrels split "pool" (e.g. evaluate.py --split pool).
Raw files are fetched through the Hugging Face cache (set HF_HOME to keep it off $HOME).

Usage:
  python src/convert_obliq.py                         # all five tasks
  python src/convert_obliq.py --tasks math writing twitter
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from paths import DATASETS_DIR

REPO = "dianetc/OBLIQ-Bench"
TASKS = {
    "twitter": "descriptive/twitter",
    "wildchat": "descriptive/wildchat",
    "math": "analogues/math",
    "writing": "analogues/writing",
    "congress": "tip-of-tongue/congress",
}
SCHEMA = pa.schema([("_id", pa.string()), ("title", pa.string()), ("text", pa.string())])


def fetch(path, required=True):
    try:
        return Path(hf_hub_download(REPO, path, repo_type="dataset"))
    except Exception as e:  # missing optional file (e.g. no qrels_pool.tsv for writing)
        if required:
            raise
        print(f"  (no {path}: {type(e).__name__})")
        return None


def jsonl_to_parquet(src, dst, batch_rows=50_000):
    """Stream a JSONL of {_id, text, ...} into parquet; ids are kept as strings."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n, batch = 0, {"_id": [], "title": [], "text": []}
    with open(src, encoding="utf-8") as f, pq.ParquetWriter(dst, SCHEMA) as writer:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            batch["_id"].append(str(row["_id"]))
            batch["title"].append(row.get("title") or "")
            batch["text"].append(row.get("text") or "")
            n += 1
            if len(batch["_id"]) >= batch_rows:
                writer.write_table(pa.table(batch, schema=SCHEMA))
                batch = {k: [] for k in batch}
        if batch["_id"]:
            writer.write_table(pa.table(batch, schema=SCHEMA))
    return n


def convert_qrels(src, dst):
    """Rewrite with the BEIR header (query-id, corpus-id, score) regardless of the source's."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(src, newline="") as f, open(dst, "w", newline="") as out:
        rows = csv.reader(f, delimiter="\t")
        next(rows)  # header: query-id/query_id, corpus-id/corpus_id, score
        w = csv.writer(out, delimiter="\t", lineterminator="\n")
        w.writerow(["query-id", "corpus-id", "score"])
        for q, d, s in rows:
            w.writerow([q, d, int(s)])
            n += 1
    return n


def convert(task, overwrite):
    base, out = TASKS[task], DATASETS_DIR / f"obliq-{task}"
    if (out / "README.md").exists() and not overwrite:
        print(f"[{task}] {out.name} exists, skipping (use --overwrite)")
        return
    print(f"[{task}] downloading {base}")
    corpus = fetch(f"{base}/corpus/corpus.jsonl")
    queries = fetch(f"{base}/queries+qrels/queries.jsonl")
    qrels = fetch(f"{base}/queries+qrels/qrels.tsv")
    pool = fetch(f"{base}/queries+qrels/qrels_pool.tsv", required=False)
    excluded = fetch(f"{base}/queries+qrels/per_query_excluded_ids.json", required=False)

    if out.exists():
        shutil.rmtree(out)
    counts = {
        "corpus": jsonl_to_parquet(corpus, out / "corpus" / "corpus-00000-of-00001.parquet"),
        "queries": jsonl_to_parquet(queries, out / "queries" / "queries-00000-of-00001.parquet"),
        "qrels (test)": convert_qrels(qrels, out / "qrels" / "test.tsv"),
    }
    if pool:
        counts["qrels (pool)"] = convert_qrels(pool, out / "qrels" / "pool.tsv")
    if excluded:
        ex = {str(q): [str(d) for d in ids] for q, ids in json.loads(excluded.read_text()).items()}
        (out / "excluded_ids.json").write_text(json.dumps(ex))
        counts["queries with excluded ids"] = len(ex)

    notes = ["- `qrels/test.tsv`: gold judgments."]
    if pool:
        notes.append("- `qrels/pool.tsv`: pooled judgments (qrels split `pool`).")
    if excluded:
        notes.append("- `excluded_ids.json`: documents removed from each query's results at search time.")
    lines = "\n".join(f"- {k}: {v:,}" for k, v in counts.items())
    (out / "README.md").write_text(
        f"# OBLIQ-Bench: {task}\n\n"
        f"Converted by `src/convert_obliq.py` from https://huggingface.co/datasets/{REPO} "
        f"(`{base}/`). License: CC-BY-4.0.\n\n{lines}\n\n" + "\n".join(notes) + "\n")
    print(f"  wrote datasets/{out.name}: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="*", choices=TASKS, default=list(TASKS))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    for task in args.tasks:
        convert(task, args.overwrite)


if __name__ == "__main__":
    main()
