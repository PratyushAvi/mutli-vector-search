"""Build adversarial-only ArguAna keyword-stuffing corpora without importing torch.

Usage: python src/make_attack_corpus.py --dataset arguana \
    --methods targeted topic hub --keywords 60 --seed 0 --overwrite

Requires pandas, pyarrow, numpy. See the generated corpus_modified/README.md
for definitions and src/verify_attack_corpus.py for optional tokenizer checks.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from paths import DATASETS_DIR

METHODS = ("targeted", "topic", "hub")
PARQUET_NAME = "corpus-00000-of-00001.parquet"
WORD_RE = re.compile(r"[a-z][a-z'-]+")
# Frozen built-in English stopwords: no downloads or NLP dependencies.
STOPWORDS = frozenset("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours
yourself yourselves he him his himself she she's her hers herself it it's its
itself they them their theirs themselves what which who whom this that that'll
these those am is are was were be been being have has had having do does did
doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down
in out on off over under again further then once here there when where why how
all any both each few more most other some such no nor not only own same so
than too very s t can will just don don't should should've now d ll m o re ve
y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn
hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't
shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn
wouldn't
""".split())


def tokenize(text):
    return [w for w in WORD_RE.findall(text.lower())
            if len(w) >= 3 and w not in STOPWORDS]


def topic_of(doc_id):
    fields = doc_id.split("-")
    if len(fields) < 3 or not fields[1]:
        raise ValueError(f"ID has no ArguAna topic field: {doc_id!r}")
    return fields[1]


def read_split(folder, columns):
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {folder}")
    tables = [pq.read_table(p) for p in files]
    schema = tables[0].schema
    if any(not t.schema.equals(schema) for t in tables):
        raise ValueError(f"Inconsistent parquet schemas in {folder}")
    frame = pa.concat_tables(tables).to_pandas()
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"{folder} requires columns {columns}")
    if frame.empty or frame["_id"].isna().any():
        raise ValueError(f"Empty split or null IDs in {folder}")
    frame["_id"] = frame["_id"].astype(str)
    if frame["_id"].duplicated().any():
        raise ValueError(f"Duplicate IDs in {folder}")
    for col in ("title", "text"):
        if col in frame:
            frame[col] = frame[col].fillna("")
    return frame.sort_values("_id", kind="stable").reset_index(drop=True), schema


def load_inputs(dataset_dir):
    corpus, schema = read_split(dataset_dir / "corpus", ("_id", "title", "text"))
    queries, _ = read_split(dataset_dir / "queries", ("_id", "text"))
    if schema.names != ["_id", "title", "text"]:
        raise ValueError("Expected the original corpus schema: _id, title, text")
    qrels_path = dataset_dir / "qrels" / "test.tsv"
    if qrels_path.exists():
        qrels = pd.read_csv(qrels_path, sep="\t", dtype={"query-id": str, "corpus-id": str})
    else:
        files = sorted((dataset_dir / "qrels").glob("test*.parquet"))
        if not files:
            raise FileNotFoundError(f"Missing {qrels_path} (or test*.parquet)")
        qrels = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if not {"query-id", "corpus-id", "score"}.issubset(qrels.columns):
        raise ValueError("Qrels require query-id, corpus-id, score columns")
    relevant = defaultdict(set)
    for row in qrels.to_dict("records"):
        if float(row["score"]) > 0:
            relevant[str(row["query-id"])].add(str(row["corpus-id"]))
    for query_id in queries["_id"]:
        if not relevant[query_id]:
            raise ValueError(f"No positive test qrels for {query_id}")
        # Some released qrels reference absent corpus IDs. Retain them in the
        # exclusion set; they cannot be selected as carriers in this corpus.
    return corpus, queries, schema, relevant


def rng_for(seed, method, doc_id, purpose):
    # Avoid Python's randomized hash and method-order/global-RNG dependencies.
    key = json.dumps([seed, method, doc_id, purpose], separators=(",", ":"))
    entropy = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big")
    return np.random.default_rng(entropy)


def rank_terms(counts, idf, unseen_idf):
    return sorted(counts, key=lambda w: (-counts[w] * idf.get(w, unseen_idf), w))


def build_corpora(corpus, queries, relevant, methods=METHODS, keywords=60, seed=0):
    """Return method -> (document records, manifest records), in stable ID order."""
    if keywords < 1 or seed < 0:
        raise ValueError("keywords must be positive and seed must be nonnegative")
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError(f"Choose distinct methods from {METHODS}")
    docs = corpus.sort_values("_id").to_dict("records")
    query_rows = queries.sort_values("_id").to_dict("records")
    original_ids = {d["_id"] for d in docs}
    df = Counter()
    topic_tf = defaultdict(Counter)
    for doc in docs:
        counts = Counter(tokenize(doc["title"] + " " + doc["text"]))
        df.update(counts.keys())
        topic_tf[topic_of(doc["_id"])].update(counts)
    idf = {w: math.log((len(docs) + 1) / (freq + 1)) + 1 for w, freq in df.items()}
    unseen_idf = math.log(len(docs) + 1) + 1
    topic_pools = {t: rank_terms(counts, idf, unseen_idf)[:300]
                   for t, counts in topic_tf.items()}
    hub_pool = sorted(df, key=lambda w: (-df[w], w))[:500]
    candidates = {}
    for q in query_rows:
        query_id = q["_id"]
        topic = topic_of(query_id)
        excluded = relevant[query_id] | {query_id}
        candidates[query_id] = [i for i, d in enumerate(docs)
                                if topic_of(d["_id"]) != topic and d["_id"] not in excluded]
        if not candidates[query_id]:
            raise ValueError(f"No eligible carrier for {query_id}")
    result = {}
    generated_ids = set()
    for method in methods:
        rows, manifests = [], []
        for index, query in enumerate(query_rows):
            query_id = query["_id"]
            topic = topic_of(query_id)
            doc_id = f"adv-hub-{index:04d}" if method == "hub" else f"adv-{method}-{query_id}"
            if doc_id in original_ids or doc_id in generated_ids:
                raise ValueError(f"Adversarial ID collision: {doc_id}")
            generated_ids.add(doc_id)
            if method == "targeted":
                words = rank_terms(Counter(tokenize(query["text"])), idf, unseen_idf)[:keywords]
                if not words:
                    raise ValueError(f"Query has no content keywords: {query_id}")
            else:
                pool = topic_pools.get(topic, []) if method == "topic" else hub_pool
                if len(pool) < keywords:
                    raise ValueError(f"{method}/{topic}: only {len(pool)} pool words for {keywords} keywords")
                words = rng_for(seed, method, doc_id, "keywords").choice(
                    pool, size=keywords, replace=False).tolist()
            carrier_index = int(rng_for(seed, method, doc_id, "carrier").choice(candidates[query_id]))
            carrier = docs[carrier_index]
            rows.append({"_id": doc_id, "title": carrier["title"],
                         "text": " ".join(words) + " " + carrier["text"]})
            manifest = {"_id": doc_id, "method": method,
                        "target_query_id": None if method == "hub" else query_id,
                        "carrier_doc_id": carrier["_id"], "injected_keywords": words,
                        "n_keywords": len(words), "seed": seed, "requested_keywords": keywords}
            if method == "topic":
                manifest["topic"] = topic
            if method == "hub":
                # Association is for carrier exclusions only, never keyword choice.
                manifest["carrier_reference_query_id"] = query_id
            manifests.append(manifest)
        result[method] = rows, manifests
    return result


def readme_text(manifests_by_method, corpus_count):
    lines = ["# ArguAna keyword-stuffing attack corpora", "",
             "Adversarial documents only; the original corpus is not copied. Each selected",
             "method produces one document per query. Concatenate its parquet with the",
             "original corpus for a later experiment. Encoding/search/evaluation are unchanged.", "",
             "## Methods and parameters", "",
             "- Word tokenizer: lowercase `[a-z][a-z'-]+`, remove the frozen English",
             "  stopword list in `src/make_attack_corpus.py` and words shorter than 3 characters.",
             f"- IDF over {corpus_count:,} original title + text documents:",
             "  `log((N + 1) / (df + 1)) + 1`. Salience is raw term count times IDF.",
             "  Unseen query terms use df=0. Ties are broken alphabetically.",
             "- targeted: top K distinct query-text terms by TF-IDF. Short queries use all",
             "  available terms, with no repetition or padding from outside the query.",
             "  IDs: `adv-targeted-<query_id>`.",
             "- topic: aggregate raw term counts over original title + text by topic,",
             "  rank by TF-IDF, retain the top 300 and sample K without replacement per doc.",
             "  Topic is the second hyphen-separated ID field; IDs: `adv-topic-<query_id>`.",
             "- hub: pool of 500 content terms with the highest corpus document frequency;",
             "  sample K without replacement per doc. IDs: `adv-hub-0000`, etc.",
             "  Keyword selection never uses query text or topic; target_query_id is null.", "",
             "Carriers are uniform random original documents from a different topic, excluding",
             "the associated query ID and every positive test-qrels document. Hub documents",
             "use sorted query IDs only for these carrier exclusions; the association is",
             "recorded in carrier_reference_query_id, not as an attack target.", "",
             "Title is the carrier title; text is `<keyword block> <unmodified carrier text>`.",
             "Keywords go first in the text, after the title when the encoder joins them.",
             "The default K=60 is an approximate token budget, not a token-length guarantee.",
             "Use the verifier to check title + keywords with bert-base-uncased; it reserves",
             "[CLS], [SEP], and a ColBERT [D] slot within the 180-token document window.", "",
             "Inputs and output rows are sorted by ID. Each document has independent NumPy",
             "PCG64 streams seeded by SHA-256 of [seed, method, document ID, purpose].",
             "Results do not depend on method ordering or Python hash randomization.",
             "Parquet preserves the original Arrow schema, including metadata, and uses",
             "fixed Zstandard/version settings. Byte identity is expected with the same",
             "inputs and dependency versions; cross-version writer identity is not promised.", "",
             "## Generated files", "",
             "Each attack_corpus_<method>/ contains corpus-00000-of-00001.parquet",
             "(_id, title, text) and manifest.jsonl. Manifest fields include _id, method,",
             "target_query_id, carrier_doc_id, injected_keywords, n_keywords, seed,",
             "requested_keywords, plus topic or carrier_reference_query_id when applicable.", "",
             "| Method | Documents | Seed | Requested K | Actual K range | Short queries |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    for method in METHODS:
        if method not in manifests_by_method:
            continue
        records = manifests_by_method[method]
        counts = [r["n_keywords"] for r in records]
        seeds = sorted({r["seed"] for r in records})
        requested = sorted({r["requested_keywords"] for r in records})
        short = sum(r["n_keywords"] < r["requested_keywords"] for r in records)
        lines.append(f"| {method} | {len(records)} | {', '.join(map(str, seeds))} | "
                     f"{', '.join(map(str, requested))} | {min(counts)}–{max(counts)} | {short} |")
    lines += ["", "## Reproduce and verify", "", "```bash",
              "python src/make_attack_corpus.py --dataset arguana --methods targeted topic hub --keywords 60 --seed 0 --overwrite",
              "python src/verify_attack_corpus.py --dataset arguana --check-tokenizer --check-determinism",
              "```", "", "Generation requires pandas, pyarrow, numpy; verification optionally uses",
              "transformers and a cached bert-base-uncased tokenizer (or --tokenizer-path).",
              "The verifier reports unavailable tokenizer dependencies explicitly.", ""]
    return "\n".join(lines)


def generate(dataset_dir, methods=METHODS, keywords=60, seed=0, overwrite=False):
    dataset_dir = Path(dataset_dir)
    output = dataset_dir / "corpus_modified"
    for method in methods:
        folder = output / f"attack_corpus_{method}"
        if folder.exists() and not overwrite:
            raise FileExistsError(f"{folder} exists; use --overwrite")
    corpus, queries, schema, relevant = load_inputs(dataset_dir)
    result = build_corpora(corpus, queries, relevant, methods, keywords, seed)
    all_manifests = {}
    # Keep README accurate when only a subset is regenerated.
    for method in METHODS:
        if method in result:
            all_manifests[method] = result[method][1]
        else:
            manifest_path = output / f"attack_corpus_{method}" / "manifest.jsonl"
            if manifest_path.exists():
                all_manifests[method] = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    readme = readme_text(all_manifests, len(corpus))
    output.mkdir(parents=True, exist_ok=True)
    # Finish all serialization before replacing any previous generated files.
    with tempfile.TemporaryDirectory(prefix=".attack-", dir=output) as temp:
        staged = Path(temp)
        for method, (rows, manifests) in result.items():
            folder = staged / f"attack_corpus_{method}"
            folder.mkdir()
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), folder / PARQUET_NAME,
                           compression="zstd", version="2.6", row_group_size=1024)
            (folder / "manifest.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                        for r in manifests), encoding="utf-8")
        (staged / "README.md").write_text(readme, encoding="utf-8")
        for method in result:
            folder = output / f"attack_corpus_{method}"
            folder.mkdir(exist_ok=True)
            for name in (PARQUET_NAME, "manifest.jsonl"):
                (staged / folder.name / name).replace(folder / name)
        (staged / "README.md").replace(output / "README.md")
    by_query = queries.set_index("_id")["text"].to_dict()
    for method, (rows, manifests) in result.items():
        record = manifests[0]
        target = record["target_query_id"]
        print(f"\n{method}: {len(rows)} docs -> {output / ('attack_corpus_' + method)}")
        print(f"  sample: {record['_id']} | carrier: {record['carrier_doc_id']}")
        print(f"  target query: {target if target is not None else 'null (query-agnostic)'}")
        if target is not None:
            print(f"  query text: {by_query[target]}")
        else:
            reference = record["carrier_reference_query_id"]
            print(f"  carrier-exclusion reference only: {reference}\n  reference query: {by_query[reference]}")
        print(f"  keywords ({record['n_keywords']}): {' '.join(record['injected_keywords'])}")
        print(f"  title: {rows[0]['title']}\n  document: {rows[0]['text']}")
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", default="arguana", choices=["arguana"])
    result.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    result.add_argument("--keywords", type=int, default=60)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--overwrite", action="store_true")
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        generate(DATASETS_DIR / args.dataset, args.methods, args.keywords, args.seed, args.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
