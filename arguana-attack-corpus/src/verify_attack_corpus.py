"""Verify attack corpora without encoding/search/evaluation or model weights."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

import make_attack_corpus as attack
from paths import DATASETS_DIR


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(dataset_dir, methods=attack.METHODS, expected_count=1406,
           check_tokenizer=False, tokenizer_path=None, check_determinism=False):
    corpus, queries, schema, relevant = attack.load_inputs(dataset_dir)
    require(len(queries) == expected_count, f"Expected {expected_count} queries, got {len(queries)}")
    docs = corpus.set_index("_id").to_dict("index")
    query_by_id = queries.set_index("_id").to_dict("index")
    query_ids = list(queries["_id"])
    seen_ids = set(docs)
    report = {}
    tokenizer = None
    tokenizer_status = "not requested"
    if check_tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path or "bert-base-uncased"), local_files_only=True)
            tokenizer_status = str(tokenizer_path or "bert-base-uncased")
        except (ImportError, OSError) as error:
            tokenizer_status = f"SKIPPED: tokenizer unavailable ({error})"
            print(tokenizer_status)
    for method in methods:
        folder = Path(dataset_dir) / "corpus_modified" / f"attack_corpus_{method}"
        files = sorted(folder.glob("*.parquet"))
        require([p.name for p in files] == [attack.PARQUET_NAME], f"{method}: unexpected parquet shards")
        table = pq.read_table(folder / attack.PARQUET_NAME)
        require(table.schema.equals(schema, check_metadata=True), f"{method}: schema mismatch")
        rows = table.to_pylist()
        records = [json.loads(line) for line in (folder / "manifest.jsonl").read_text().splitlines()]
        require(len(rows) == len(records) == expected_count, f"{method}: wrong document/manifest count")
        max_content_wordpieces = 0
        keyword_counts = []
        for index, (row, record) in enumerate(zip(rows, records)):
            doc_id = row["_id"]
            require(doc_id == record["_id"] and doc_id not in seen_ids, f"{method}: ID collision/mismatch")
            seen_ids.add(doc_id)
            require(record["method"] == method, f"{doc_id}: wrong method")
            query_id = record.get("carrier_reference_query_id") if method == "hub" else record["target_query_id"]
            require(query_id == query_ids[index], f"{doc_id}: missing/duplicate/out-of-order query association")
            expected_id = f"adv-hub-{index:04d}" if method == "hub" else f"adv-{method}-{query_id}"
            require(doc_id == expected_id, f"{doc_id}: incorrect ID format")
            if method == "hub":
                require(record["target_query_id"] is None, f"{doc_id}: hub must not have a target")
            if method == "topic":
                require(record["topic"] == attack.topic_of(query_id), f"{doc_id}: wrong topic")
            carrier_id = record["carrier_doc_id"]
            require(carrier_id in docs, f"{doc_id}: missing carrier")
            require(carrier_id != query_id and carrier_id not in relevant[query_id], f"{doc_id}: excluded carrier")
            require(attack.topic_of(carrier_id) != attack.topic_of(query_id), f"{doc_id}: same-topic carrier")
            words = record["injected_keywords"]
            require(len(words) == len(set(words)) == record["n_keywords"] > 0, f"{doc_id}: invalid keyword count")
            require(all(attack.tokenize(w) == [w] for w in words), f"{doc_id}: invalid keyword")
            requested = record["requested_keywords"]
            if method == "targeted":
                query_words = set(attack.tokenize(query_by_id[query_id]["text"]))
                require(set(words).issubset(query_words), f"{doc_id}: non-query targeted keyword")
                require(len(words) == min(requested, len(query_words)), f"{doc_id}: targeted shortfall")
            else:
                require(len(words) == requested, f"{doc_id}: keyword shortfall")
            block = " ".join(words)
            require(row["title"] == docs[carrier_id]["title"], f"{doc_id}: carrier title changed")
            require(row["text"] == block + " " + docs[carrier_id]["text"], f"{doc_id}: incorrect text layout")
            if tokenizer is not None:
                prefix = (row["title"] + " " + block).strip()
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                max_content_wordpieces = max(max_content_wordpieces, len(prefix_ids))
                # Existing ColBERT tokenize reserves one marker slot before a
                # tokenizer call at max_length=179; BERT alone has one extra slot.
                require(len(prefix_ids) + 3 <= 180,
                        f"{doc_id}: title + keywords need {len(prefix_ids) + 3} tokens including special/marker slots")
                full_text = (row["title"] + " " + row["text"]).strip()
                truncated = tokenizer.encode(full_text, truncation=True, max_length=179)
                require(truncated[1:1 + len(prefix_ids)] == prefix_ids, f"{doc_id}: keyword prefix truncated")
            keyword_counts.append(len(words))
        deterministic = "not requested"
        if check_determinism:
            seeds = {r["seed"] for r in records}
            requested_counts = {r["requested_keywords"] for r in records}
            require(len(seeds) == len(requested_counts) == 1, f"{method}: inconsistent parameters")
            rebuilt_rows, rebuilt_records = attack.build_corpora(
                corpus, queries, relevant, [method], requested_counts.pop(), seeds.pop())[method]
            require(rebuilt_records == records, f"{method}: rebuilt manifests differ")
            with tempfile.TemporaryDirectory(prefix="verify-attack-") as temp:
                rebuilt = Path(temp) / attack.PARQUET_NAME
                pq.write_table(pa.Table.from_pylist(rebuilt_rows, schema=schema), rebuilt,
                               compression="zstd", version="2.6", row_group_size=1024)
                require(file_hash(rebuilt) == file_hash(folder / attack.PARQUET_NAME),
                        f"{method}: rebuilt parquet bytes differ")
            deterministic = "byte-identical parquet and identical manifest records"
        report[method] = {"documents": len(rows), "keyword_range": [min(keyword_counts), max(keyword_counts)],
                          "short_queries": sum(r["n_keywords"] < r["requested_keywords"] for r in records),
                          "schema_ids_carriers_layout": "passed",
                          "max_title_keyword_wordpieces": max_content_wordpieces if tokenizer is not None else None,
                          "max_tokens_including_specials_and_marker": max_content_wordpieces + 3 if tokenizer is not None else None,
                          "tokenizer": tokenizer_status, "determinism": deterministic,
                          "parquet_sha256": file_hash(folder / attack.PARQUET_NAME)}
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="arguana", choices=["arguana"])
    parser.add_argument("--methods", nargs="+", choices=attack.METHODS, default=list(attack.METHODS))
    parser.add_argument("--expected-count", type=int, default=1406)
    parser.add_argument("--check-tokenizer", action="store_true")
    parser.add_argument("--tokenizer-path", type=Path, help="Local bert-base-uncased tokenizer directory")
    parser.add_argument("--check-determinism", action="store_true")
    args = parser.parse_args()
    try:
        verify(DATASETS_DIR / args.dataset, args.methods, args.expected_count,
               args.check_tokenizer, args.tokenizer_path, args.check_determinism)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
