"""Small fixtures for attack semantics, carrier exclusions and reproducibility."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import make_attack_corpus as attack
import verify_attack_corpus as verifier


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "arguana"
    for part in ("corpus", "queries", "qrels"):
        (root / part).mkdir(parents=True)
    schema = pa.schema([(n, pa.large_string()) for n in ("_id", "title", "text")],
                       metadata={b"fixture": b"preserve me"})
    rows = [
        {"_id": "test-alpha-q", "title": "", "text": "apple apple common common cherry"},
        {"_id": "test-beta-q", "title": "beta", "text": "banana common date"},
        {"_id": "test-gamma-positive", "title": "", "text": "common elephant"},
        {"_id": "test-delta-safe", "title": "delta", "text": "common fig grape"},
        {"_id": "test-beta-alt", "title": "", "text": "banana date common"},
    ]
    # Deliberately shuffled across shards.
    pq.write_table(pa.Table.from_pylist(rows[:2], schema=schema), root / "corpus" / "b.parquet")
    pq.write_table(pa.Table.from_pylist(rows[2:], schema=schema), root / "corpus" / "a.parquet")
    queries = [{"_id": "test-beta-q", "title": "", "text": "banana"},
               {"_id": "test-alpha-q", "title": "", "text": "apple apple cherry unseen"}]
    pq.write_table(pa.Table.from_pylist(queries, schema=schema), root / "queries" / "q.parquet")
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "test-alpha-q\ttest-gamma-positive\t1\n"
        "test-alpha-q\ttest-beta-alt\t2\n"
        "test-beta-q\ttest-gamma-positive\t1\n"
        "test-alpha-q\ttest-delta-safe\t0\n")
    return root


def quiet_generate(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return attack.generate(*args, **kwargs)


def hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file()}


def test_cli_defaults_and_tokenization():
    args = attack.parser().parse_args([])
    assert (args.dataset, args.methods, args.keywords, args.seed, args.overwrite) == (
        "arguana", list(attack.METHODS), 60, 0, False)
    assert attack.tokenize("The CAT's well-being, don't AI I 123 abc!") == ["cat's", "well-being", "abc"]


def test_methods_and_verification(dataset):
    output = quiet_generate(dataset, keywords=2)
    report = verifier.verify(dataset, expected_count=2, check_determinism=True)
    assert all(v["documents"] == 2 for v in report.values())
    assert report["targeted"]["short_queries"] == 1
    records = [json.loads(x) for x in (output / "attack_corpus_targeted/manifest.jsonl").read_text().splitlines()]
    # Raw TF wins for apple; unseen query vocabulary receives smoothed df=0 IDF.
    assert records[0]["injected_keywords"] == ["apple", "unseen"]
    assert records[1]["injected_keywords"] == ["banana"]
    assert "Short queries use all" in (output / "README.md").read_text()


def test_all_qrels_and_topic_exclusions(dataset):
    corpus, queries, _, relevant = attack.load_inputs(dataset)
    for seed in range(12):
        results = attack.build_corpora(corpus, queries, relevant, keywords=2, seed=seed)
        for _, records in results.values():
            for record in records:
                qid = record["target_query_id"] or record["carrier_reference_query_id"]
                carrier = record["carrier_doc_id"]
                assert carrier not in relevant[qid] | {qid}
                assert attack.topic_of(carrier) != attack.topic_of(qid)


def test_topic_pool_uses_aggregate_tf_idf_and_hub_uses_df(dataset, monkeypatch):
    corpus, queries, _, relevant = attack.load_inputs(dataset)
    seen = {}

    class FirstTerms:
        def __init__(self, key):
            self.key = key

        def choice(self, pool, size=None, replace=True):
            if size is None:
                return pool[0]
            seen[self.key] = list(pool)
            return attack.np.array(pool[:size])

    monkeypatch.setattr(attack, "rng_for", lambda seed, method, doc_id, purpose: FirstTerms((method, doc_id, purpose)))
    attack.build_corpora(corpus, queries, relevant, keywords=2)
    assert seen[("hub", "adv-hub-0000", "keywords")][:3] == ["common", "banana", "date"]
    assert seen[("topic", "adv-topic-test-alpha-q", "keywords")] == ["apple", "cherry", "common"]


def test_method_order_and_query_knowledge_boundaries(dataset):
    corpus, queries, _, relevant = attack.load_inputs(dataset)
    combined = attack.build_corpora(corpus, queries, relevant, keywords=2)
    reordered = attack.build_corpora(corpus.iloc[::-1], queries.iloc[::-1], relevant,
                                   methods=["hub", "topic", "targeted"], keywords=2)
    assert combined == reordered
    altered = queries.copy()
    altered["text"] = "entirely unrelated replacement"
    for method in ("topic", "hub"):
        assert attack.build_corpora(corpus, altered, relevant, methods=[method], keywords=2)[method] == combined[method]
    assert attack.build_corpora(corpus, queries, relevant, keywords=2, seed=7)["hub"] != combined["hub"]


def test_overwrite_subset_and_failure_preserve_outputs(dataset):
    output = quiet_generate(dataset, keywords=2)
    before = hashes(output)
    with pytest.raises(FileExistsError):
        quiet_generate(dataset, keywords=2)
    with pytest.raises(ValueError, match="pool words"):
        quiet_generate(dataset, keywords=999, overwrite=True)
    assert hashes(output) == before
    quiet_generate(dataset, methods=["targeted"], keywords=1, seed=5, overwrite=True)
    after = hashes(output)
    assert all(after[name] == digest for name, digest in before.items() if "attack_corpus_hub/" in name or "attack_corpus_topic/" in name)
    readme = (output / "README.md").read_text()
    assert "| targeted | 2 | 5 | 1 |" in readme
    assert "| hub | 2 | 0 | 2 |" in readme


def test_independent_cli_processes_are_byte_identical(dataset):
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import make_attack_corpus as a; "
            "a.DATASETS_DIR=a.Path(sys.argv[2]); "
            "sys.argv=['make_attack_corpus.py','--keywords','2','--overwrite']; a.main(); "
            "assert 'torch' not in sys.modules and 'encode' not in sys.modules")
    for hash_seed in ("1", "99"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, "-B", "-c", code, str(SRC), str(dataset.parent)],
                       env=env, check=True, capture_output=True, text=True)
        current = hashes(dataset / "corpus_modified")
        if hash_seed == "1":
            first = current
        else:
            assert current == first


def test_missing_qrels_and_carrier_fail_closed(dataset):
    corpus, queries, _, relevant = attack.load_inputs(dataset)
    relevant[queries.iloc[0]["_id"]] = set(corpus["_id"])
    with pytest.raises(ValueError, match="No eligible carrier"):
        attack.build_corpora(corpus, queries, relevant, keywords=2)
    (dataset / "qrels/test.tsv").unlink()
    with pytest.raises(FileNotFoundError, match="Missing"):
        quiet_generate(dataset, keywords=2)


def test_verifier_detects_corruption(dataset):
    output = quiet_generate(dataset, keywords=2)
    path = output / "attack_corpus_hub/manifest.jsonl"
    records = [json.loads(x) for x in path.read_text().splitlines()]
    records[0]["carrier_doc_id"] = "test-gamma-positive"
    path.write_text("\n".join(map(json.dumps, records)) + "\n")
    with pytest.raises(ValueError, match="excluded carrier"):
        verifier.verify(dataset, methods=["hub"], expected_count=2)
