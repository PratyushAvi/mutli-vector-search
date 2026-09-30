"""
Encode BEIR-style datasets into multi-vector (per-token) embeddings.

For every dataset under datasets/ (e.g. datasets/arguana/{corpus,queries}/*.parquet)
this writes

    datasets/<name>/vec_corpus/<method>/<tag>.vectors.npy   float16 [total_vectors, dim]
    datasets/<name>/vec_corpus/<method>/<tag>.offsets.npy   int64   [n_items + 1]
    datasets/<name>/vec_corpus/<method>/<tag>.ids.json      item ids, row i <-> offsets[i]:offsets[i+1]
    datasets/<name>/vec_corpus/<method>/<tag>.meta.json     model, dims, vector-count stats, settings

and the same under vec_queries/. <method> is `colbert` or `bert`. <tag> looks like `colbertv2.0_d128_nv32`:
method, embedding dimension, max vectors per item (for ColBERT queries every query
has exactly nv vectors because of [MASK] query augmentation).

The vectors of item i are `vectors[offsets[i]:offsets[i+1]]` and are L2-normalised,
so a dot product is a cosine similarity. Use `load_vectors` to read them back.

Methods:
  colbert  colbert-ir/colbertv2.0: BERT + linear projection to 128-d, [Q]/[D] markers,
           queries padded with [MASK] to --query-maxlen, punctuation dropped from docs.
  bert     bert-base-uncased last hidden layer (768-d), [CLS]/[SEP] dropped.

Requires: pip install torch transformers pandas pyarrow numpy

Usage:
  python src/encode.py --method colbert
  python src/encode.py --method bert --datasets arguana scidocs --limit 100
"""

import argparse
import json
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel

from paths import DATASETS_DIR, ROOT

MODELS = {
    "colbert": "colbert-ir/colbertv2.0",
    "bert": "bert-base-uncased",
}


class ColBERT(BertPreTrainedModel):
    """Same layout as the HF_ColBERT checkpoint (`bert.*` + `linear.weight`)."""

    def __init__(self, config, dim=128):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)
        self.post_init()

    def forward(self, input_ids, attention_mask):
        return self.linear(self.bert(input_ids, attention_mask=attention_mask).last_hidden_state)


class Encoder:
    def __init__(self, method, device, query_maxlen, doc_maxlen):
        self.method = method
        self.device = device
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.model_name = MODELS[method]
        self.tok = AutoTokenizer.from_pretrained(self.model_name)

        if method == "colbert":
            self.model = ColBERT.from_pretrained(self.model_name)
            self.dim = self.model.linear.out_features
            vocab = self.tok.get_vocab()
            self.q_marker, self.d_marker = vocab["[unused0]"], vocab["[unused1]"]
            self.skiplist = {
                self.tok.encode(p, add_special_tokens=False)[0] for p in string.punctuation
            }
        else:
            self.model = BertModel.from_pretrained(self.model_name, add_pooling_layer=False)
            self.dim = self.model.config.hidden_size
        self.model.to(device).eval()

    def tag(self, kind):
        name = self.model_name.split("/")[-1]
        maxlen = self.query_maxlen if kind == "queries" else self.doc_maxlen
        return f"{name}_d{self.dim}_nv{maxlen}"

    def tokenize(self, texts, kind):
        """Returns per item: input ids, attention mask, and which positions become output vectors."""
        is_query = kind == "queries"
        maxlen = self.query_maxlen if is_query else self.doc_maxlen
        tok, cls, sep = self.tok, self.tok.cls_token_id, self.tok.sep_token_id

        if self.method == "bert":
            ids = tok(texts, truncation=True, max_length=maxlen)["input_ids"]
            attn = [[1] * len(x) for x in ids]
            keep = [[t not in (cls, sep) for t in x] for x in ids]
            return ids, attn, keep

        # ColBERT: [CLS] [Q]/[D] tokens... [SEP]; one slot is reserved for the marker.
        ids = tok(texts, truncation=True, max_length=maxlen - 1)["input_ids"]
        marker = self.q_marker if is_query else self.d_marker
        ids = [[x[0], marker] + x[1:] for x in ids]
        if is_query:
            # Query augmentation: pad with [MASK] (not attended to), keep every position.
            attn = [[1] * len(x) + [0] * (maxlen - len(x)) for x in ids]
            ids = [x + [tok.mask_token_id] * (maxlen - len(x)) for x in ids]
            keep = [[True] * maxlen for _ in ids]
        else:
            attn = [[1] * len(x) for x in ids]
            keep = [[t not in self.skiplist for t in x] for x in ids]
        return ids, attn, keep

    @torch.no_grad()
    def embed(self, ids, attn):
        """Pads a batch of token id lists and returns normalised per-token vectors [B, L, dim]."""
        L = max(len(x) for x in ids)
        pad = self.tok.pad_token_id
        input_ids = torch.tensor([x + [pad] * (L - len(x)) for x in ids], device=self.device)
        mask = torch.tensor([a + [0] * (L - len(a)) for a in attn], device=self.device)
        if self.method == "colbert":
            out = self.model(input_ids, mask)
        else:
            out = self.model(input_ids, attention_mask=mask).last_hidden_state
        return nn.functional.normalize(out.float(), p=2, dim=-1)


def load_texts(split_dir, kind):
    df = pd.read_parquet(split_dir)
    if kind == "corpus":
        texts = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    else:
        texts = df["text"].fillna("")
    return df["_id"].astype(str).tolist(), texts.tolist()


def encode_split(enc, dataset_dir, kind, batch_size, limit, dtype, overwrite):
    ids_, texts = load_texts(dataset_dir / kind, kind)
    if limit:
        ids_, texts = ids_[:limit], texts[:limit]

    out_dir = dataset_dir / f"vec_{kind}" / enc.method
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / enc.tag(kind)
    if Path(f"{prefix}.meta.json").exists() and not overwrite:
        print(f"  skip {prefix.relative_to(ROOT)} (exists, use --overwrite)")
        return

    tok_ids, attn, keep = enc.tokenize(texts, kind)
    counts = np.array([sum(k) for k in keep], dtype=np.int64)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    # Preallocate on disk so large corpora never have to fit in RAM.
    vectors = np.lib.format.open_memmap(
        f"{prefix}.vectors.npy", mode="w+", dtype=dtype, shape=(int(offsets[-1]), enc.dim)
    )

    # Length-sorted batches minimise padding.
    order = np.argsort([len(x) for x in tok_ids], kind="stable")
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        batch = order[b : b + batch_size]
        out = enc.embed([tok_ids[i] for i in batch], [attn[i] for i in batch]).cpu().numpy()
        for row, i in enumerate(batch):
            k = np.asarray(keep[i])
            vectors[offsets[i] : offsets[i + 1]] = out[row, : len(k)][k]
        done = min(b + batch_size, len(order))
        if (b // batch_size) % 50 == 0 or done == len(order):
            print(f"  {kind}: {done}/{len(order)}  ({time.time() - t0:.0f}s)", flush=True)
    vectors.flush()
    del vectors

    np.save(f"{prefix}.offsets.npy", offsets)
    with open(f"{prefix}.ids.json", "w") as f:
        json.dump(ids_, f)
    meta = {
        "dataset": dataset_dir.name,
        "split": kind,
        "method": enc.method,
        "model": enc.model_name,
        "dim": enc.dim,
        "dtype": np.dtype(dtype).name,
        "normalized": True,
        "max_len": enc.query_maxlen if kind == "queries" else enc.doc_maxlen,
        "num_items": len(ids_),
        "num_vectors": int(offsets[-1]),
        "vectors_per_item": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": round(float(counts.mean()), 2),
        },
        "query_augmentation": enc.method == "colbert" and kind == "queries",
        "punctuation_removed": enc.method == "colbert" and kind == "corpus",
        "limit": limit,
    }
    with open(f"{prefix}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {prefix.relative_to(ROOT)}.*  {meta['vectors_per_item']}")


def load_vectors(prefix, mmap=True):
    """Load `<prefix>.{vectors,offsets}.npy` etc. Returns (vectors, offsets, ids, meta)."""
    prefix = str(prefix)
    vectors = np.load(f"{prefix}.vectors.npy", mmap_mode="r" if mmap else None)
    offsets = np.load(f"{prefix}.offsets.npy")
    with open(f"{prefix}.ids.json") as f:
        ids = json.load(f)
    with open(f"{prefix}.meta.json") as f:
        meta = json.load(f)
    return vectors, offsets, ids, meta


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=MODELS, default="colbert")
    p.add_argument("--datasets", nargs="*", help="dataset names (default: all under datasets/)")
    p.add_argument("--splits", nargs="*", choices=["corpus", "queries"], default=["corpus", "queries"])
    p.add_argument("--query-maxlen", type=int, default=32, help="max (ColBERT: exact) vectors per query")
    p.add_argument("--doc-maxlen", type=int, default=180, help="max tokens per document")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0, help="only encode the first N items (smoke tests)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    names = args.datasets or sorted(d.name for d in DATASETS_DIR.iterdir() if (d / "corpus").is_dir())
    device = pick_device(args.device)
    enc = Encoder(args.method, device, args.query_maxlen, args.doc_maxlen)
    print(f"method={args.method} model={enc.model_name} dim={enc.dim} device={device}")

    for name in names:
        print(f"[{name}]")
        for kind in args.splits:
            encode_split(enc, DATASETS_DIR / name, kind, args.batch_size, args.limit, args.dtype, args.overwrite)


if __name__ == "__main__":
    main()
