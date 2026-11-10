"""
Beam search where vector distance decides what is in the beam and Gemini decides what to expand next.

The beam holds the L closest nodes seen so far, as in beam.beam_search. Before each pop, Gemini ranks the
documents in the beam against the full query, and the highest-ranked unexpanded one is expanded. Gemini is
only called when the beam's contents changed since its last ranking. The final beam is returned in
Gemini's order.

The API key is read from gemini_api_key.txt at the repository root (gitignored). Rankings are cached per
(model, query, beam) in a JSONL file, so a rerun replays without API calls.
"""

import json
import time
import urllib.error
import urllib.request

import numpy as np

from paths import ROOT

KEY_FILE = ROOT / "gemini_api_key.txt"
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PROMPT = """You are ranking documents for a search query. Rank the documents from most to least relevant to \
the query. Answer with the ids of every document, each exactly once, most relevant first.

Query:
{query}

Documents:
{docs}"""
SCHEMA = {"type": "ARRAY", "items": {"type": "INTEGER"}}


def load_api_key():
    key = KEY_FILE.read_text().strip() if KEY_FILE.exists() else ""
    if not key:
        raise RuntimeError(f"put a Gemini API key in {KEY_FILE}")
    return key


class GeminiRanker:
    """rank(query_id, query, doc_ids, doc_texts) -> positions into doc_ids, most relevant first. Ids Gemini
    leaves out are appended in their given order (counted in `missing`). `usage` counts API calls and the
    input / output tokens Gemini reports (output includes thinking); pass the same dict to several rankers
    to keep one running total."""

    def __init__(self, model, cache_path, max_chars=None, retries=5, usage=None):
        self.model, self.cache_path, self.max_chars, self.retries = model, cache_path, max_chars, retries
        self.key = load_api_key()
        self.cache, self.missing = {}, 0
        self.usage = usage if usage is not None else {}
        for k in ("calls", "input tokens", "output tokens"):
            self.usage.setdefault(k, 0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                r = json.loads(line)
                if r["model"] == model:
                    self.cache[r["query"], tuple(r["docs"])] = r["ranking"]

    def rank(self, qid, query, doc_ids, doc_texts):
        cache_key = (qid, tuple(doc_ids))
        if cache_key not in self.cache:
            order = self.ask(query, doc_texts)
            self.cache[cache_key] = [doc_ids[i] for i in order]
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"model": self.model, "query": qid, "docs": list(doc_ids),
                                    "ranking": self.cache[cache_key]}) + "\n")
        position = {d: i for i, d in enumerate(doc_ids)}
        return [position[d] for d in self.cache[cache_key]]

    def ask(self, query, texts):
        n = len(texts)
        docs = "\n\n".join(f"[{i}] {t[:self.max_chars]}" for i, t in enumerate(texts))
        body = {"contents": [{"parts": [{"text": PROMPT.format(query=query, docs=docs)}]}],
                "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA}}
        response = self.post(body)
        meta = response.get("usageMetadata", {})
        self.usage["input tokens"] += meta.get("promptTokenCount", 0)
        self.usage["output tokens"] += meta.get("candidatesTokenCount", 0) + meta.get("thoughtsTokenCount", 0)
        text = "".join(p.get("text", "") for p in response["candidates"][0]["content"]["parts"])
        order = list(dict.fromkeys(i for i in json.loads(text) if 0 <= i < n))
        self.missing += n - len(order)
        ranked = set(order)
        return order + [i for i in range(n) if i not in ranked]

    def post(self, body):
        request = urllib.request.Request(URL.format(model=self.model), data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=300) as r:
                    self.usage["calls"] += 1
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 503) or attempt == self.retries - 1:
                    raise RuntimeError(f"Gemini API error {e.code}: {e.read().decode()[:500]}") from e
                time.sleep(2 ** attempt)


def llm_beam_search(graph, X, q, ranker, qid, query, doc_ids, doc_texts, L, entry, on_step=None):
    """Beam search for q from `entry` with beam width L. The beam is the L closest nodes seen so far; before
    each pop, `ranker` (a GeminiRanker) orders the beam and the first unexpanded node in that order is
    expanded. doc_ids / doc_texts are indexed by graph node. on_step(step, unexpanded), if given, is called
    after each expansion with the number of expansions so far and of beam nodes still unexpanded.
    Returns (beam_ids, trace): the final beam in the ranker's order, and the trace of beam.beam_search."""
    n = len(X)
    seen = np.full(n, -1, dtype=np.int64)
    inserted = np.full(n, -1, dtype=np.int64)
    expanded = np.full(n, -1, dtype=np.int64)
    touches = np.zeros(n, dtype=np.int64)

    def ranked(beam):
        if len(beam) == 1:
            return beam
        order = ranker.rank(qid, query, [doc_ids[v] for v in beam], [doc_texts[v] for v in beam])
        return beam[order]

    seen[entry] = inserted[entry] = 0
    beam_ids = np.array([entry], dtype=np.int64)
    beam_d = np.array([((X[entry] - q) ** 2).sum()], dtype=np.float32)
    priority, ranked_beam = beam_ids, tuple(beam_ids)
    step, dists = 0, 1
    while (expanded[beam_ids] < 0).any():
        if tuple(beam_ids) != ranked_beam:
            priority, ranked_beam = ranked(beam_ids), tuple(beam_ids)
        u = priority[np.argmax(expanded[priority] < 0)]
        step += 1
        expanded[u] = step
        nbrs = graph[u]
        touches[nbrs] += 1
        new = nbrs[seen[nbrs] < 0]
        if len(new):
            seen[new] = step
            dists += len(new)
            ids = np.concatenate([beam_ids, new])
            d = np.concatenate([beam_d, ((X[new] - q) ** 2).sum(1)])
            keep = np.argsort(d, kind="stable")[:L]
            beam_ids, beam_d = ids[keep], d[keep]
            entered = beam_ids[inserted[beam_ids] < 0]
            inserted[entered] = step
        if on_step is not None:
            on_step(step, int((expanded[beam_ids] < 0).sum()))

    if tuple(beam_ids) != ranked_beam:
        priority = ranked(beam_ids)
    trace = {"seen": seen, "inserted": inserted, "expanded": expanded, "touches": touches,
             "steps": step, "dists": dists}
    return priority, trace
