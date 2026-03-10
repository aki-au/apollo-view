"""
All of the shared utilities
"""

import json
import os
import re
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

EMBEDDINGS_DIR = Path("./embeddings")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_PATH = Path("./calpers.db")

CONFIG = {
    "TOP_K_DENSE": 10,
    "TOP_K_SPARSE": 10,
    "TOP_K_FINAL": 5,
    "RRF_K": 60,
    "RECENCY_BOOST_WEIGHT": 0.15,
    "SOURCE_TYPE_BOOST": 1.15,
    "TOPIC_MATCH_BOOST": 1.10,
    "MAX_SECTION_CONTEXT_CHARS": 5000,
    "MAX_INSIGHT_CONTEXT_CHARS": 7000,
    "TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY", "tvly-dev-xx"),
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(agent: str, message: str) -> None:
    print(f"LOG: {agent}: {message}")


# ---------------------------------------------------------------------------
# LLM helpers (agents 1, 2, 6)
# ---------------------------------------------------------------------------

def ollama_generate(prompt: str, timeout: int = 180) -> str:
    """Send a prompt to the local Ollama instance and return the response."""
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


def extract_json(raw: str):
    """Best-effort extraction of a JSON object or array from raw LLM output."""
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip().rstrip("`").strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Text helpers (agents 2, 3, 4, 5)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", text.lower())


def parse_csv_field(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Embedding / ChromaDB helpers (agents 3 and 4)
# ---------------------------------------------------------------------------

def make_st_embedding_function(model):
    """Return a ChromaDB-compatible embedding function backed by a SentenceTransformer."""
    import chromadb

    class STEmbeddingFunction(chromadb.EmbeddingFunction):
        def __init__(self, st_model):
            self._model = st_model

        def __call__(self, input):
            texts = [input] if isinstance(input, str) else list(input)
            return self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

    return STEmbeddingFunction(model)


def build_bm25_index(collection, agent_label: str = "Agent"):
    """Build a BM25Okapi index over the full ChromaDB collection."""
    from rank_bm25 import BM25Okapi

    log(agent_label, "Building BM25 index from corpus")
    results = collection.get(include=["documents", "metadatas"])

    all_ids = results.get("ids", [])
    all_texts = results.get("documents", [])
    all_metadatas = results.get("metadatas", [])

    if not all_ids:
        log(agent_label, "BM25 build skipped because corpus is empty")
        return None, [], [], []

    tokenized = [tokenize(text) for text in all_texts]
    bm25 = BM25Okapi(tokenized)

    log(agent_label, f"BM25 index built chunks={len(all_ids)}")
    return bm25, all_ids, all_texts, all_metadatas


def dense_search(collection, query: str, top_k: int, plan_name: str | None = None) -> list:
    """Dense (embedding) search against a ChromaDB collection."""
    query_k = max(top_k * 3, top_k)
    results = collection.query(
        query_texts=[query],
        n_results=query_k,
        include=["documents", "metadatas", "distances"],
    )

    if plan_name:
        results = _filter_dense_results_by_plan(results, plan_name)

    chunks = []
    for chunk_id, text, meta, dist in zip(
        results["ids"][0][:top_k],
        results["documents"][0][:top_k],
        results["metadatas"][0][:top_k],
        results["distances"][0][:top_k],
    ):
        chunks.append((chunk_id, text, meta, dist))
    return chunks


def _filter_dense_results_by_plan(results, plan_name: str):
    """Filter dense search results to only include chunks matching the plan."""
    filtered = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    target = normalize_text(plan_name)

    for chunk_id, doc, meta, dist in zip(
        results.get("ids", [[]])[0],
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        meta_plan = normalize_text(meta.get("pension_plan", ""))
        collection_name = normalize_text(meta.get("collection", ""))
        if meta_plan == target or collection_name == "calpers_docs":
            filtered["ids"][0].append(chunk_id)
            filtered["documents"][0].append(doc)
            filtered["metadatas"][0].append(meta)
            filtered["distances"][0].append(dist)

    return filtered


def sparse_search(
    bm25,
    all_ids: list,
    all_texts: list,
    all_metadatas: list,
    query: str,
    top_k: int,
    plan_name: str | None = None,
) -> list:
    """BM25 sparse search, optionally filtering by plan name."""
    if bm25 is None:
        return []

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    candidate_indices = []
    target = normalize_text(plan_name) if plan_name else None

    for idx, score in enumerate(scores):
        if score <= 0:
            continue
        if target:
            meta = all_metadatas[idx]
            meta_plan = normalize_text(meta.get("pension_plan", ""))
            collection_name = normalize_text(meta.get("collection", ""))
            if not (meta_plan == target or collection_name == "calpers_docs"):
                continue
        candidate_indices.append(idx)

    top_indices = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        (all_ids[i], all_texts[i], all_metadatas[i], float(scores[i]))
        for i in top_indices
    ]


def reciprocal_rank_fusion(
    dense_chunks: list,
    sparse_chunks: list,
    top_k: int,
    k: int = 60,
    pool_multiplier: int = 3,
) -> list:
    """Merge dense and sparse result lists via Reciprocal Rank Fusion."""
    from collections import defaultdict

    rrf_scores: dict[str, float] = defaultdict(float)
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}
    dense_distances: dict[str, float] = {}
    sparse_scores: dict[str, float] = {}

    for rank, (chunk_id, _, _, distance) in enumerate(dense_chunks, start=1):
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        dense_ranks[chunk_id] = rank
        dense_distances[chunk_id] = distance

    for rank, (chunk_id, _, _, bm25_score) in enumerate(sparse_chunks, start=1):
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        sparse_ranks[chunk_id] = rank
        sparse_scores[chunk_id] = bm25_score

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        (
            chunk_id,
            score,
            dense_ranks.get(chunk_id),
            sparse_ranks.get(chunk_id),
            dense_distances.get(chunk_id),
            sparse_scores.get(chunk_id),
        )
        for chunk_id, score in ranked[: max(top_k * pool_multiplier, top_k)]
    ]


def dedupe_by_id(chunks: list) -> list:
    """Remove duplicate chunks by chunk_id attribute, preserving order."""
    seen: set[str] = set()
    deduped = []
    for chunk in chunks:
        cid = chunk.chunk_id
        if cid not in seen:
            seen.add(cid)
            deduped.append(chunk)
    return deduped