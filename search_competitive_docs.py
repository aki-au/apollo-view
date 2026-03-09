import re
from pathlib import Path
from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


EMBEDDINGS_DIR = Path("./embeddings")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "market_intel"

TOP_K_DENSE = 12
TOP_K_SPARSE = 12
TOP_K_FINAL = 6
RRF_K = 60

RECENCY_BOOST_WEIGHT = 0.30
TOPIC_MATCH_BOOST = 1.10
STRATEGY_MATCH_BOOST = 1.12
SOURCE_TYPE_BOOST = 1.08
DIVERSITY_PENALTY = 0.92


def log(agent, message):
    print(f"LOG: {agent}: {message}")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", text.lower())


def parse_csv_field(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def canonicalize_strategy(strategy: str) -> str:
    s = normalize_text(strategy)

    aliases = {
        "core fixed income": [
            "core fixed income", "core bond", "aggregate", "agg",
            "us aggregate", "investment grade core"
        ],
        "global fixed income": [
            "global fixed income", "global bond", "global bonds", "global aggregate"
        ],
        "high yield": [
            "high yield", "junk bond", "below investment grade", "credit spread"
        ],
        "emerging market debt": [
            "emerging market debt", "em debt", "emd", "emerging debt", "embi"
        ],
        "short duration": [
            "short duration", "short-term bond", "short term bond", "1-3 year"
        ],
        "inflation-linked": [
            "inflation-linked", "inflation linked", "tips", "linkers"
        ],
    }

    for canonical, values in aliases.items():
        if any(v in s for v in values):
            return canonical
    return strategy


def detect_question_topics(question: str) -> set[str]:
    q = normalize_text(question)
    topic_aliases = {
        "fixed_income": [
            "fixed income", "bond", "aggregate", "core fixed", "global fixed"
        ],
        "duration_risk": [
            "duration", "duration risk", "interest rate risk", "rate sensitivity"
        ],
        "macro": [
            "inflation", "fed", "federal reserve", "recession", "yield curve",
            "macro", "gdp", "monetary policy", "central bank", "policy rate"
        ],
        "esg": [
            "esg", "climate", "sustainability", "stewardship"
        ],
        "high_yield": [
            "high yield", "credit spread", "junk bond"
        ],
        "emerging_markets": [
            "emerging market", "embi", "sovereign debt", "em debt"
        ],
        "liquidity": [
            "liquidity", "liquidity stress", "market functioning"
        ],
        "policy": [
            "policy", "central bank", "monetary policy", "rate cuts", "rate hikes"
        ],
    }

    found = set()
    for topic, aliases in topic_aliases.items():
        if any(alias in q for alias in aliases):
            found.add(topic)
    return found


def detect_time_horizon(question: str) -> str:
    q = normalize_text(question)

    if any(x in q for x in ["next 12 months", "next 6-12 months", "next year", "over the next year"]):
        return "forward_12m"
    if any(x in q for x in ["next 6 months", "near term", "short term"]):
        return "forward_6m"
    if any(x in q for x in ["right now", "currently", "today", "current"]):
        return "current"
    if any(x in q for x in ["long term", "longer term", "structural"]):
        return "long_term"
    return "general"


def detect_preferred_source_types(question: str) -> set[str]:
    q = normalize_text(question)
    preferred = set()

    if any(x in q for x in ["outlook", "macro", "inflation", "fed", "central bank", "gdp"]):
        preferred.add("market_research")
    if any(x in q for x in ["credit spread", "high yield", "em debt", "sovereign debt"]):
        preferred.add("market_research")

    return preferred


def expand_query(question: str, strategy: str = None) -> str:
    q = normalize_text(question)
    additions = []

    if "fixed income" in q or "bond" in q:
        additions.extend(["rates", "yields", "duration"])
    if "inflation" in q:
        additions.extend(["cpi", "disinflation", "policy rate"])
    if "fed" in q or "central bank" in q or "monetary policy" in q:
        additions.extend(["rate cuts", "rate hikes", "policy path"])
    if "credit spread" in q or "high yield" in q:
        additions.extend(["spread widening", "default risk", "credit cycle"])
    if "emerging market" in q or "em debt" in q:
        additions.extend(["sovereign risk", "dollar strength", "capital flows"])
    if "duration" in q:
        additions.extend(["yield curve", "term premium", "rate volatility"])

    strategy_canonical = canonicalize_strategy(strategy or "")
    if strategy_canonical == "core fixed income":
        additions.extend(["investment grade", "aggregate index"])
    elif strategy_canonical == "global fixed income":
        additions.extend(["currency", "sovereign", "global rates"])
    elif strategy_canonical == "high yield":
        additions.extend(["credit spreads", "defaults"])
    elif strategy_canonical == "emerging market debt":
        additions.extend(["hard currency", "local currency", "sovereign"])
    elif strategy_canonical == "inflation-linked":
        additions.extend(["real yields", "breakevens", "tips"])
    elif strategy_canonical == "short duration":
        additions.extend(["front end", "cash alternatives", "short maturity"])

    additions = list(dict.fromkeys(additions))
    if additions:
        return q + " " + " ".join(additions)
    return q


class MarketChunk:
    def __init__(
        self,
        chunk_id,
        text,
        metadata,
        rrf_score,
        final_score,
        confidence_score,
        dense_rank=None,
        sparse_rank=None,
        dense_distance=None,
        sparse_score=None,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.rrf_score = rrf_score
        self.final_score = final_score
        self.confidence_score = confidence_score
        self.dense_rank = dense_rank
        self.sparse_rank = sparse_rank
        self.dense_distance = dense_distance
        self.sparse_score = sparse_score

    def __repr__(self):
        source = self.metadata.get("source_file", "unknown")
        date = self.metadata.get("date", "unknown")
        return (
            f"MarketChunk(source={source!r}, date={date!r}, "
            f"rrf={self.rrf_score:.4f}, final={self.final_score:.4f}, "
            f"confidence={self.confidence_score:.3f})"
        )


class MarketQuestionResult:
    def __init__(self, question, chunks, expanded_query=None):
        self.question = question
        self.chunks = chunks
        self.expanded_query = expanded_query

    def __repr__(self):
        return f"MarketQuestionResult(question={self.question[:50]!r}, chunks={len(self.chunks)})"


class MarketIntelResult:
    def __init__(self, strategy, question_results):
        self.strategy = strategy
        self.question_results = question_results

    def __repr__(self):
        total = sum(len(r.chunks) for r in self.question_results)
        return (
            f"MarketIntelResult(strategy={self.strategy!r}, "
            f"questions={len(self.question_results)}, total_chunks={total})"
        )

    def to_context_string(self, max_chunks_per_question: int = 3, preview_chars: int = 300) -> str:
        lines = [f"=== MARKET INTELLIGENCE: {self.strategy} ===\n"]

        for qr in self.question_results:
            lines.append(f"Question: {qr.question}")
            if qr.expanded_query and qr.expanded_query != normalize_text(qr.question):
                lines.append(f"Query expansion: {qr.expanded_query}")

            if not qr.chunks:
                lines.append("  No relevant market research found.\n")
                continue

            for i, chunk in enumerate(qr.chunks[:max_chunks_per_question], 1):
                meta = chunk.metadata
                source = meta.get("source_file", "unknown")
                date = meta.get("date", "unknown")
                topics = meta.get("topics", "")
                weight = meta.get("recency_weight", "?")
                page = meta.get("page_number", "?")
                source_type = meta.get("source_type", "unknown")

                lines.append(
                    f"  [{i}] {source} (p{page}) | source_type={source_type} | {date} | "
                    f"weight={weight} | final={chunk.final_score:.4f} | confidence={chunk.confidence_score:.3f}"
                )
                lines.append(f"      topics: {topics}")
                lines.append(f"      {chunk.text[:preview_chars].replace(chr(10), ' ')}...")
                lines.append("")

            if len(qr.chunks) > max_chunks_per_question:
                lines.append(f"  ... {len(qr.chunks) - max_chunks_per_question} more chunks omitted")

        return "\n".join(lines)


def build_bm25_index(collection):
    log("Agent 4", "Building BM25 index from market_intel corpus")
    results = collection.get(include=["documents", "metadatas"])

    all_ids = results.get("ids", [])
    all_texts = results.get("documents", [])
    all_metadatas = results.get("metadatas", [])

    if not all_ids:
        log("Agent 4", "BM25 build skipped because corpus is empty")
        return None, [], [], []

    tokenized = [tokenize(text) for text in all_texts]
    bm25 = BM25Okapi(tokenized)

    log("Agent 4", f"BM25 index built chunks={len(all_ids)}")
    return bm25, all_ids, all_texts, all_metadatas


def dense_search(collection, query: str, top_k: int) -> list:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for chunk_id, text, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append((chunk_id, text, meta, dist))
    return chunks


def sparse_search(bm25, all_ids, all_texts, all_metadatas, query: str, top_k: int) -> list:
    if bm25 is None:
        return []

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        [i for i, s in enumerate(scores) if s > 0],
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return [
        (all_ids[i], all_texts[i], all_metadatas[i], float(scores[i]))
        for i in top_indices
    ]


def reciprocal_rank_fusion(dense_chunks: list, sparse_chunks: list, top_k: int, k: int = RRF_K) -> list:
    rrf_scores = defaultdict(float)
    dense_ranks = {}
    sparse_ranks = {}
    dense_dists = {}
    sparse_scores = {}

    for rank, (chunk_id, _, _, dist) in enumerate(dense_chunks, start=1):
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        dense_ranks[chunk_id] = rank
        dense_dists[chunk_id] = dist

    for rank, (chunk_id, _, _, score) in enumerate(sparse_chunks, start=1):
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        sparse_ranks[chunk_id] = rank
        sparse_scores[chunk_id] = score

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        (
            chunk_id,
            score,
            dense_ranks.get(chunk_id),
            sparse_ranks.get(chunk_id),
            dense_dists.get(chunk_id),
            sparse_scores.get(chunk_id),
        )
        for chunk_id, score in ranked[: max(top_k * 4, top_k)]
    ]


def compute_confidence(rrf_score, dense_rank, sparse_rank, dense_distance, sparse_score):
    confidence = rrf_score

    if dense_rank is not None:
        confidence += 1.0 / (dense_rank + 5)
    if sparse_rank is not None:
        confidence += 1.0 / (sparse_rank + 5)
    if dense_distance is not None:
        confidence += max(0.0, 0.25 - min(float(dense_distance), 0.25))
    if sparse_score is not None:
        confidence += min(float(sparse_score) / 20.0, 0.25)

    return confidence


def rerank(fused_chunks: list, chunk_lookup: dict, question: str, strategy: str, top_k: int) -> list:
    question_topics = detect_question_topics(question)
    preferred_source_types = detect_preferred_source_types(question)
    canonical_strategy = canonicalize_strategy(strategy or "")

    reranked = []
    for chunk_id, rrf_score, dense_rank, sparse_rank, dense_dist, sparse_sc in fused_chunks:
        if chunk_id not in chunk_lookup:
            continue

        text, meta = chunk_lookup[chunk_id]
        final_score = rrf_score

        recency = safe_float(meta.get("recency_weight", 0.5), default=0.5)
        final_score *= (1 + RECENCY_BOOST_WEIGHT * recency)

        chunk_topics = set(parse_csv_field(meta.get("topics", "")))
        if question_topics and chunk_topics.intersection(question_topics):
            final_score *= TOPIC_MATCH_BOOST

        source_type = normalize_text(meta.get("source_type", ""))
        if preferred_source_types and source_type in preferred_source_types:
            final_score *= SOURCE_TYPE_BOOST

        strategy_text = normalize_text(text + " " + str(meta.get("topics", "")) + " " + str(meta.get("source_file", "")))
        if canonical_strategy and canonical_strategy != strategy:
            if canonical_strategy in strategy_text:
                final_score *= STRATEGY_MATCH_BOOST
        elif strategy and normalize_text(strategy) in strategy_text:
            final_score *= STRATEGY_MATCH_BOOST

        confidence_score = compute_confidence(
            rrf_score=rrf_score,
            dense_rank=dense_rank,
            sparse_rank=sparse_rank,
            dense_distance=dense_dist,
            sparse_score=sparse_sc,
        )

        reranked.append(
            (
                chunk_id,
                final_score,
                confidence_score,
                rrf_score,
                dense_rank,
                sparse_rank,
                dense_dist,
                sparse_sc,
            )
        )

    reranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return reranked[:top_k]


def diversify_chunks(chunks: list[MarketChunk], top_k: int) -> list[MarketChunk]:
    selected = []
    source_counts = defaultdict(int)

    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        adjusted_score = chunk.final_score * (DIVERSITY_PENALTY ** source_counts[source])
        candidate = MarketChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata=chunk.metadata,
            rrf_score=chunk.rrf_score,
            final_score=adjusted_score,
            confidence_score=chunk.confidence_score,
            dense_rank=chunk.dense_rank,
            sparse_rank=chunk.sparse_rank,
            dense_distance=chunk.dense_distance,
            sparse_score=chunk.sparse_score,
        )
        selected.append(candidate)
        source_counts[source] += 1

    selected.sort(key=lambda c: (c.final_score, c.confidence_score), reverse=True)

    final = []
    seen_ids = set()
    for chunk in selected:
        if chunk.chunk_id not in seen_ids:
            seen_ids.add(chunk.chunk_id)
            final.append(chunk)
        if len(final) >= top_k:
            break

    return final


class MarketIntelAgent:
    def __init__(
        self,
        embeddings_dir: Path = EMBEDDINGS_DIR,
        collection_name: str = COLLECTION_NAME,
        top_k_dense: int = TOP_K_DENSE,
        top_k_sparse: int = TOP_K_SPARSE,
        top_k_final: int = TOP_K_FINAL,
    ):
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_final = top_k_final

        log("Agent 4", f"Loading embedding model={EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        log("Agent 4", f"Connecting to ChromaDB path={embeddings_dir}")
        self.client = chromadb.PersistentClient(path=str(embeddings_dir))

        class STEmbeddingFunction(chromadb.EmbeddingFunction):
            def __init__(self, model):
                self._model = model

            def __call__(self, input):
                texts = [input] if isinstance(input, str) else list(input)
                return self._model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()

        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=STEmbeddingFunction(self.model),
        )

        if self.collection.count() == 0:
            raise ValueError(f"Collection '{collection_name}' is empty")

        self.bm25, self.all_ids, self.all_texts, self.all_metadatas = build_bm25_index(self.collection)

        self.chunk_lookup = {
            chunk_id: (text, meta)
            for chunk_id, text, meta in zip(self.all_ids, self.all_texts, self.all_metadatas)
        }

        log("Agent 4", f"Ready indexed_chunks={self.collection.count()}")

    def retrieve(self, question: str, strategy: str) -> tuple[list[MarketChunk], str]:
        expanded_query = expand_query(question, strategy=strategy)

        dense_chunks = dense_search(self.collection, expanded_query, self.top_k_dense)
        sparse_chunks = sparse_search(
            self.bm25,
            self.all_ids,
            self.all_texts,
            self.all_metadatas,
            expanded_query,
            self.top_k_sparse,
        )

        fused = reciprocal_rank_fusion(dense_chunks, sparse_chunks, self.top_k_final)
        reranked = rerank(
            fused_chunks=fused,
            chunk_lookup=self.chunk_lookup,
            question=question,
            strategy=strategy,
            top_k=max(self.top_k_final * 2, self.top_k_final),
        )

        retrieved = []
        for chunk_id, final_score, confidence_score, rrf_score, dense_rank, sparse_rank, dense_dist, sparse_sc in reranked:
            if chunk_id in self.chunk_lookup:
                text, meta = self.chunk_lookup[chunk_id]
                retrieved.append(
                    MarketChunk(
                        chunk_id=chunk_id,
                        text=text,
                        metadata=meta,
                        rrf_score=rrf_score,
                        final_score=final_score,
                        confidence_score=confidence_score,
                        dense_rank=dense_rank,
                        sparse_rank=sparse_rank,
                        dense_distance=dense_dist,
                        sparse_score=sparse_sc,
                    )
                )

        retrieved = diversify_chunks(retrieved, top_k=self.top_k_final)
        return retrieved, expanded_query

    def run(self, questions: list, strategy: str) -> MarketIntelResult:
        log("Agent 4", f"Running market_intelligence questions={len(questions)} strategy={strategy!r}")

        question_results = []

        for i, question in enumerate(questions, 1):
            log("Agent 4", f"Question {i}/{len(questions)}: {question[:100]}")
            chunks, expanded_query = self.retrieve(question, strategy=strategy)
            question_results.append(
                MarketQuestionResult(
                    question=question,
                    chunks=chunks,
                    expanded_query=expanded_query,
                )
            )
            log("Agent 4", f"Retrieved chunks={len(chunks)} expanded_query={expanded_query}")

        result = MarketIntelResult(
            strategy=strategy,
            question_results=question_results,
        )

        log("Agent 4", f"Complete result={result}")
        return result


if __name__ == "__main__":
    test_questions = [
        "What is the current macro outlook for fixed income?",
        "What are the key risks in the rate environment right now?",
        "How are central bank policies affecting fixed income markets?",
        "What is the outlook for credit spreads in the next 6-12 months?",
        "How does the current inflation environment affect core fixed income?",
    ]

    agent = MarketIntelAgent()
    result = agent.run(
        questions=test_questions,
        strategy="Core Fixed Income",
    )

    print(result)
    print()
    print(result.to_context_string(max_chunks_per_question=3, preview_chars=220)[:2500])