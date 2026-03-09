import re
from pathlib import Path
from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


EMBEDDINGS_DIR = Path("./embeddings")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "calpers_docs"

TOP_K_DENSE = 10
TOP_K_SPARSE = 10
TOP_K_FINAL = 5
RRF_K = 60

RECENCY_BOOST_WEIGHT = 0.15
SOURCE_TYPE_BOOST = 1.15
TOPIC_MATCH_BOOST = 1.10


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


def detect_preferred_source_types(question: str) -> set[str]:
    q = normalize_text(question)
    preferred = set()

    if any(x in q for x in ["board", "committee", "meeting", "transcript", "minutes"]):
        preferred.add("board_minutes")
    if "policy" in q:
        preferred.add("investment_policy")
    if "rfp" in q or "request for proposal" in q:
        preferred.add("rfp")
    if "annual report" in q:
        preferred.update({"annual_investment_report", "acfr"})
    if any(x in q for x in ["consultant", "consultants", "meketa", "wilshire"]):
        preferred.update({"board_minutes", "annual_investment_report"})
    if any(x in q for x in ["esg", "climate", "stewardship", "sustainability"]):
        preferred.update({"board_minutes", "rfp", "investment_policy"})
    if any(x in q for x in ["most recent", "recently", "latest"]):
        preferred.add("board_minutes")

    return preferred


def detect_question_topics(question: str) -> set[str]:
    q = normalize_text(question)
    topic_aliases = {
        "fixed_income": ["fixed income", "bond", "aggregate", "core fixed", "global fixed"],
        "duration_risk": ["duration", "duration risk", "interest rate risk", "rate sensitivity"],
        "esg": ["esg", "climate", "sustainability", "stewardship", "responsible investment"],
        "fees": ["fee", "fees", "bps", "basis points", "pricing", "cost"],
        "manager_review": ["under review", "probation", "watchlist", "manager review", "performance review"],
        "macro": ["inflation", "fed", "federal reserve", "recession", "yield curve", "macro"],
        "funded_status": ["funded status", "funding ratio", "liability", "actuarial"],
        "rfp": ["rfp", "request for proposal", "finalist", "search"],
    }

    found = set()
    for topic, aliases in topic_aliases.items():
        if any(alias in q for alias in aliases):
            found.add(topic)
    return found


class RetrievedChunk:
    def __init__(
        self,
        chunk_id,
        text,
        metadata,
        rrf_score,
        final_score,
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
        self.dense_rank = dense_rank
        self.sparse_rank = sparse_rank
        self.dense_distance = dense_distance
        self.sparse_score = sparse_score

    def __repr__(self):
        source = self.metadata.get("source_file", "unknown")
        date = self.metadata.get("date", "unknown")
        tone = self.metadata.get("tone", "unknown")
        return (
            f"RetrievedChunk(source={source!r}, date={date!r}, tone={tone!r}, "
            f"rrf={self.rrf_score:.4f}, final={self.final_score:.4f})"
        )


class TopicShiftSignal:
    def __init__(self, topic, chunks):
        self.topic = topic
        self.chunks = chunks

    def __repr__(self):
        dates = [c.metadata.get("date", "?") for c in self.chunks]
        tones = [c.metadata.get("tone", "?") for c in self.chunks]
        return f"TopicShiftSignal(topic={self.topic!r}, dates={dates}, tones={tones})"

    def to_display_string(self):
        lines = [f"TOPIC SHIFT SIGNAL: {self.topic.upper().replace('_', ' ')}"]
        for chunk in self.chunks:
            date = chunk.metadata.get("date", "unknown")
            source = chunk.metadata.get("source_file", "unknown")
            tone = chunk.metadata.get("tone", "unknown")
            weight = chunk.metadata.get("recency_weight", "?")
            preview = chunk.text[:200].replace("\n", " ")
            lines.append(
                f"  [{date}] {source} | tone={tone} | recency={weight} | final_score={chunk.final_score:.4f}"
            )
            lines.append(f"  \"{preview}...\"")
        lines.append("  Recommendation: Board tone on this topic may have shifted over time.")
        return "\n".join(lines)


class QuestionResult:
    def __init__(self, question, chunks):
        self.question = question
        self.chunks = chunks

    def __repr__(self):
        return f"QuestionResult(question={self.question[:50]!r}, chunks={len(self.chunks)})"


class RAGResult:
    def __init__(self, plan_name, question_results, topic_shift_signals):
        self.plan_name = plan_name
        self.question_results = question_results
        self.topic_shift_signals = topic_shift_signals

    def __repr__(self):
        total_chunks = sum(len(r.chunks) for r in self.question_results)
        return (
            f"RAGResult(plan={self.plan_name!r}, questions={len(self.question_results)}, "
            f"total_chunks={total_chunks}, topic_shifts={len(self.topic_shift_signals)})"
        )

    def to_context_string(self, max_chunks_per_question: int = 4, preview_chars: int = 300):
        lines = [f"=== DOCUMENT RAG: {self.plan_name} ===\n"]

        if self.topic_shift_signals:
            lines.append("TOPIC SHIFT SIGNALS")
            for signal in self.topic_shift_signals:
                lines.append(signal.to_display_string())
                lines.append("")

        lines.append("RETRIEVED DOCUMENTS")
        for qr in self.question_results:
            lines.append(f"\nQuestion: {qr.question}")
            if not qr.chunks:
                lines.append("  No relevant documents found.")
                continue

            chunks_to_show = qr.chunks[:max_chunks_per_question]
            for i, chunk in enumerate(chunks_to_show, 1):
                meta = chunk.metadata
                source = meta.get("source_file", "unknown")
                date = meta.get("date", "unknown")
                tone = meta.get("tone", "neutral")
                topics = meta.get("topics", "")
                weight = meta.get("recency_weight", "?")
                source_type = meta.get("source_type", "unknown")
                page = meta.get("page_number", "?")

                lines.append(
                    f"  [{i}] {source} (p{page}) | source_type={source_type} | {date} | "
                    f"tone={tone} | weight={weight} | final={chunk.final_score:.4f} | rrf={chunk.rrf_score:.4f}"
                )
                lines.append(f"      topics: {topics}")
                lines.append(f"      {chunk.text[:preview_chars].replace(chr(10), ' ')}...")
                lines.append("")

            if len(qr.chunks) > max_chunks_per_question:
                lines.append(f"  ... {len(qr.chunks) - max_chunks_per_question} more chunks omitted")

        return "\n".join(lines)


def build_bm25_index(collection):
    log("Agent 3", "Building BM25 index from ChromaDB corpus")
    results = collection.get(include=["documents", "metadatas"])

    all_ids = results.get("ids", [])
    all_texts = results.get("documents", [])
    all_metadatas = results.get("metadatas", [])

    if not all_ids:
        log("Agent 3", "BM25 build skipped because corpus is empty")
        return None, [], [], []

    tokenized = [tokenize(text) for text in all_texts]
    bm25 = BM25Okapi(tokenized)

    log("Agent 3", f"BM25 index built chunks={len(all_ids)}")
    return bm25, all_ids, all_texts, all_metadatas


def filter_dense_results_by_plan(results, plan_name: str):
    if not plan_name:
        return results

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


def dense_search(collection, query: str, top_k: int, plan_name: str = None) -> list:
    query_k = max(top_k * 3, top_k)
    results = collection.query(
        query_texts=[query],
        n_results=query_k,
        include=["documents", "metadatas", "distances"],
    )

    results = filter_dense_results_by_plan(results, plan_name)

    chunks = []
    for chunk_id, text, meta, dist in zip(
        results["ids"][0][:top_k],
        results["documents"][0][:top_k],
        results["metadatas"][0][:top_k],
        results["distances"][0][:top_k],
    ):
        chunks.append((chunk_id, text, meta, dist))

    return chunks


def sparse_search(bm25, all_ids, all_texts, all_metadatas, query: str, top_k: int, plan_name: str = None) -> list:
    if bm25 is None:
        return []

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    candidate_indices = []
    target = normalize_text(plan_name) if plan_name else None

    for idx, score in enumerate(scores):
        if score <= 0:
            continue
        meta = all_metadatas[idx]
        meta_plan = normalize_text(meta.get("pension_plan", ""))
        collection_name = normalize_text(meta.get("collection", ""))
        if target and not (meta_plan == target or collection_name == "calpers_docs"):
            continue
        candidate_indices.append(idx)

    top_indices = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)[:top_k]

    chunks = []
    for idx in top_indices:
        chunks.append((all_ids[idx], all_texts[idx], all_metadatas[idx], float(scores[idx])))

    return chunks


def reciprocal_rank_fusion(dense_chunks: list, sparse_chunks: list, top_k: int, k: int = RRF_K) -> list:
    rrf_scores = defaultdict(float)
    dense_ranks = {}
    sparse_ranks = {}
    dense_distances = {}
    sparse_scores = {}

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
        for chunk_id, score in ranked[: max(top_k * 3, top_k)]
    ]


def rerank_fused_chunks(fused_chunks, chunk_lookup, question: str, top_k: int):
    preferred_source_types = detect_preferred_source_types(question)
    question_topics = detect_question_topics(question)

    reranked = []
    for chunk_id, rrf_score, dense_rank, sparse_rank, dense_distance, sparse_score in fused_chunks:
        if chunk_id not in chunk_lookup:
            continue

        text, meta = chunk_lookup[chunk_id]
        final_score = rrf_score

        recency = safe_float(meta.get("recency_weight", 0.5), default=0.5)
        final_score *= (1 + RECENCY_BOOST_WEIGHT * recency)

        source_type = meta.get("source_type", "")
        if preferred_source_types and source_type in preferred_source_types:
            final_score *= SOURCE_TYPE_BOOST

        chunk_topics = set(parse_csv_field(meta.get("topics", "")))
        if question_topics and chunk_topics.intersection(question_topics):
            final_score *= TOPIC_MATCH_BOOST

        reranked.append(
            (
                chunk_id,
                final_score,
                rrf_score,
                dense_rank,
                sparse_rank,
                dense_distance,
                sparse_score,
            )
        )

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


def dedupe_chunks_by_id(chunks: list) -> list:
    seen = set()
    deduped = []
    for chunk in chunks:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            deduped.append(chunk)
    return deduped


def pick_representative_chunk(chunks: list, prefer_tone: str):
    candidates = [c for c in chunks if c.metadata.get("tone") == prefer_tone]
    if not candidates:
        return None
    candidates.sort(
        key=lambda c: (c.metadata.get("date", ""), c.final_score),
        reverse=True,
    )
    return candidates[0]


def detect_topic_shifts(all_chunks: list) -> list:
    deduped_chunks = dedupe_chunks_by_id(all_chunks)
    topic_chunks = defaultdict(list)

    for chunk in deduped_chunks:
        topics = parse_csv_field(chunk.metadata.get("topics", ""))
        tone = chunk.metadata.get("tone", "neutral")
        date = chunk.metadata.get("date", "")

        if tone == "neutral" or not topics or not date:
            continue

        for topic in topics:
            if topic and topic != "general":
                topic_chunks[topic].append(chunk)

    signals = []

    for topic, chunks in topic_chunks.items():
        if len(chunks) < 2:
            continue

        concern_chunks = [c for c in chunks if c.metadata.get("tone") == "concern"]
        positive_chunks = [c for c in chunks if c.metadata.get("tone") == "positive"]

        if not concern_chunks or not positive_chunks:
            continue

        concern_dates = {c.metadata.get("date") for c in concern_chunks}
        positive_dates = {c.metadata.get("date") for c in positive_chunks}

        if concern_dates == positive_dates:
            continue

        best_positive = pick_representative_chunk(chunks, "positive")
        best_concern = pick_representative_chunk(chunks, "concern")

        if best_positive and best_concern:
            signal_chunks = sorted(
                [best_positive, best_concern],
                key=lambda c: c.metadata.get("date", ""),
            )
            signals.append(TopicShiftSignal(topic=topic, chunks=signal_chunks))

    high_value_topics = {
        "duration_risk",
        "esg",
        "fees",
        "fixed_income",
        "manager_review",
        "funded_status",
        "macro",
    }

    priority_signals = [s for s in signals if s.topic in high_value_topics]
    other_signals = [s for s in signals if s.topic not in high_value_topics]

    priority_signals.sort(key=lambda s: s.topic)
    other_signals.sort(key=lambda s: s.topic)
    return priority_signals + other_signals


class DocumentRAGAgent:
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

        log("Agent 3", f"Loading embedding model={EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        log("Agent 3", f"Connecting to ChromaDB path={embeddings_dir}")
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

        log("Agent 3", f"Ready indexed_chunks={self.collection.count()}")

    def retrieve(self, question: str, plan_name: str = None) -> list:
        dense_chunks = dense_search(self.collection, question, self.top_k_dense, plan_name=plan_name)
        sparse_chunks = sparse_search(
            self.bm25,
            self.all_ids,
            self.all_texts,
            self.all_metadatas,
            question,
            self.top_k_sparse,
            plan_name=plan_name,
        )

        fused = reciprocal_rank_fusion(dense_chunks, sparse_chunks, self.top_k_final)
        reranked = rerank_fused_chunks(fused, self.chunk_lookup, question, self.top_k_final)

        retrieved = []
        for chunk_id, final_score, rrf_score, dense_rank, sparse_rank, dense_distance, sparse_score in reranked:
            if chunk_id in self.chunk_lookup:
                text, meta = self.chunk_lookup[chunk_id]
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        text=text,
                        metadata=meta,
                        rrf_score=rrf_score,
                        final_score=final_score,
                        dense_rank=dense_rank,
                        sparse_rank=sparse_rank,
                        dense_distance=dense_distance,
                        sparse_score=sparse_score,
                    )
                )

        return retrieved

    def run(self, questions: list, plan_name: str) -> RAGResult:
        log("Agent 3", f"Running document_rag questions={len(questions)} plan={plan_name}")

        question_results = []
        all_chunks = []

        for i, question in enumerate(questions, 1):
            log("Agent 3", f"Question {i}/{len(questions)}: {question[:100]}")
            chunks = self.retrieve(question, plan_name=plan_name)
            question_results.append(QuestionResult(question=question, chunks=chunks))
            all_chunks.extend(chunks)
            log("Agent 3", f"Retrieved chunks={len(chunks)}")

        all_chunks = dedupe_chunks_by_id(all_chunks)

        log("Agent 3", f"Running topic_shift_detector total_unique_chunks={len(all_chunks)}")
        topic_shift_signals = detect_topic_shifts(all_chunks)
        log("Agent 3", f"Topic shift signals found={len(topic_shift_signals)}")

        for signal in topic_shift_signals:
            log("Agent 3", f"Topic shift detected={signal}")

        result = RAGResult(
            plan_name=plan_name,
            question_results=question_results,
            topic_shift_signals=topic_shift_signals,
        )

        log("Agent 3", f"Complete result={result}")
        return result


if __name__ == "__main__":
    test_questions = [
        "What has the CalPERS board said about fixed income recently?",
        "What concerns has the board raised about duration risk?",
        "What are CalPERS ESG requirements for fixed income managers?",
        "Has CalPERS issued any RFPs for fixed income?",
        "What is the CalPERS investment policy for fixed income?",
        "What did the most recent board meeting discuss about fixed income?",
    ]

    agent = DocumentRAGAgent()
    result = agent.run(questions=test_questions, plan_name="CalPERS")

    print(result)
    print()
    print(result.to_context_string(max_chunks_per_question=3, preview_chars=220)[:3000])