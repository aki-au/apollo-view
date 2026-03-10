import re
from pathlib import Path
from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from utils import (
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL,
    CONFIG,
    log,
    normalize_text,
    tokenize,
    parse_csv_field,
    safe_float,
    make_st_embedding_function,
    build_bm25_index,
    dense_search,
    sparse_search,
    reciprocal_rank_fusion,
    dedupe_by_id,
)


COLLECTION_NAME = "calpers_docs"

TOP_K_DENSE = CONFIG["TOP_K_DENSE"]
TOP_K_SPARSE = CONFIG["TOP_K_SPARSE"]
TOP_K_FINAL = CONFIG["TOP_K_FINAL"]
RRF_K = CONFIG["RRF_K"]

RECENCY_BOOST_WEIGHT = CONFIG["RECENCY_BOOST_WEIGHT"]
SOURCE_TYPE_BOOST = CONFIG["SOURCE_TYPE_BOOST"]
TOPIC_MATCH_BOOST = CONFIG["TOPIC_MATCH_BOOST"]


# Question analysis helpers

def detect_preferred_source_types(question: str) -> set[str]:
    q = normalize_text(question)
    preferred: set[str] = set()

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

    found: set[str] = set()
    for topic, aliases in topic_aliases.items():
        if any(alias in q for alias in aliases):
            found.add(topic)
    return found


# Data classes

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


# Reranking

def rerank_fused_chunks(fused_chunks, chunk_lookup, question, top_k):
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
            (chunk_id, final_score, rrf_score, dense_rank, sparse_rank, dense_distance, sparse_score)
        )

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


def dedupe_chunks_by_id(chunks):
    seen = set()
    deduped = []
    for chunk in chunks:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            deduped.append(chunk)
    return deduped


# Topic shift detection

def pick_representative_chunk(chunks, prefer_tone):
    candidates = [c for c in chunks if c.metadata.get("tone") == prefer_tone]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c.metadata.get("date", ""), c.final_score), reverse=True)
    return candidates[0]


def detect_topic_shifts(all_chunks):
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
        "duration_risk", "esg", "fees", "fixed_income",
        "manager_review", "funded_status", "macro",
    }

    priority_signals = [s for s in signals if s.topic in high_value_topics]
    other_signals = [s for s in signals if s.topic not in high_value_topics]

    priority_signals.sort(key=lambda s: s.topic)
    other_signals.sort(key=lambda s: s.topic)
    return priority_signals + other_signals


# Agent class

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

        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=make_st_embedding_function(self.model),
        )

        if self.collection.count() == 0:
            raise ValueError(f"Collection '{collection_name}' is empty")

        self.bm25, self.all_ids, self.all_texts, self.all_metadatas = build_bm25_index(
            self.collection, agent_label="Agent 3"
        )

        self.chunk_lookup = {
            chunk_id: (text, meta)
            for chunk_id, text, meta in zip(self.all_ids, self.all_texts, self.all_metadatas)
        }

        log("Agent 3", f"Ready indexed_chunks={self.collection.count()}")

    def retrieve(self, question, plan_name=None):
        dense_chunks = dense_search(self.collection, question, self.top_k_dense, plan_name=plan_name)
        sparse_chunks = sparse_search(
            self.bm25, self.all_ids, self.all_texts, self.all_metadatas,
            question, self.top_k_sparse, plan_name=plan_name,
        )

        fused = reciprocal_rank_fusion(dense_chunks, sparse_chunks, self.top_k_final, k=RRF_K)
        reranked = rerank_fused_chunks(fused, self.chunk_lookup, question, self.top_k_final)

        retrieved = []
        for chunk_id, final_score, rrf_score, dense_rank, sparse_rank, dense_distance, sparse_score in reranked:
            if chunk_id in self.chunk_lookup:
                text, meta = self.chunk_lookup[chunk_id]
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=chunk_id, text=text, metadata=meta,
                        rrf_score=rrf_score, final_score=final_score,
                        dense_rank=dense_rank, sparse_rank=sparse_rank,
                        dense_distance=dense_distance, sparse_score=sparse_score,
                    )
                )

        return retrieved

    def run(self, questions, plan_name):
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
