import hashlib
import re
from pathlib import Path
from datetime import datetime, date
from typing import List, Tuple, Dict, Optional

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


UNSTRUCTURED_DIR = Path("./unstructured-data")
CALPERS_DOCS_DIR = UNSTRUCTURED_DIR / "calpERS-specific-documents"
MARKET_INTEL_DIR = UNSTRUCTURED_DIR / "competitive-outlook"
EXCERPTS_DIR = UNSTRUCTURED_DIR / "example-excerpts"

EMBEDDINGS_DIR = Path("./embeddings")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
MIN_CHUNK_WORDS = 40
BATCH_SIZE = 64

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TODAY = date.today()


FILENAME_TO_SOURCE_TYPE = {
    "202503-BoardBOA": "board_minutes",
    "202506-Invest-Agenda": "board_minutes",
    "202509-Full-Agenda": "board_minutes",
    "202509-invest-Agenda-item05a-01": "board_minutes",
    "202509-invest-Agenda-item05a-04": "board_minutes",
    "202511-Finance-Agenda": "board_minutes",
    "202511-invest-transcript": "board_minutes",
    "acfr-2025": "acfr",
    "annual-investment-report": "annual_investment_report",
    "total-fund-investment-policy": "investment_policy",
    "goldman": "market_research",
    "pimco": "market_research",
    "march2025": "board_minutes",
    "june2025": "board_minutes",
    "nov2025": "board_minutes",
    "rfp-doc": "rfp",
}

FILENAME_TO_DATE = {
    "202503-BoardBOA": "2025-03-17",
    "202506-Invest-Agenda": "2025-06-16",
    "202509-Full-Agenda": "2025-09-17",
    "202509-invest-Agenda-item05a-01": "2025-09-17",
    "202509-invest-Agenda-item05a-04": "2025-09-17",
    "202511-Finance-Agenda": "2025-11-17",
    "202511-invest-transcript": "2025-11-17",
    "acfr-2025": "2025-06-30",
    "annual-investment-report": "2025-06-30",
    "total-fund-investment-policy": "2025-01-01",
    "goldman": "2026-01-01",
    "pimco": "2026-01-01",
    "march2025": "2025-03-17",
    "june2025": "2025-06-16",
    "nov2025": "2025-11-17",
    "rfp-doc": "2025-12-01",
}

TOPIC_PATTERNS = {
    "fixed_income": [
        r"\bfixed income\b",
        r"\bcore fixed income\b",
        r"\bglobal fixed income\b",
        r"\bbloomberg (u\.?s\.? )?aggregate\b",
        r"\baggregate bond\b",
        r"\bbond(s)?\b",
    ],
    "duration_risk": [
        r"\bduration\b",
        r"\bduration risk\b",
        r"\brate sensitivity\b",
        r"\binterest rate risk\b",
        r"\brate exposure\b",
        r"\byield curve\b",
    ],
    "esg": [
        r"\besg\b",
        r"\benvironmental[, ]+social[, ]+and[, ]+governance\b",
        r"\bclimate risk\b",
        r"\bsustainab(?:ility|le)\b",
        r"\bstewardship\b",
        r"\bissuer governance\b",
        r"\bsovereign governance\b",
        r"\bresponsible investment\b",
    ],
    "fees": [
        r"\bfee(s)?\b",
        r"\bbasis points?\b",
        r"\bbps\b",
        r"\bpricing\b",
        r"\bexpense(s)?\b",
        r"\bcost structure\b",
        r"\bfee competitiveness\b",
        r"\bmanagement fee\b",
    ],
    "manager_review": [
        r"\bunder review\b",
        r"\bprobation\b",
        r"\bperformance review\b",
        r"\bwatchlist\b",
        r"\bmanager c\b",
        r"\bund(er)?performance\b",
        r"\bbelow benchmark\b",
        r"\bnegative alpha\b",
    ],
    "macro": [
        r"\binflation\b",
        r"\bfederal reserve\b",
        r"\bfed\b",
        r"\brate hike(s)?\b",
        r"\bmonetary policy\b",
        r"\brecession\b",
        r"\bgdp\b",
        r"\bstagflation\b",
        r"\bcentral bank(s)?\b",
        r"\bfiscal deficit(s)?\b",
    ],
    "funded_status": [
        r"\bfunded status\b",
        r"\bfunding ratio\b",
        r"\bactuarial\b",
        r"\bliabilit(y|ies)\b",
        r"\bunfunded\b",
    ],
    "diversification": [
        r"\bdiversif(?:y|ication|ied)\b",
        r"\ballocation\b",
        r"\basset mix\b",
        r"\bportfolio construction\b",
    ],
    "emerging_markets": [
        r"\bemerging market(s)?\b",
        r"\bembi\b",
        r"\bsovereign debt\b",
        r"\bem debt\b",
    ],
    "high_yield": [
        r"\bhigh yield\b",
        r"\bcredit spread(s)?\b",
        r"\bjunk bond(s)?\b",
        r"\bbelow investment grade\b",
    ],
    "liability_matching": [
        r"\bliability match(?:ing)?\b",
        r"\bliability driven\b",
        r"\bldi\b",
        r"\bduration match(?:ing)?\b",
    ],
    "evaluation_criteria": [
        r"\bevaluation criteria\b",
        r"\bselection criteria\b",
        r"\bscoring\b",
        r"\brfp\b",
        r"\brequest for proposal\b",
    ],
    "team_stability": [
        r"\bteam stability\b",
        r"\bpersonnel\b",
        r"\bturnover\b",
        r"\bkey person\b",
        r"\bportfolio manager\b",
        r"\bstaff turnover\b",
    ],
    "risk_management": [
        r"\brisk management\b",
        r"\bdrawdown\b",
        r"\bvolatility\b",
        r"\bstress test(?:ing)?\b",
        r"\bscenario analysis\b",
        r"\btracking error\b",
        r"\bvalue at risk\b",
        r"\bvar\b",
    ],
}

CONCERN_PATTERNS = [
    r"\braised concerns?\b",
    r"\bexpressed concerns?\b",
    r"\bworried about\b",
    r"\bskeptical\b",
    r"\bnot convinced\b",
    r"\bunderperform(?:ed|ing)?\b",
    r"\bbelow benchmark\b",
    r"\bon probation\b",
    r"\bquestioned whether\b",
    r"\bpushed back\b",
    r"\bfalling short\b",
    r"\bdisappointed\b",
    r"\binsufficient\b",
    r"\binadequate\b",
    r"\bfailed to\b",
    r"\blacks?\b",
    r"\bweakness(?:es)?\b",
    r"\bdoes not meet\b",
    r"\bdid not meet\b",
    r"\btrailing\b",
    r"\bnegative alpha\b",
    r"\bduration risk\b",
    r"\brate risk\b",
]

POSITIVE_PATTERNS = [
    r"\bstrong performance\b",
    r"\boutperform(?:ed|ing)?\b",
    r"\bimpressed\b",
    r"\babove benchmark\b",
    r"\btop quartile\b",
    r"\bexceeds?\b",
    r"\bconfident\b",
    r"\bpleased\b",
    r"\brecommends?\b",
    r"\bconsistent alpha\b",
    r"\bwell-positioned\b",
    r"\bsolid\b",
    r"\bcommends?\b",
    r"\bpositive trajectory\b",
    r"\bimproved\b",
    r"\bahead of benchmark\b",
    r"\bstabilizing force\b",
    r"\bsupportive\b",
]

PERSON_ALIASES = {
    "Theresa Taylor": [
        r"\btheresa taylor\b",
        r"\bpresident taylor\b",
        r"\bms\.? taylor\b",
    ],
    "Frank Miller": [
        r"\bfrank miller\b",
        r"\bvice president miller\b",
    ],
    "David Miller": [
        r"\bdavid miller\b",
    ],
    "Eraina Ortega": [
        r"\beraina ortega\b",
        r"\bms\.? ortega\b",
        r"\bortega\b",
    ],
    "Rob Feckner": [
        r"\brob feckner\b",
        r"\bfeckner\b",
    ],
    "Stacie Olivares": [
        r"\bstacie olivares\b",
        r"\bms\.? olivares\b",
        r"\bolivares\b",
    ],
    "Sharris Jones": [
        r"\bsharris jones\b",
        r"\bms\.? jones\b",
        r"\bjones\b",
    ],
    "Malia Cohen": [
        r"\bmalia (m\. )?cohen\b",
    ],
    "Fiona Ma": [
        r"\bfiona ma\b",
    ],
    "Lisa Middleton": [
        r"\blisa middleton\b",
        r"\bmiddleton\b",
    ],
    "Adrian Velasco": [
        r"\badrian velasco\b",
        r"\bvelasco\b",
    ],
    "Marisol Bennett": [
        r"\bmarisol bennett\b",
        r"\bbennett\b",
    ],
    "Kenneth Park": [
        r"\bkenneth (y\. )?park\b",
        r"\bpark\b",
    ],
    "Natalie Greer": [
        r"\bnatalie (s\. )?greer\b",
        r"\bgreer\b",
    ],
    "Stephen Gilmore": [
        r"\bstephen gilmore\b",
        r"\bmr\.? gilmore\b",
        r"\bgilmore\b",
        r"\bchief investment officer\b",
        r"\bcio\b",
    ],
    "Tom Toth": [
        r"\btom toth\b",
        r"\btoth\b",
    ],
    "Steve McCourt": [
        r"\bsteve mccourt\b",
        r"\bmccourt\b",
    ],
}


def compile_pattern_map(pattern_map: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    return {
        key: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        for key, patterns in pattern_map.items()
    }


COMPILED_TOPIC_PATTERNS = compile_pattern_map(TOPIC_PATTERNS)
COMPILED_PERSON_ALIASES = compile_pattern_map(PERSON_ALIASES)
COMPILED_CONCERN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CONCERN_PATTERNS]
COMPILED_POSITIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in POSITIVE_PATTERNS]


def log(message: str) -> None:
    print(f"LOG: {message}")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def split_into_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n+", text)
    paragraphs = []

    for block in blocks:
        cleaned = normalize_whitespace(block.replace("\n", " "))
        if cleaned:
            paragraphs.append(cleaned)

    return paragraphs


def estimate_word_count(text: str) -> int:
    return len(text.split())


def get_source_type(filename: str, stats: dict) -> str:
    for pattern, source_type in FILENAME_TO_SOURCE_TYPE.items():
        if pattern.lower() in filename.lower():
            return source_type

    stats["unknown_source_type_files"] += 1
    log(f"unknown_source_type file={filename}")
    return "unknown"


def get_doc_date(filename: str, stats: dict) -> str:
    for pattern, doc_date in FILENAME_TO_DATE.items():
        if pattern.lower() in filename.lower():
            return doc_date

    stats["fallback_date_files"] += 1
    log(f"fallback_date file={filename}")
    return "2025-01-01"


def calculate_recency_weight(doc_date_str: str) -> float:
    try:
        doc_date = datetime.strptime(doc_date_str, "%Y-%m-%d").date()
        days_old = (TODAY - doc_date).days
        if days_old < 30:
            return 1.00
        if days_old < 90:
            return 0.95
        if days_old < 180:
            return 0.85
        if days_old < 365:
            return 0.70
        if days_old < 730:
            return 0.50
        return 0.30
    except Exception:
        return 0.50


def extract_topics(text: str) -> str:
    found = []
    for topic, patterns in COMPILED_TOPIC_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            found.append(topic)
    return ",".join(found) if found else "general"


def classify_tone(text: str) -> str:
    concern_hits = sum(1 for pattern in COMPILED_CONCERN_PATTERNS if pattern.search(text))
    positive_hits = sum(1 for pattern in COMPILED_POSITIVE_PATTERNS if pattern.search(text))

    if concern_hits > positive_hits:
        return "concern"
    if positive_hits > concern_hits:
        return "positive"
    return "neutral"


def extract_people(text: str) -> str:
    found = []
    for canonical_name, patterns in COMPILED_PERSON_ALIASES.items():
        if any(pattern.search(text) for pattern in patterns):
            found.append(canonical_name)
    return ",".join(found)


def extract_text_from_pdf(filepath: Path) -> List[Tuple[str, int]]:
    reader = PdfReader(str(filepath))
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append((text.strip(), i + 1))

    return pages


def extract_total_pdf_pages(filepath: Path) -> int:
    reader = PdfReader(str(filepath))
    return len(reader.pages)


def extract_text_from_md(filepath: Path) -> List[Tuple[str, int]]:
    text = filepath.read_text(encoding="utf-8")
    return [(text.strip(), 1)]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[str] = []
    current_words: List[str] = []

    def flush_current() -> None:
        nonlocal current_words
        if current_words:
            chunk = normalize_whitespace(" ".join(current_words))
            if estimate_word_count(chunk) >= MIN_CHUNK_WORDS:
                chunks.append(chunk)
            current_words = []

    for para in paragraphs:
        para_words = para.split()

        if len(para_words) <= chunk_size:
            if len(current_words) + len(para_words) <= chunk_size:
                current_words.extend(para_words)
            else:
                overlap_words = current_words[-overlap:] if overlap > 0 else []
                flush_current()
                current_words = overlap_words + para_words
            continue

        sentence_like_parts = re.split(r"(?<=[.!?;])\s+", para)
        for part in sentence_like_parts:
            part = normalize_whitespace(part)
            if not part:
                continue

            part_words = part.split()

            if len(part_words) <= chunk_size:
                if len(current_words) + len(part_words) <= chunk_size:
                    current_words.extend(part_words)
                else:
                    overlap_words = current_words[-overlap:] if overlap > 0 else []
                    flush_current()
                    current_words = overlap_words + part_words
                continue

            start = 0
            step = max(chunk_size - overlap, 1)
            while start < len(part_words):
                piece = part_words[start:start + chunk_size]
                if current_words:
                    flush_current()
                current_words = piece
                flush_current()
                if overlap > 0:
                    current_words = piece[-overlap:]
                else:
                    current_words = []
                start += step

    flush_current()
    return chunks


def make_chunk_id(filepath: Path, page_number: int, chunk: str) -> str:
    normalized = normalize_for_hash(chunk)
    digest = hashlib.md5(
        f"{filepath.stem}|{page_number}|{normalized}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{filepath.stem}-p{page_number}-{digest}"


def make_ingest_stats() -> dict:
    return {
        "files_seen": 0,
        "files_processed": 0,
        "files_failed": 0,
        "pages_seen": 0,
        "pages_with_text": 0,
        "pages_without_text": 0,
        "chunks_created": 0,
        "chunks_ingested": 0,
        "chunks_skipped_short": 0,
        "duplicate_chunks_skipped": 0,
        "unknown_source_type_files": 0,
        "fallback_date_files": 0,
        "batches_written": 0,
    }


def batch_upsert(collection, records: List[dict], stats: dict, batch_size: int = BATCH_SIZE) -> None:
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        collection.upsert(
            ids=[r["id"] for r in batch],
            documents=[r["document"] for r in batch],
            metadatas=[r["metadata"] for r in batch],
        )
        stats["batches_written"] += 1


def ingest_directory(
    directory: Path,
    collection,
    collection_name: str,
    pension_plan: Optional[str],
    stats: dict,
) -> None:
    files = sorted(list(directory.glob("*.pdf")) + list(directory.glob("*.md")))

    if not files:
        log(f"no_files directory={directory}")
        return

    log(f"ingest_directory name={directory.name} collection={collection_name}")

    for filepath in files:
        stats["files_seen"] += 1
        filename = filepath.name

        source_type = get_source_type(filename, stats)
        doc_date = get_doc_date(filename, stats)
        recency_weight = calculate_recency_weight(doc_date)

        log(
            f"file={filename} type={source_type} date={doc_date} recency={recency_weight:.2f}"
        )

        try:
            if filepath.suffix.lower() == ".pdf":
                total_pages = extract_total_pdf_pages(filepath)
                pages = extract_text_from_pdf(filepath)
            else:
                total_pages = 1
                pages = extract_text_from_md(filepath)
        except Exception as e:
            stats["files_failed"] += 1
            log(f"extract_failed file={filename} error={repr(e)}")
            continue

        stats["pages_seen"] += total_pages
        stats["pages_with_text"] += len(pages)
        stats["pages_without_text"] += max(total_pages - len(pages), 0)

        if not pages:
            log(f"skip_no_text file={filename}")
            continue

        seen_chunk_hashes = set()
        records_to_upsert = []

        for page_text, page_number in pages:
            chunks = chunk_text(page_text)
            stats["chunks_created"] += len(chunks)

            for chunk in chunks:
                if estimate_word_count(chunk) < MIN_CHUNK_WORDS:
                    stats["chunks_skipped_short"] += 1
                    continue

                normalized_hash = normalize_for_hash(chunk)
                if normalized_hash in seen_chunk_hashes:
                    stats["duplicate_chunks_skipped"] += 1
                    continue
                seen_chunk_hashes.add(normalized_hash)

                chunk_id = make_chunk_id(filepath, page_number, chunk)
                topics = extract_topics(chunk)
                tone = classify_tone(chunk)
                people = extract_people(chunk)

                metadata = {
                    "chunk_id": chunk_id,
                    "source_file": filename,
                    "source_type": source_type,
                    "pension_plan": pension_plan or "",
                    "date": doc_date,
                    "recency_weight": recency_weight,
                    "topics": topics,
                    "tone": tone,
                    "people_mentioned": people,
                    "confidence": "verbatim",
                    "page_number": page_number,
                    "collection": collection_name,
                }

                records_to_upsert.append({
                    "id": chunk_id,
                    "document": chunk,
                    "metadata": metadata,
                })

        if records_to_upsert:
            batch_upsert(collection, records_to_upsert, stats, batch_size=BATCH_SIZE)

        stats["chunks_ingested"] += len(records_to_upsert)
        stats["files_processed"] += 1
        log(f"chunks_ingested file={filename} count={len(records_to_upsert)}")


def verify(calpers_collection, market_collection, stats: dict) -> None:
    log("verification")
    log(f"calpers_docs_chunks={calpers_collection.count()}")
    log(f"market_intel_chunks={market_collection.count()}")

    log("ingestion_stats")
    log(f"files_seen={stats['files_seen']}")
    log(f"files_processed={stats['files_processed']}")
    log(f"files_failed={stats['files_failed']}")
    log(f"pages_seen={stats['pages_seen']}")
    log(f"pages_with_text={stats['pages_with_text']}")
    log(f"pages_without_text={stats['pages_without_text']}")
    log(f"chunks_created={stats['chunks_created']}")
    log(f"chunks_ingested={stats['chunks_ingested']}")
    log(f"chunks_skipped_short={stats['chunks_skipped_short']}")
    log(f"duplicate_chunks_skipped={stats['duplicate_chunks_skipped']}")
    log(f"unknown_source_type_files={stats['unknown_source_type_files']}")
    log(f"fallback_date_files={stats['fallback_date_files']}")
    log(f"batches_written={stats['batches_written']}")

    log("sample_query='duration risk concerns'")
    results = calpers_collection.query(
        query_texts=["duration risk concerns fixed income"],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        preview = normalize_whitespace(doc[:180])
        log(
            f"result index={i} source={meta.get('source_file')} "
            f"page={meta.get('page_number')} date={meta.get('date')} "
            f"weight={meta.get('recency_weight')} distance={dist:.4f}"
        )
        log(f"topics={meta.get('topics')} tone={meta.get('tone')}")
        log(f"preview={preview}")


def reset_collection(client, name: str, embed_fn):
    try:
        client.delete_collection(name)
        log(f"collection_deleted name={name}")
    except Exception:
        log(f"collection_not_deleted name={name}")

    collection = client.get_or_create_collection(
        name=name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    log(f"collection_ready name={name}")
    return collection


def main() -> None:
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    log(f"embeddings_dir={EMBEDDINGS_DIR.resolve()}")
    log(f"loading_model={EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    class STEmbeddingFunction(chromadb.EmbeddingFunction):
        def __call__(self, input):
            texts = [input] if isinstance(input, str) else list(input)
            return model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
            ).tolist()

    client = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR))
    embed_fn = STEmbeddingFunction()

    calpers_collection = reset_collection(client, "calpers_docs", embed_fn)
    market_collection = reset_collection(client, "market_intel", embed_fn)

    stats = make_ingest_stats()

    ingest_directory(
        CALPERS_DOCS_DIR,
        calpers_collection,
        "calpers_docs",
        pension_plan="CalPERS",
        stats=stats,
    )

    ingest_directory(
        EXCERPTS_DIR,
        calpers_collection,
        "calpers_docs",
        pension_plan="CalPERS",
        stats=stats,
    )

    ingest_directory(
        MARKET_INTEL_DIR,
        market_collection,
        "market_intel",
        pension_plan=None,
        stats=stats,
    )

    verify(calpers_collection, market_collection, stats)


if __name__ == "__main__":
    main()