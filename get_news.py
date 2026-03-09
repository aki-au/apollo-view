import os
import re
from datetime import date, datetime
from urllib.parse import urlparse


MOCK_MODE = False
MAX_RESULTS = 4
TAVILY_API_KEY = 'tvly-dev-'


def log(agent, message):
    print(f"LOG: {agent}: {message}")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", normalize_text(text))


def safe_parse_date(value: str):
    if not value:
        return None

    value = str(value).strip()

    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date()
        except Exception:
            pass

    match = re.match(r"^(\d{4}-\d{2}-\d{2})", value)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except Exception:
            pass

    return None


def recency_score(published_date: str) -> float:
    d = safe_parse_date(published_date)
    if not d:
        return 0.35

    days_old = (date.today() - d).days
    if days_old <= 7:
        return 1.0
    if days_old <= 30:
        return 0.85
    if days_old <= 90:
        return 0.65
    if days_old <= 180:
        return 0.45
    return 0.25


def keyword_overlap_score(query: str, title: str, summary: str) -> float:
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0

    doc_tokens = set(tokenize(title) + tokenize(summary))
    overlap = len(q_tokens.intersection(doc_tokens))
    return min(overlap / max(len(q_tokens), 1), 1.0)


def detect_preferred_sources(query: str) -> set[str]:
    q = normalize_text(query)
    preferred = set()

    if any(x in q for x in ["board", "governance", "trustee", "oversight", "investment committee"]):
        preferred.update({"calpers", "pensions & investments"})
    if any(x in q for x in ["fixed income", "rate", "fed", "yield", "aggregate", "treasury", "credit spreads"]):
        preferred.update({"wall street journal", "bloomberg", "financial times", "pensions & investments", "institutional investor"})
    if any(x in q for x in ["esg", "climate", "stewardship", "sustainability"]):
        preferred.update({"calpers", "institutional investor", "pensions & investments"})

    return preferred


def source_preference_score(source: str, preferred_sources: set[str]) -> float:
    source_norm = normalize_text(source)
    if not preferred_sources:
        return 0.0
    return 1.0 if source_norm in preferred_sources else 0.0


def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def dedupe_news_items(items):
    seen = set()
    deduped = []

    for item in items:
        key = (
            normalize_text(item.title),
            normalize_text(item.url),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped


def expand_query(query: str, plan_name: str, strategy: str) -> str:
    q = normalize_text(query)
    additions = [plan_name, strategy]

    if any(x in q for x in ["board", "governance", "trustee", "committee"]):
        additions.extend(["investment committee", "board oversight"])
    if any(x in q for x in ["fixed income", "bond", "aggregate"]):
        additions.extend(["rates", "yields", "duration"])
    if any(x in q for x in ["fed", "rate", "inflation", "central bank"]):
        additions.extend(["monetary policy", "treasury yields"])
    if any(x in q for x in ["esg", "climate", "stewardship"]):
        additions.extend(["proxy voting", "sustainability", "disclosure"])
    if "core fixed income" in normalize_text(strategy):
        additions.extend(["bloomberg aggregate", "investment grade"])

    additions = [a for a in additions if a]
    expanded = q + " " + " ".join(dict.fromkeys(additions))
    return expanded.strip()


class NewsItem:
    def __init__(
        self,
        title,
        url,
        published_date,
        source,
        summary,
        relevance_score=None,
        final_score=None,
        mode=None,
    ):
        self.title = title
        self.url = url
        self.published_date = published_date
        self.source = source
        self.summary = summary
        self.relevance_score = relevance_score
        self.final_score = final_score
        self.mode = mode

    def __repr__(self):
        return (
            f"NewsItem(title={self.title[:50]!r}, source={self.source!r}, "
            f"date={self.published_date!r}, final_score={self.final_score})"
        )


class NewsQuestionResult:
    def __init__(self, question, items, expanded_query=None, mode_used=None):
        self.question = question
        self.items = items
        self.expanded_query = expanded_query
        self.mode_used = mode_used

    def __repr__(self):
        return f"NewsQuestionResult(question={self.question[:50]!r}, items={len(self.items)})"


class NewsResult:
    def __init__(self, plan_name, strategy, question_results):
        self.plan_name = plan_name
        self.strategy = strategy
        self.question_results = question_results

    def __repr__(self):
        total = sum(len(r.items) for r in self.question_results)
        return f"NewsResult(plan={self.plan_name!r}, strategy={self.strategy!r}, total_items={total})"

    def to_context_string(self, max_items_per_question: int = 3) -> str:
        lines = [f"=== NEWS: {self.plan_name} / {self.strategy} ===\n"]

        for qr in self.question_results:
            lines.append(f"Question: {qr.question}")
            if qr.expanded_query and normalize_text(qr.expanded_query) != normalize_text(qr.question):
                lines.append(f"Expanded query: {qr.expanded_query}")
            if qr.mode_used:
                lines.append(f"Mode used: {qr.mode_used}")

            if not qr.items:
                lines.append("  No recent news found.\n")
                continue

            for i, item in enumerate(qr.items[:max_items_per_question], 1):
                score_str = f" | final_score={item.final_score:.3f}" if item.final_score is not None else ""
                rel_str = f" | relevance={item.relevance_score:.2f}" if item.relevance_score is not None else ""
                lines.append(f"  [{i}] {item.title}")
                lines.append(f"      {item.source} | {item.published_date}{rel_str}{score_str}")
                lines.append(f"      {item.summary}")
                lines.append(f"      {item.url}")
                lines.append("")

            if len(qr.items) > max_items_per_question:
                lines.append(f"  ... {len(qr.items) - max_items_per_question} more items omitted\n")

        return "\n".join(lines)


MOCK_NEWS_BANK = {
    "calpers_board": [
        NewsItem(
            title="CalPERS Board Approves Updated Investment Beliefs Framework",
            url="https://www.calpers.ca.gov/news/2026/investment-beliefs",
            published_date="2026-02-14",
            source="CalPERS",
            summary="The CalPERS board approved revisions to its investment beliefs, emphasizing long-term value creation, climate risk integration, and fee transparency.",
            mode="mock",
        ),
        NewsItem(
            title="CalPERS CIO Outlines 2026 Fixed Income Priorities at Annual Forum",
            url="https://www.pionline.com/calpers-cio-fixed-income-2026",
            published_date="2026-01-28",
            source="Pensions & Investments",
            summary="The CIO highlighted duration management, credit quality preservation, and reducing fee drag as key fixed income priorities for 2026.",
            mode="mock",
        ),
        NewsItem(
            title="CalPERS Faces Pension Obligation Bond Debate Amid Funding Gap",
            url="https://www.latimes.com/calpers-pension-bond-2026",
            published_date="2026-02-03",
            source="Los Angeles Times",
            summary="California legislators are revisiting pension obligation bonds as a way to address the funding gap, with staff analysis expected in a future board meeting.",
            mode="mock",
        ),
    ],
    "fixed_income_market": [
        NewsItem(
            title="Fed Signals Patience on Rate Cuts as Inflation Stays Above Target",
            url="https://www.wsj.com/fed-rate-cuts-2026-patience",
            published_date="2026-02-20",
            source="Wall Street Journal",
            summary="Federal Reserve officials indicated they are in no rush to cut rates further, leaving Treasury yields elevated and duration positioning under scrutiny.",
            mode="mock",
        ),
        NewsItem(
            title="Bloomberg Aggregate Index Sees Volatile Start to 2026",
            url="https://www.bloomberg.com/agg-index-2026-volatility",
            published_date="2026-02-10",
            source="Bloomberg",
            summary="The Bloomberg U.S. Aggregate Bond Index had a volatile start to the year as stronger macro data pushed rate expectations higher.",
            mode="mock",
        ),
        NewsItem(
            title="Investment Grade Credit Spreads Tighten Despite Macro Uncertainty",
            url="https://www.ft.com/ig-credit-spreads-2026",
            published_date="2026-02-05",
            source="Financial Times",
            summary="Investment grade spreads remained tight despite macro uncertainty, raising questions about limited upside relative to duration risk.",
            mode="mock",
        ),
    ],
    "calpers_governance": [
        NewsItem(
            title="CalPERS Trustee Election Results Shift Board Composition",
            url="https://www.pionline.com/calpers-trustee-election-2026",
            published_date="2026-01-15",
            source="Pensions & Investments",
            summary="New member-elected trustees joined the board, potentially increasing scrutiny of manager fees and ESG implementation.",
            mode="mock",
        ),
        NewsItem(
            title="CalPERS Launches External Manager Fee Transparency Initiative",
            url="https://www.calpers.ca.gov/news/2026/fee-transparency",
            published_date="2026-02-01",
            source="CalPERS",
            summary="CalPERS announced a new standardized fee reporting initiative for external managers after board concerns about fee drag.",
            mode="mock",
        ),
    ],
    "esg": [
        NewsItem(
            title="CalPERS Votes Against 12 Directors Over Climate Disclosure Failures",
            url="https://www.calpers.ca.gov/news/2026/proxy-voting-climate",
            published_date="2026-02-18",
            source="CalPERS",
            summary="CalPERS opposed director nominees at multiple companies over climate disclosure shortcomings, reinforcing its ESG posture.",
            mode="mock",
        ),
        NewsItem(
            title="Fixed Income ESG Integration Gains Traction Among Public Pensions",
            url="https://www.institutionalinvestor.com/fixed-income-esg-2026",
            published_date="2026-01-22",
            source="Institutional Investor",
            summary="A growing share of public pensions now require ESG integration in fixed income mandates, though implementation standards vary.",
            mode="mock",
        ),
    ],
    "fixed_income_strategy": [
        NewsItem(
            title="Core Fixed Income Managers Face Pressure as Active Returns Lag",
            url="https://www.pionline.com/core-fixed-income-active-returns-2026",
            published_date="2026-02-12",
            source="Pensions & Investments",
            summary="Active core fixed income managers underperformed on average in 2025, increasing pressure to justify fees.",
            mode="mock",
        ),
        NewsItem(
            title="Public Pensions Reassessing Core Fixed Income Allocations in 2026",
            url="https://www.institutionalinvestor.com/pensions-fixed-income-2026",
            published_date="2026-02-08",
            source="Institutional Investor",
            summary="Large public pensions are revisiting core fixed income allocations in the higher-rate environment and weighing duration and manager structure decisions.",
            mode="mock",
        ),
    ],
    "calpers_annual_meeting": [
        NewsItem(
            title="CalPERS 2026 Annual Review to Focus on Manager Accountability",
            url="https://www.pionline.com/calpers-annual-review-2026",
            published_date="2026-02-22",
            source="Pensions & Investments",
            summary="CalPERS is expected to scrutinize active manager performance more closely in 2026, especially in fixed income mandates nearing renewal.",
            mode="mock",
        ),
    ],
}

MOCK_TOPIC_ROUTING = [
    (["board meeting", "board news", "calpers board"], "calpers_board"),
    (["fixed income market", "rate", "fed", "treasury", "yield", "aggregate", "bloomberg agg"], "fixed_income_market"),
    (["governance", "trustee", "oversight", "board composition", "fee transparency"], "calpers_governance"),
    (["esg", "climate", "sustainability", "stewardship", "proxy"], "esg"),
    (["fixed income strategy", "active manager", "core fixed income strategy", "allocation"], "fixed_income_strategy"),
    (["annual meeting", "annual review", "manager accountability"], "calpers_annual_meeting"),
]


def _mock_search(query: str, plan_name: str, strategy: str, max_results: int) -> list[NewsItem]:
    q = normalize_text(query)
    matched_keys = []

    for keywords, bank_key in MOCK_TOPIC_ROUTING:
        if any(kw in q for kw in keywords):
            if bank_key not in matched_keys:
                matched_keys.append(bank_key)

    if not matched_keys:
        if "esg" in q or "climate" in q:
            matched_keys = ["esg"]
        elif "governance" in q or "board" in q:
            matched_keys = ["calpers_governance", "calpers_board"]
        elif "fixed income" in q or "rate" in q or "yield" in q:
            matched_keys = ["fixed_income_market", "fixed_income_strategy"]
        else:
            matched_keys = ["calpers_board"]

    items = []
    for key in matched_keys:
        items.extend(MOCK_NEWS_BANK.get(key, []))

    items = dedupe_news_items(items)
    return rank_news_items(items, query=query, plan_name=plan_name, strategy=strategy)[:max_results]


def _tavily_search(query: str, plan_name: str, strategy: str, max_results: int) -> list[NewsItem]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)
    enriched_query = expand_query(query, plan_name, strategy)

    response = client.search(
        query=enriched_query,
        search_depth="advanced",
        max_results=max_results * 2,
        include_answer=False,
    )

    items = []
    for r in response.get("results", []):
        items.append(
            NewsItem(
                title=r.get("title", ""),
                url=r.get("url", ""),
                published_date=r.get("published_date", "unknown"),
                source=r.get("source", extract_domain(r.get("url", "")) or "unknown"),
                summary=r.get("content", ""),
                relevance_score=r.get("score"),
                mode="live",
            )
        )

    items = dedupe_news_items(items)
    return rank_news_items(items, query=query, plan_name=plan_name, strategy=strategy)[:max_results]


def rank_news_items(items: list[NewsItem], query: str, plan_name: str, strategy: str) -> list[NewsItem]:
    preferred_sources = detect_preferred_sources(query)
    enriched_query = expand_query(query, plan_name, strategy)

    ranked = []
    for item in items:
        title = item.title or ""
        summary = item.summary or ""
        source = item.source or ""

        base_relevance = float(item.relevance_score) if item.relevance_score is not None else 0.55
        recency = recency_score(item.published_date)
        overlap = keyword_overlap_score(enriched_query, title, summary)
        source_pref = source_preference_score(source, preferred_sources)

        final_score = (
            0.45 * base_relevance +
            0.30 * recency +
            0.20 * overlap +
            0.05 * source_pref
        )

        item.final_score = final_score
        ranked.append(item)

    ranked.sort(
        key=lambda x: (
            x.final_score if x.final_score is not None else 0.0,
            safe_parse_date(x.published_date) or date(1970, 1, 1),
        ),
        reverse=True,
    )
    return ranked


class NewsAgent:
    def __init__(self, max_results: int = MAX_RESULTS, mock_mode: bool | None = None):
        self.max_results = max_results

        if mock_mode is None:
            self.mock_mode = MOCK_MODE or not bool(TAVILY_API_KEY)
        else:
            self.mock_mode = mock_mode

        if self.mock_mode:
            self.mode_name = "mock"
        elif TAVILY_API_KEY:
            self.mode_name = "live"
        else:
            self.mode_name = "mock"

        log("Agent 5", f"Ready mode={self.mode_name} max_results_per_question={max_results}")

    def search(self, query: str, plan_name: str, strategy: str) -> tuple[list[NewsItem], str, str]:
        expanded_query = expand_query(query, plan_name, strategy)

        if self.mock_mode:
            return _mock_search(query, plan_name, strategy, self.max_results), expanded_query, "mock"

        try:
            items = _tavily_search(query, plan_name, strategy, self.max_results)
            return items, expanded_query, "live"
        except Exception as e:
            log("Agent 5", f"Live search failed error={repr(e)} fallback=mock")
            return _mock_search(query, plan_name, strategy, self.max_results), expanded_query, "mock_fallback"

    def run(self, questions: list, plan_name: str, strategy: str) -> NewsResult:
        log("Agent 5", f"Running news questions={len(questions)} plan={plan_name!r} strategy={strategy!r}")

        question_results = []

        for i, question in enumerate(questions, 1):
            log("Agent 5", f"Question {i}/{len(questions)}: {question[:100]}")
            items, expanded_query, mode_used = self.search(question, plan_name, strategy)
            question_results.append(
                NewsQuestionResult(
                    question=question,
                    items=items,
                    expanded_query=expanded_query,
                    mode_used=mode_used,
                )
            )
            log("Agent 5", f"Items found={len(items)} mode_used={mode_used}")

        result = NewsResult(
            plan_name=plan_name,
            strategy=strategy,
            question_results=question_results,
        )

        log("Agent 5", f"Complete result={result}")
        return result


if __name__ == "__main__":
    test_questions = [
        "CalPERS board meeting news",
        "Fixed income market news and trends",
        "CalPERS governance and investment oversight news",
        "ESG news and developments relevant to CalPERS",
        "Fixed income strategy news from industry sources",
        "CalPERS annual meeting and review news",
    ]

    agent = NewsAgent(mock_mode=False)
    result = agent.run(
        questions=test_questions,
        plan_name="CalPERS",
        strategy="Core Fixed Income",
    )

    print(result)
    print()
    print(result.to_context_string(max_items_per_question=3))