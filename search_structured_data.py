import json
import re
import sqlite3
from pathlib import Path
from typing import Optional
import requests


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
DB_PATH = Path("./calpers.db")


def log(agent, message):
    print(f"LOG: {agent}: {message}")


def ollama_generate(prompt: str, timeout: int = 120) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


DB_SCHEMA = """
Tables in the CalPERS database:

plan_financials:
  plan_name, total_aum_billions, funded_status_pct, fy2025_return_pct,
  discount_rate_pct, return_1yr_pct, return_3yr_pct, return_5yr_pct,
  return_10yr_pct, return_20yr_pct, total_members, annual_benefit_payments_billions,
  employer_contribution_rate_pct, last_actuarial_review_date,
  next_actuarial_review_date, investment_consultants, fiscal_year_end

manager_roster:
  manager_id, manager_name, strategy_type, aum_managed_billions,
  mandate_start_date, contract_renewal_date, benchmark, fee_bps,
  fee_dollars_millions, mandate_status, internal_or_external,
  primary_contact, last_review_date, notes

performance:
  manager_id, manager_name, strategy_type, return_1yr_pct, return_3yr_pct,
  return_5yr_pct, benchmark_return_1yr_pct, benchmark_return_3yr_pct,
  benchmark_return_5yr_pct, alpha_1yr_bps, alpha_3yr_bps, alpha_5yr_bps,
  sharpe_ratio, tracking_error_pct, information_ratio,
  performance_vs_peers, last_performance_review_date

board_members:
  member_id, name, role, appointment_type, professional_background,
  tenure_years, term_expiry_date, committee_memberships, known_priorities,
  finance_expertise_level, typical_question_focus, notes

key_dates:
  date_id, event_type, event_name, date, related_manager_id,
  priority, description, action_required, action_description

past_meeting_notes:
  note_id, meeting_date, meeting_type, calpers_attendees, apex_attendees,
  location, strategy_discussed, key_discussion_points, board_concerns_raised,
  action_items, outcome, follow_up_required, internal_notes, sentiment_score

competitive_intelligence:
  competitor_id, firm_name, strategy_focus, estimated_calpers_aum_billions,
  is_current_calpers_manager, known_fee_range_bps, esg_rating,
  recent_mandate_wins, recent_mandate_losses, known_strengths,
  known_weaknesses, threat_level, notes
"""


STRATEGY_ALIASES = {
    "core fixed income": [
        "core fixed income",
        "core bond",
        "core bonds",
        "aggregate",
        "agg",
        "us aggregate",
        "bloomberg us aggregate",
    ],
    "global fixed income": [
        "global fixed income",
        "global bond",
        "global bonds",
        "global aggregate",
    ],
    "inflation-linked": [
        "inflation-linked",
        "inflation linked",
        "tips",
        "linkers",
        "inflation bonds",
    ],
    "short duration": [
        "short duration",
        "short-term bond",
        "short term bond",
        "1-3 year",
        "ultra short",
    ],
    "emerging market debt": [
        "emerging market debt",
        "em debt",
        "emd",
        "emerging debt",
        "embi",
        "sovereign debt",
    ],
    "high yield": [
        "high yield",
        "junk bond",
        "junk bonds",
        "below investment grade",
        "credit spreads",
    ],
}


QUESTION_ALIASES = {
    "funded_status": ["funded status", "funding ratio", "funded"],
    "aum": ["aum", "assets under management", "total assets", "plan size"],
    "fees": ["fee", "fees", "basis points", "bps", "pricing", "cost", "expense"],
    "performance": [
        "performance", "returns", "alpha", "benchmark", "track record",
        "outperform", "underperform", "information ratio", "sharpe"
    ],
    "renewal": ["contract renewal", "renewal date", "contract expir", "upcoming renewal", "renewal timeline"],
    "watchlist": ["probation", "watchlist", "under review", "mandate status", "flagged", "at risk"],
    "board": ["board member", "board composition", "trustee", "decision maker", "board background"],
    "esg": ["esg", "environmental", "sustainability", "climate", "stewardship", "responsible invest"],
    "competitor": ["competitor", "competition", "competing firm", "peer manager", "other managers", "threat"],
    "calendar": ["key date", "calendar", "upcoming event", "board meeting", "rfp deadline", "schedule", "next meeting", "deadline"],
    "meeting_history": ["meeting history", "past meeting", "relationship history", "previous meeting", "meeting notes", "prior interaction"],
    "roster": ["manager roster", "current managers", "who manages", "manager list", "external managers"],
}


TIME_SCOPE_ALIASES = {
    "upcoming": ["upcoming", "next", "future", "deadline", "calendar", "schedule"],
    "recent": ["recent", "latest", "currently", "now", "current", "most recent"],
    "historical": ["historical", "past", "prior", "previous", "over time"],
}


class ParsedQuestion:
    def __init__(self, raw_question, intents=None, strategy=None, manager=None, time_scope=None):
        self.raw_question = raw_question
        self.intents = intents or []
        self.strategy = strategy
        self.manager = manager
        self.time_scope = time_scope

    def __repr__(self):
        return (
            f"ParsedQuestion(intents={self.intents}, strategy={self.strategy}, "
            f"manager={self.manager}, time_scope={self.time_scope})"
        )


class AnswerRecord:
    def __init__(self, question, label, data, source, used_llm_fallback=False, sql=None):
        self.question = question
        self.label = label
        self.data = data
        self.source = source
        self.used_llm_fallback = used_llm_fallback
        self.sql = sql

    def __repr__(self):
        return f"AnswerRecord(label={self.label!r}, rows={len(self.data)}, source={self.source!r})"

    def is_empty(self):
        return len(self.data) == 0


class StructuredDataResult:
    def __init__(self, plan_name, manager_name, answers):
        self.plan_name = plan_name
        self.manager_name = manager_name
        self.answers = answers

    def __repr__(self):
        answered = sum(1 for a in self.answers if not a.is_empty())
        unanswered = sum(1 for a in self.answers if a.is_empty())
        return f"StructuredDataResult(plan={self.plan_name!r}, answered={answered}, unanswered={unanswered})"

    def to_context_string(self, max_rows_per_answer: int = 5):
        lines = [f"=== STRUCTURED DATA: {self.plan_name} ===\n"]

        for answer in self.answers:
            lines.append(f"{answer.label}")
            lines.append(f"Question: {answer.question}")

            if answer.is_empty():
                lines.append("Result: No data found.\n")
                continue

            rows_to_show = answer.data[:max_rows_per_answer]

            for row in rows_to_show:
                row_lines = []
                for key, value in row.items():
                    if isinstance(value, str) and value.startswith("["):
                        try:
                            parsed = json.loads(value)
                            if isinstance(parsed, list):
                                value = ", ".join(str(x) for x in parsed) if parsed else "None"
                        except Exception:
                            pass
                    row_lines.append(f"  {key}: {value}")
                lines.append("\n".join(row_lines))
                lines.append("")

            if len(answer.data) > max_rows_per_answer:
                lines.append(f"  ... {len(answer.data) - max_rows_per_answer} more rows omitted\n")

            lines.append(f"[Source: {answer.source}]\n")

        return "\n".join(lines)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def detect_strategy(text: str) -> Optional[str]:
    q = normalize_text(text)
    best_strategy = None
    best_len = 0

    for canonical, aliases in STRATEGY_ALIASES.items():
        for alias in aliases:
            if alias in q and len(alias) > best_len:
                best_strategy = canonical
                best_len = len(alias)

    return best_strategy


def detect_time_scope(text: str) -> Optional[str]:
    q = normalize_text(text)
    best_scope = None
    best_score = 0

    for scope, aliases in TIME_SCOPE_ALIASES.items():
        score = sum(1 for alias in aliases if alias in q)
        if score > best_score:
            best_scope = scope
            best_score = score

    return best_scope


def detect_manager_from_question(question: str, manager_name: str = None) -> Optional[str]:
    q = normalize_text(question)

    if manager_name and normalize_text(manager_name) in q:
        return manager_name

    if manager_name and any(
        phrase in q
        for phrase in [
            "our manager", "our mandate", "our performance",
            "our fees", "our contract", "our track record",
            "we", "our strategy"
        ]
    ):
        return manager_name

    return manager_name


def extract_intents(question: str) -> list[str]:
    q = normalize_text(question)
    scored = []

    for intent, aliases in QUESTION_ALIASES.items():
        score = score_keywords(q, aliases)
        if score > 0:
            scored.append((intent, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [intent for intent, _ in scored]


def parse_question(question: str, manager_name: str = None) -> ParsedQuestion:
    return ParsedQuestion(
        raw_question=question,
        intents=extract_intents(question),
        strategy=detect_strategy(question),
        manager=detect_manager_from_question(question, manager_name=manager_name),
        time_scope=detect_time_scope(question),
    )


def run_query(db_path: Path, sql: str, params: tuple = ()) -> list:
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = [dict(row) for row in cursor.fetchall()]
        return rows
    except Exception as e:
        log("Agent 2", f"SQL error: {repr(e)}")
        return []


def validate_sql(sql: str) -> bool:
    if not sql or not isinstance(sql, str):
        return False

    cleaned = sql.strip()
    upper = cleaned.upper()

    if upper.count(";") > 1:
        return False
    if ";" in cleaned[:-1]:
        return False
    if not upper.startswith("SELECT"):
        return False

    disallowed = [
        "DROP ", "DELETE ", "INSERT ", "UPDATE ", "ALTER ", "CREATE ",
        "ATTACH ", "DETACH ", "PRAGMA ", "REINDEX ", "VACUUM ",
        "WITH ", "--", "/*", "*/"
    ]
    if any(token in upper for token in disallowed):
        return False

    allowed_tables = [
        "PLAN_FINANCIALS", "MANAGER_ROSTER", "PERFORMANCE",
        "BOARD_MEMBERS", "KEY_DATES", "PAST_MEETING_NOTES",
        "COMPETITIVE_INTELLIGENCE"
    ]
    if " FROM " in upper and not any(tbl in upper for tbl in allowed_tables):
        return False

    return True


def score_keywords(question: str, keywords: list[str]) -> int:
    q = normalize_text(question)
    score = 0
    for kw in keywords:
        if kw in q:
            score += max(1, len(kw.split()))
    return score


def is_full_roster_question(question: str) -> bool:
    q = normalize_text(question)
    phrases = ["current managers", "manager roster", "who manages", "manager list", "external managers"]
    return any(p in q for p in phrases)


def build_plan_financials_query(parsed: ParsedQuestion, plan_name=None):
    return (
        "Plan Financials",
        """
        SELECT plan_name, total_aum_billions, funded_status_pct,
               fy2025_return_pct, discount_rate_pct,
               return_1yr_pct, return_3yr_pct, return_5yr_pct,
               return_10yr_pct, return_20yr_pct,
               total_members, annual_benefit_payments_billions,
               employer_contribution_rate_pct, investment_consultants,
               fiscal_year_end
        FROM plan_financials
        LIMIT 1
        """,
        (),
    )


def build_manager_roster_query(parsed: ParsedQuestion, plan_name=None):
    sql = """
    SELECT manager_name, strategy_type, aum_managed_billions,
           mandate_status, fee_bps, contract_renewal_date,
           benchmark, mandate_start_date, primary_contact
    FROM manager_roster
    WHERE 1=1
    """
    params = []

    if parsed.strategy:
        sql += " AND lower(strategy_type) = ?"
        params.append(parsed.strategy)

    if parsed.manager and not is_full_roster_question(parsed.raw_question):
        sql += " AND lower(manager_name) LIKE ?"
        params.append(f"%{normalize_text(parsed.manager)}%")

    sql += " ORDER BY strategy_type, manager_name"
    return "Manager Roster", sql, tuple(params)


def build_watchlist_query(parsed: ParsedQuestion, plan_name=None):
    sql = """
    SELECT manager_name, strategy_type, mandate_status,
           contract_renewal_date, fee_bps, notes
    FROM manager_roster
    WHERE mandate_status != 'Active'
    """
    params = []

    if parsed.strategy:
        sql += " AND lower(strategy_type) = ?"
        params.append(parsed.strategy)

    if parsed.manager:
        sql += " AND lower(manager_name) LIKE ?"
        params.append(f"%{normalize_text(parsed.manager)}%")

    sql += " ORDER BY contract_renewal_date ASC"
    return "Managers Under Review or Probation", sql, tuple(params)


def build_renewal_query(parsed: ParsedQuestion, plan_name=None):
    sql = """
    SELECT manager_name, strategy_type, mandate_status,
           contract_renewal_date, fee_bps, benchmark
    FROM manager_roster
    WHERE 1=1
    """
    params = []

    if parsed.strategy:
        sql += " AND lower(strategy_type) = ?"
        params.append(parsed.strategy)

    if parsed.manager:
        sql += " AND lower(manager_name) LIKE ?"
        params.append(f"%{normalize_text(parsed.manager)}%")

    sql += " ORDER BY contract_renewal_date ASC"
    return "Contract Renewals", sql, tuple(params)


def build_fee_query(parsed: ParsedQuestion, plan_name=None):
    sql = """
    SELECT manager_name, strategy_type, fee_bps,
           fee_dollars_millions, mandate_status, benchmark,
           contract_renewal_date
    FROM manager_roster
    WHERE 1=1
    """
    params = []

    if parsed.strategy:
        sql += " AND lower(strategy_type) = ?"
        params.append(parsed.strategy)

    if parsed.manager:
        sql += " AND lower(manager_name) LIKE ?"
        params.append(f"%{normalize_text(parsed.manager)}%")

    sql += " ORDER BY fee_bps DESC"
    return "Fee Structure", sql, tuple(params)


def build_performance_query(parsed: ParsedQuestion, plan_name=None):
    sql = """
    SELECT p.manager_name, p.strategy_type,
           p.return_1yr_pct, p.benchmark_return_1yr_pct, p.alpha_1yr_bps,
           p.return_3yr_pct, p.benchmark_return_3yr_pct, p.alpha_3yr_bps,
           p.return_5yr_pct, p.benchmark_return_5yr_pct, p.alpha_5yr_bps,
           p.sharpe_ratio, p.tracking_error_pct, p.information_ratio,
           p.performance_vs_peers, p.last_performance_review_date,
           m.mandate_status
    FROM performance p
    JOIN manager_roster m ON p.manager_id = m.manager_id
    WHERE 1=1
    """
    params = []

    if parsed.strategy:
        sql += " AND lower(p.strategy_type) = ?"
        params.append(parsed.strategy)

    if parsed.manager:
        sql += " AND lower(p.manager_name) LIKE ?"
        params.append(f"%{normalize_text(parsed.manager)}%")

    q = normalize_text(parsed.raw_question)
    if "underperform" in q or "negative alpha" in q or "lagged" in q:
        sql += " ORDER BY p.alpha_1yr_bps ASC"
    else:
        sql += " ORDER BY p.alpha_1yr_bps DESC"

    return "Performance vs Benchmark", sql, tuple(params)


def build_board_query(parsed: ParsedQuestion, plan_name=None):
    q = normalize_text(parsed.raw_question)

    sql = """
    SELECT name, role, appointment_type, professional_background,
           tenure_years, committee_memberships, known_priorities,
           finance_expertise_level, typical_question_focus, notes
    FROM board_members
    WHERE 1=1
    """
    params = []

    if "esg" in q or "climate" in q or "sustainability" in q:
        sql += " AND lower(known_priorities) LIKE ?"
        params.append("%esg%")

    sql += """
    ORDER BY
        CASE role
            WHEN 'President' THEN 1
            WHEN 'Vice President' THEN 2
            ELSE 3
        END,
        tenure_years DESC
    """
    return "Board Members", sql, tuple(params)


def build_competitor_query(parsed: ParsedQuestion, plan_name=None):
    sql = """
    SELECT firm_name, is_current_calpers_manager, estimated_calpers_aum_billions,
           known_fee_range_bps, esg_rating, threat_level,
           known_strengths, known_weaknesses, strategy_focus
    FROM competitive_intelligence
    WHERE 1=1
    """
    params = []

    if parsed.strategy:
        sql += " AND lower(strategy_focus) LIKE ?"
        params.append(f"%{parsed.strategy}%")

    sql += """
    ORDER BY
        CASE threat_level
            WHEN 'High' THEN 1
            WHEN 'Medium' THEN 2
            ELSE 3
        END,
        firm_name
    """
    return "Competitive Landscape", sql, tuple(params)


def build_calendar_query(parsed: ParsedQuestion, plan_name=None):
    sql = """
    SELECT k.event_type, k.event_name, k.date, k.priority,
           k.description, k.action_required, k.action_description,
           m.manager_name, m.strategy_type
    FROM key_dates k
    LEFT JOIN manager_roster m ON k.related_manager_id = m.manager_id
    WHERE 1=1
    """
    params = []

    if parsed.strategy:
        sql += " AND (lower(m.strategy_type) = ? OR lower(k.event_name) LIKE ? OR lower(k.description) LIKE ?)"
        params.extend([parsed.strategy, f"%{parsed.strategy}%", f"%{parsed.strategy}%"])

    if parsed.manager:
        manager_norm = normalize_text(parsed.manager)
        sql += " AND (lower(m.manager_name) LIKE ? OR lower(k.event_name) LIKE ? OR lower(k.description) LIKE ?)"
        params.extend([f"%{manager_norm}%", f"%{manager_norm}%", f"%{manager_norm}%"])

    if parsed.time_scope == "upcoming":
        sql += " AND k.date >= date('now')"

    sql += " ORDER BY k.date ASC LIMIT 20"
    return "Key Dates and Calendar", sql, tuple(params)


def build_meeting_history_query(parsed: ParsedQuestion, plan_name=None):
    sql = """
    SELECT meeting_date, meeting_type, strategy_discussed,
           outcome, sentiment_score, board_concerns_raised,
           action_items, internal_notes, location, apex_attendees
    FROM past_meeting_notes
    WHERE 1=1
    """
    params = []

    if parsed.strategy:
        sql += " AND lower(strategy_discussed) LIKE ?"
        params.append(f"%{parsed.strategy}%")

    if parsed.manager:
        manager_norm = normalize_text(parsed.manager)
        sql += """
        AND (
            lower(internal_notes) LIKE ?
            OR lower(apex_attendees) LIKE ?
            OR lower(strategy_discussed) LIKE ?
        )
        """
        params.extend([f"%{manager_norm}%", f"%{manager_norm}%", f"%{manager_norm}%"])

    sql += " ORDER BY meeting_date DESC"
    return "Past Meeting Notes", sql, tuple(params)


ROUTES = [
    {
        "label": "Plan Financials",
        "keywords": QUESTION_ALIASES["funded_status"] + QUESTION_ALIASES["aum"],
        "builder": build_plan_financials_query,
    },
    {
        "label": "Manager Roster",
        "keywords": QUESTION_ALIASES["roster"],
        "builder": build_manager_roster_query,
    },
    {
        "label": "Managers Under Review or Probation",
        "keywords": QUESTION_ALIASES["watchlist"],
        "builder": build_watchlist_query,
    },
    {
        "label": "Contract Renewals",
        "keywords": QUESTION_ALIASES["renewal"],
        "builder": build_renewal_query,
    },
    {
        "label": "Fee Structure",
        "keywords": QUESTION_ALIASES["fees"],
        "builder": build_fee_query,
    },
    {
        "label": "Performance vs Benchmark",
        "keywords": QUESTION_ALIASES["performance"],
        "builder": build_performance_query,
    },
    {
        "label": "Board Members",
        "keywords": QUESTION_ALIASES["board"] + QUESTION_ALIASES["esg"],
        "builder": build_board_query,
    },
    {
        "label": "Competitive Landscape",
        "keywords": QUESTION_ALIASES["competitor"],
        "builder": build_competitor_query,
    },
    {
        "label": "Key Dates and Calendar",
        "keywords": QUESTION_ALIASES["calendar"],
        "builder": build_calendar_query,
    },
    {
        "label": "Past Meeting Notes",
        "keywords": QUESTION_ALIASES["meeting_history"],
        "builder": build_meeting_history_query,
    },
]


def rank_routes(question: str):
    ranked = []
    for route in ROUTES:
        score = score_keywords(question, route["keywords"])
        if score > 0:
            ranked.append((route, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def select_routes(question: str, allow_multi_route: bool = True):
    ranked = rank_routes(question)
    if not ranked:
        return []

    if not allow_multi_route:
        return [ranked[0][0]]

    selected = [ranked[0][0]]
    if len(ranked) > 1:
        best_score = ranked[0][1]
        second_score = ranked[1][1]

        if second_score >= max(2, int(best_score * 0.6)):
            selected.append(ranked[1][0])

    return selected


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    deduped = []

    for row in rows:
        key = tuple(sorted(row.items()))
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return deduped


def merge_answer_records(question: str, records: list[AnswerRecord]) -> AnswerRecord:
    if not records:
        return AnswerRecord(
            question=question,
            label=f"Unanswered: {question[:60]}",
            data=[],
            source="not_found",
            used_llm_fallback=False,
            sql=None,
        )

    merged_rows = []
    labels = []
    sources = []
    sql_parts = []
    used_llm = False

    for record in records:
        merged_rows.extend(record.data)
        labels.append(record.label)
        sources.append(record.source)
        if record.sql:
            sql_parts.append(record.sql)
        used_llm = used_llm or record.used_llm_fallback

    merged_rows = dedupe_rows(merged_rows)

    return AnswerRecord(
        question=question,
        label=" + ".join(dict.fromkeys(labels)),
        data=merged_rows,
        source="multi_route" if len(records) > 1 else sources[0],
        used_llm_fallback=used_llm,
        sql="\n-- NEXT QUERY --\n".join(sql_parts) if sql_parts else None,
    )


def llama_generate_sql(question: str, plan_name: str = None, manager_name: str = None) -> str:
    context_lines = []
    if plan_name:
        context_lines.append(f"Plan name: {plan_name}")
    if manager_name:
        context_lines.append(f"Manager name: {manager_name}")

    context_block = "\n".join(context_lines) if context_lines else "No extra context."

    prompt = f"""You are a SQL expert. Given the following database schema and question,
write a single SQLite SELECT query to answer the question.

{DB_SCHEMA}

Context:
{context_block}

Question: {question}

Rules:
- Write only one SELECT statement
- Use only tables and columns that exist in the schema above
- Use SQLite syntax only
- Do not use DROP, DELETE, INSERT, UPDATE, ALTER, CREATE, ATTACH, DETACH, PRAGMA, WITH, comments, or multiple statements
- Prefer exact existing columns over invented ones
- Return only the SQL query with no explanation and no markdown

SQL:"""

    try:
        raw = ollama_generate(prompt, timeout=60)
        sql = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip()
        sql = sql.split(";")[0].strip()
        if sql:
            sql += ";"
        return sql
    except Exception as e:
        log("Agent 2", f"Llama SQL generation failed: {repr(e)}")
        return ""


class StructuredDataAgent:
    def __init__(self, db_path: Path = DB_PATH, allow_multi_route: bool = True):
        self.db_path = db_path
        self.allow_multi_route = allow_multi_route

    def answer_via_routes(self, question: str, plan_name: str = None, manager_name: str = None) -> list[AnswerRecord]:
        parsed = parse_question(question, manager_name=manager_name)
        log("Agent 2", f"Parsed question: {parsed}")

        routes = select_routes(question, allow_multi_route=self.allow_multi_route)
        if not routes:
            return []

        results = []
        for route in routes:
            label, sql, params = route["builder"](parsed, plan_name=plan_name)
            data = run_query(self.db_path, sql, params)
            log("Agent 2", f"Route match: {label} rows={len(data)}")
            results.append(
                AnswerRecord(
                    question=question,
                    label=label,
                    data=data,
                    source="rule_based",
                    used_llm_fallback=False,
                    sql=sql,
                )
            )

        return results

    def answer_question(self, question: str, plan_name: str = None, manager_name: str = None) -> AnswerRecord:
        route_records = self.answer_via_routes(question, plan_name=plan_name, manager_name=manager_name)
        if route_records:
            return merge_answer_records(question, route_records)

        log("Agent 2", f"No route match, trying LLM fallback for question='{question[:80]}'")
        sql = llama_generate_sql(question, plan_name=plan_name, manager_name=manager_name)

        if sql and validate_sql(sql):
            data = run_query(self.db_path, sql)
            log("Agent 2", f"LLM SQL rows={len(data)}")
            return AnswerRecord(
                question=question,
                label=f"Generated Query: {question[:60]}",
                data=data,
                source="llm_generated",
                used_llm_fallback=True,
                sql=sql,
            )

        log("Agent 2", f"Could not answer question='{question[:80]}'")
        return AnswerRecord(
            question=question,
            label=f"Unanswered: {question[:60]}",
            data=[],
            source="not_found",
            used_llm_fallback=False,
            sql=None,
        )

    def run(self, questions: list, plan_name: str, manager_name: str = None) -> StructuredDataResult:
        log("Agent 2", f"Running structured data questions count={len(questions)}")

        answers = []
        for i, question in enumerate(questions, 1):
            log("Agent 2", f"Question {i}/{len(questions)}: {question[:100]}")
            answers.append(self.answer_question(question, plan_name=plan_name, manager_name=manager_name))

        answered = sum(1 for a in answers if not a.is_empty())
        unanswered = sum(1 for a in answers if a.is_empty())
        llm_used = sum(1 for a in answers if a.used_llm_fallback)

        log("Agent 2", f"Complete answered={answered} unanswered={unanswered} llm_fallback_used={llm_used}")

        return StructuredDataResult(
            plan_name=plan_name,
            manager_name=manager_name,
            answers=answers,
        )


if __name__ == "__main__":
    test_questions = [
        "What are the funded status and AUM levels?",
        "Which managers currently cover Core Fixed Income?",
        "What are the current fees for the Core Fixed Income strategy?",
        "How does our performance track record compare to the benchmark?",
        "What is the current renewal timeline for the Core Fixed Income mandate, and are there any watchlist flags?",
        "What are the board members' known priorities and ESG focus areas?",
        "Who are the main competitors in Core Fixed Income?",
        "What upcoming dates or deadlines matter for this strategy?",
        "What do prior meetings suggest about board concerns for our strategy?",
    ]

    agent = StructuredDataAgent(db_path=DB_PATH, allow_multi_route=True)
    result = agent.run(
        questions=test_questions,
        plan_name="CalPERS",
        manager_name=None,
    )

    print(result)
    print()
    print(result.to_context_string(max_rows_per_answer=3)[:3000])