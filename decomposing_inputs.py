import json
import re
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


def log(agent, message):
    print(f"LOG: {agent}: {message}")


BASE_FALLBACK_QUERIES = {
    "structured_data": [
        "What is the plan's current funded status and AUM?",
        "Which managers currently cover this strategy for the plan?",
        "Are any managers in this strategy on probation or under review?",
        "What are the relevant contract renewal dates for this strategy?",
        "What fees are current managers charging for similar mandates?",
        "What benchmark is used for this strategy?",
    ],
    "document_rag": [
        "What has the board said about this strategy recently?",
        "What concerns has the board raised about this strategy?",
        "What ESG expectations has the board expressed for this strategy?",
        "Has the plan discussed any searches, reviews, or RFPs for this strategy?",
        "What does the investment policy say about this strategy?",
        "What did the most recent investment committee materials say about this strategy?",
    ],
    "market_intelligence": [
        "What is the current macro outlook for this strategy?",
        "What are the main risks in the current rate and spread environment?",
        "How are central bank policies affecting this strategy?",
        "What market conditions are most relevant to pitching this strategy now?",
    ],
    "news": [
        "recent pension fund news",
        "fixed income market news this week",
    ],
}


MEETING_TYPE_INSTRUCTIONS = {
    "RFP Finalist": {
        "goal": (
            "This is a competitive selection process for a potential new mandate. "
            "The manager is trying to win business."
        ),
        "structured_data_focus": [
            "current roster in the relevant strategy",
            "incumbent manager fees and performance",
            "board member priorities and decision-maker profiles",
            "evaluation criteria and weighting if available",
            "upcoming renewal, search, or consultant review timelines",
        ],
        "document_rag_focus": [
            "board dissatisfaction or unmet needs in the current program",
            "language showing what the board wants changed or improved",
            "recent discussion of ESG expectations in this strategy",
            "consultant recommendations about manager structure or roster changes",
            "signals about why a search was opened or why an incumbent may be vulnerable",
        ],
        "market_focus": [
            "macro arguments that support the manager's strategy",
            "current market risks the board is likely to ask about",
            "how the strategy fits the plan's needs in the current environment",
        ],
        "news_focus": [
            "recent news on the pension plan",
            "recent news on the relevant strategy or market segment",
            "recent news affecting public pensions, governance, or fixed income mandates",
        ],
        "question_style": (
            "Questions should identify competitive angles, board pain points, and decision criteria."
        ),
        "avoid": (
            "Do not focus on defending historical relationship issues unless they are directly relevant to the search."
        ),
    },
    "Annual Review": {
        "goal": (
            "This is a mandate defense meeting. The manager is trying to retain or strengthen an existing relationship."
        ),
        "structured_data_focus": [
            "the manager's performance versus benchmark across relevant periods",
            "alpha, tracking error, information ratio, peer ranking, and fees",
            "renewal timeline and any watchlist, review, or probation flags",
            "how the manager compares with peer managers in the same strategy",
            "which board members are most likely to scrutinize performance, fees, ESG, or risk",
        ],
        "document_rag_focus": [
            "recent board comments about the manager, the strategy, or the program",
            "specific concerns raised in prior meetings",
            "changes in tone toward the strategy or the mandate over time",
            "ESG criticism, fee pressure, duration or risk concerns, and consultant commentary",
            "language suggesting what the manager must address to maintain board confidence",
        ],
        "market_focus": [
            "current market conditions affecting evaluation of recent performance",
            "how macro conditions may explain or challenge recent results",
            "arguments the board may expect the manager to address in defending positioning",
        ],
        "news_focus": [
            "recent plan news relevant to governance or investment oversight",
            "recent strategy-specific market developments that could affect the review discussion",
        ],
        "question_style": (
            "Questions should diagnose risk to the relationship, identify criticism, and surface what must be answered clearly."
        ),
        "avoid": (
            "Do not frame questions mainly as new-business positioning unless there is evidence the mandate is effectively being re-competed."
        ),
    },
    "First Introduction": {
        "goal": (
            "This is an exploratory relationship-building meeting. The manager is trying to understand the plan and become relevant."
        ),
        "structured_data_focus": [
            "current roster and coverage by strategy",
            "gaps or concentrations in the current manager lineup",
            "board and consultant roles",
            "who influences hiring decisions and how searches are typically run",
            "what timelines or triggers usually lead to new mandates",
        ],
        "document_rag_focus": [
            "board priorities and recurring concerns",
            "themes the board discusses repeatedly in this strategy",
            "how the plan talks about diversification, ESG, risk, fees, and structure",
            "consultant recommendations and long-term portfolio direction",
        ],
        "market_focus": [
            "current market topics likely to be top of mind for the board",
            "macro themes that make the strategy more or less relevant to the plan",
        ],
        "news_focus": [
            "recent news about the plan",
            "recent governance or strategy developments relevant to relationship building",
        ],
        "question_style": (
            "Questions should emphasize discovery, relevance, path-to-mandate, and relationship mapping."
        ),
        "avoid": (
            "Do not assume an active search or an existing relationship unless context indicates it."
        ),
    },
}


class MeetingContext:
    def __init__(
        self,
        plan_name,
        strategy,
        meeting_type,
        meeting_date,
        manager_name=None,
        additional_notes=None,
    ):
        self.plan_name = plan_name
        self.strategy = strategy
        self.meeting_type = meeting_type
        self.meeting_date = meeting_date
        self.manager_name = manager_name
        self.additional_notes = additional_notes

    def __repr__(self):
        return (
            f"MeetingContext(plan_name={self.plan_name!r}, "
            f"strategy={self.strategy!r}, "
            f"meeting_type={self.meeting_type!r}, "
            f"meeting_date={self.meeting_date!r}, "
            f"manager_name={self.manager_name!r})"
        )


class DecomposedQuery:
    def __init__(
        self,
        plan_name,
        strategy,
        meeting_type,
        meeting_date,
        manager_name=None,
        structured_data=None,
        document_rag=None,
        market_intelligence=None,
        news=None,
        used_fallback=False,
    ):
        self.plan_name = plan_name
        self.strategy = strategy
        self.meeting_type = meeting_type
        self.meeting_date = meeting_date
        self.manager_name = manager_name
        self.structured_data = structured_data or []
        self.document_rag = document_rag or []
        self.market_intelligence = market_intelligence or []
        self.news = news or []
        self.used_fallback = used_fallback

    def __repr__(self):
        return (
            f"DecomposedQuery(plan_name={self.plan_name!r}, "
            f"strategy={self.strategy!r}, "
            f"total_queries={self.total_queries()}, "
            f"used_fallback={self.used_fallback})"
        )

    def all_queries(self):
        return {
            "structured_data": self.structured_data,
            "document_rag": self.document_rag,
            "market_intelligence": self.market_intelligence,
            "news": self.news,
        }

    def total_queries(self):
        return sum(len(q) for q in self.all_queries().values())


def ollama_generate(prompt: str, timeout: int = 120) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


def extract_json(raw: str):
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def validate_structure(parsed: dict) -> bool:
    required_keys = {"structured_data", "document_rag", "market_intelligence", "news"}

    if not isinstance(parsed, dict):
        return False

    if not required_keys.issubset(parsed.keys()):
        return False

    for key in required_keys:
        value = parsed[key]
        if not isinstance(value, list) or len(value) == 0:
            return False
        if not all(isinstance(q, str) and q.strip() for q in value):
            return False

    return True


def format_focus_list(items):
    return "\n".join(f"- {item}" for item in items)


def get_meeting_type_guidance(context: MeetingContext) -> dict:
    default_guidance = {
        "goal": "Help the manager prepare comprehensively for the meeting.",
        "structured_data_focus": [
            "plan financials",
            "manager roster",
            "fees",
            "renewal timelines",
            "board member priorities",
        ],
        "document_rag_focus": [
            "recent board discussion",
            "investment policy language",
            "RFP or search activity",
            "consultant recommendations",
        ],
        "market_focus": [
            "current macro environment",
            "strategy-relevant market conditions",
            "current risk themes",
        ],
        "news_focus": [
            "recent plan news",
            "recent strategy news",
        ],
        "question_style": "Questions should be specific, decision-useful, and meeting-relevant.",
        "avoid": "Avoid vague or generic research questions.",
    }
    return MEETING_TYPE_INSTRUCTIONS.get(context.meeting_type, default_guidance)


def build_prompt(context: MeetingContext) -> str:
    manager_name = context.manager_name or "the manager"
    guidance = get_meeting_type_guidance(context)

    notes_section = (
        f"\nAdditional manager context:\n{context.additional_notes}"
        if context.additional_notes
        else ""
    )

    return f"""You are helping an institutional investment manager prepare for a meeting with a public pension plan.

Meeting details:
- Pension plan: {context.plan_name}
- Strategy: {context.strategy}
- Meeting type: {context.meeting_type}
- Meeting date: {context.meeting_date}
- Manager: {manager_name}{notes_section}

Primary objective:
{guidance["goal"]}

You must generate research sub-questions for EXACTLY these four categories:

1. structured_data
Questions answerable from structured records such as:
- plan financials
- funded status
- AUM
- manager roster
- fees
- performance metrics
- contract renewal dates
- board member roles and priorities

For this meeting, prioritize:
{format_focus_list(guidance["structured_data_focus"])}

2. document_rag
Questions answerable from unstructured plan materials such as:
- board minutes
- investment committee transcripts
- annual reports
- investment policy statements
- RFPs
- consultant materials

For this meeting, prioritize:
{format_focus_list(guidance["document_rag_focus"])}

3. market_intelligence
Questions about current market and macro conditions relevant to the strategy.

For this meeting, prioritize:
{format_focus_list(guidance["market_focus"])}

4. news
Short search queries for recent developments relevant to the pension plan, the strategy, or the meeting context.

For this meeting, prioritize:
{format_focus_list(guidance["news_focus"])}

Question style requirements:
- {guidance["question_style"]}
- Make the questions specific to {context.plan_name} and {context.strategy}
- Make the questions useful for this exact meeting type, not generic research
- Prefer questions that reveal decision criteria, current concerns, board sentiment, manager risk, or competitive positioning
- For structured_data, document_rag, and market_intelligence: use full questions
- For news: use short search queries, not full questions
- Generate 4 to 6 items per category
- Do not use placeholders
- Do not mention that information may be unavailable
- Do not hardcode any firm names except the manager name provided in the meeting details when directly relevant

Avoid:
{guidance["avoid"]}

Respond ONLY with a valid JSON object in exactly this format:

{{
  "structured_data": [
    "question 1",
    "question 2"
  ],
  "document_rag": [
    "question 1",
    "question 2"
  ],
  "market_intelligence": [
    "question 1",
    "question 2"
  ],
  "news": [
    "search query 1",
    "search query 2"
  ]
}}"""


def build_fallback_queries(context: MeetingContext) -> dict:
    plan = context.plan_name
    strategy = context.strategy
    manager = context.manager_name or "the current manager"
    meeting_type = context.meeting_type

    if meeting_type == "RFP Finalist":
        return {
            "structured_data": [
                f"Which managers currently run {strategy} for {plan}?",
                f"What fees are current {strategy} managers charging at {plan}?",
                f"How have incumbent {strategy} managers performed versus benchmark at {plan}?",
                f"Which board members and consultants are most influential in selecting {strategy} managers at {plan}?",
                f"What evaluation criteria or scoring factors are used for manager selection at {plan}?",
            ],
            "document_rag": [
                f"What has the board recently said about {strategy} at {plan}?",
                f"What concerns has the board raised about the current {strategy} program at {plan}?",
                f"Has {plan} discussed replacing or adding managers in {strategy}?",
                f"What ESG requirements has {plan} expressed for {strategy} mandates?",
                f"What consultant recommendations has {plan} received about its {strategy} manager structure?",
            ],
            "market_intelligence": [
                f"What current macro conditions are most relevant to pitching {strategy} today?",
                f"What risks in rates, spreads, or liquidity are most likely to matter in a {strategy} pitch right now?",
                f"What market arguments best support an active {strategy} mandate in the current environment?",
                f"What current fixed income themes are most relevant to public pension buyers of {strategy}?",
            ],
            "news": [
                f"{plan} recent investment news",
                f"{plan} manager search news",
                f"{strategy} market news",
                f"public pension fixed income mandate news",
            ],
        }

    if meeting_type == "Annual Review":
        return {
            "structured_data": [
                f"How has {manager} performed versus benchmark for {strategy} at {plan} across recent periods?",
                f"What fees is {manager} charging relative to peer managers in {strategy} at {plan}?",
                f"Is {manager} under review, on watchlist, or near renewal for {strategy} at {plan}?",
                f"How does {manager}'s risk-adjusted performance compare with peer {strategy} managers at {plan}?",
                f"Which board members are most likely to focus on performance, fees, ESG, or duration risk in this review?",
            ],
            "document_rag": [
                f"What has the board recently said about {strategy} or manager performance at {plan}?",
                f"What concerns has the board raised that could affect confidence in {manager}?",
                f"How has board tone toward {strategy} changed over recent meetings at {plan}?",
                f"What ESG, fee, or risk concerns have appeared in recent committee materials for {strategy} at {plan}?",
                f"What consultant comments or staff recommendations could affect {manager}'s standing at {plan}?",
            ],
            "market_intelligence": [
                f"What macro conditions are most relevant to evaluating recent {strategy} performance right now?",
                f"What market developments could explain or challenge recent positioning in {strategy}?",
                f"What current board-level concerns in fixed income markets are most likely to come up in an annual review?",
                f"What current duration, spread, or policy risks are most relevant to defending {strategy} results?",
            ],
            "news": [
                f"{plan} recent investment committee news",
                f"{plan} governance news",
                f"{strategy} market news",
                f"public pension fixed income review news",
            ],
        }

    if meeting_type == "First Introduction":
        return {
            "structured_data": [
                f"Which managers currently cover {strategy} for {plan}?",
                f"Are there gaps, concentrations, or recent changes in the {strategy} roster at {plan}?",
                f"Who are the key board members, staff, and consultants involved in manager selection at {plan}?",
                f"What typically triggers a manager search or new mandate in {strategy} at {plan}?",
                f"What are the current funded status, AUM, and portfolio priorities that shape demand for {strategy} at {plan}?",
            ],
            "document_rag": [
                f"What recurring themes has the board discussed about {strategy} at {plan}?",
                f"What priorities has the board expressed around ESG, fees, diversification, or risk in {strategy}?",
                f"What consultant recommendations suggest future openings or needs in {strategy} at {plan}?",
                f"What does the investment policy suggest about how {strategy} fits within the portfolio at {plan}?",
                f"What has the board recently said that reveals pain points or unmet needs relevant to {strategy}?",
            ],
            "market_intelligence": [
                f"What current market themes make {strategy} more relevant to public pension plans right now?",
                f"What macro conditions are likely to be top of mind for a first conversation about {strategy}?",
                f"What fixed income topics would help establish credibility in an introductory meeting on {strategy}?",
                f"What market risks or opportunities should be referenced to make {strategy} relevant today?",
            ],
            "news": [
                f"{plan} recent news",
                f"{plan} investment committee news",
                f"{strategy} market news",
                f"public pension fixed income news",
            ],
        }

    return {
        "structured_data": [
            f"What is the current funded status and AUM of {plan}?",
            f"Which managers currently cover {strategy} for {plan}?",
            f"Are any managers in {strategy} under review or on probation at {plan}?",
            f"What fees are current managers charging for similar {strategy} mandates at {plan}?",
            f"What benchmark and performance metrics are used for {strategy} at {plan}?",
        ],
        "document_rag": [
            f"What has the board said recently about {strategy} at {plan}?",
            f"What concerns has the board raised about {strategy} at {plan}?",
            f"What ESG expectations has {plan} expressed for {strategy} mandates?",
            f"Has {plan} discussed any RFPs, reviews, or searches related to {strategy}?",
            f"What do recent committee materials say about {strategy} at {plan}?",
        ],
        "market_intelligence": [
            f"What is the current macro outlook most relevant to {strategy}?",
            f"What are the main market risks affecting {strategy} right now?",
            f"How are rate, spread, and policy conditions affecting {strategy} today?",
            f"What market developments are most relevant to preparing for this {strategy} meeting?",
        ],
        "news": [
            f"{plan} recent news",
            f"{plan} investment committee",
            f"{strategy} market news",
        ],
    }


class QueryDecompositionAgent:
    def run(self, context: MeetingContext) -> DecomposedQuery:
        log("Agent 1", f"Decomposing query for {context.plan_name} - {context.strategy}")
        log("Agent 1", f"Meeting type: {context.meeting_type} on {context.meeting_date}")

        parsed = None
        used_fallback = False

        try:
            prompt = build_prompt(context)
            raw = ollama_generate(prompt)
            parsed = extract_json(raw)

            if parsed is None:
                log("Agent 1", "Could not extract JSON from model response. Using fallback.")
                used_fallback = True
            elif not validate_structure(parsed):
                log("Agent 1", "Model returned invalid JSON structure. Using fallback.")
                used_fallback = True
            else:
                total = sum(len(v) for v in parsed.values())
                log("Agent 1", f"Successfully decomposed into {total} sub-questions.")

        except requests.exceptions.ConnectionError:
            log("Agent 1", "Cannot connect to Ollama. Using fallback.")
            used_fallback = True
        except Exception as e:
            log("Agent 1", f"Error: {repr(e)}. Using fallback.")
            used_fallback = True

        if used_fallback or parsed is None:
            parsed = build_fallback_queries(context)
            log("Agent 1", f"Built meeting-type-specific fallback with {sum(len(v) for v in parsed.values())} queries.")

        return DecomposedQuery(
            plan_name=context.plan_name,
            strategy=context.strategy,
            meeting_type=context.meeting_type,
            meeting_date=context.meeting_date,
            manager_name=context.manager_name,
            structured_data=parsed["structured_data"],
            document_rag=parsed["document_rag"],
            market_intelligence=parsed["market_intelligence"],
            news=parsed["news"],
            used_fallback=used_fallback,
        )


if __name__ == "__main__":
    agent = QueryDecompositionAgent()

    context = MeetingContext(
        plan_name="CalPERS",
        strategy="Core Fixed Income",
        meeting_type="RFP Finalist",
        meeting_date="2026-03-15",
        manager_name="Apex Capital Management",
        additional_notes="We have managed this mandate for 3 years. Board has raised fee concerns.",
    )

    result = agent.run(context)

    print("Decomposed Query")
    print(f"Plan: {result.plan_name}")
    print(f"Strategy: {result.strategy}")
    print(f"Manager: {result.manager_name}")
    print(f"Meeting: {result.meeting_type} on {result.meeting_date}")
    print(f"Fallback: {result.used_fallback}")
    print(f"Total queries: {result.total_queries()}")

    for category, questions in result.all_queries().items():
        print(f"\n{category.upper()}:")
        for q in questions:
            print(f"- {q}")