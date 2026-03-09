import json
import re
from datetime import date
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

MAX_SECTION_CONTEXT_CHARS = 5000
MAX_INSIGHT_CONTEXT_CHARS = 7000

MAX_STRUCTURED_ROWS = 3
MAX_RAG_CHUNKS = 5
MAX_MARKET_CHUNKS = 5
MAX_NEWS_ITEMS = 5

ALLOWED_DATA_SOURCES = {"structured", "rag", "market", "news"}


def log(agent, message):
    print(f"LOG: {agent}: {message}")


def ollama_generate(prompt: str, timeout: int = 180) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


def extract_json(raw: str):
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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_key(text: str) -> str:
    return normalize_text(text).lower()


def sanitize_inline_html(html: str) -> str:
    html = re.sub(r"```(?:html)?", "", html, flags=re.IGNORECASE).strip().rstrip("`").strip()
    html = re.sub(r"</?(html|head|body|style|script)[^>]*>", "", html, flags=re.IGNORECASE)

    allowed_tags = {"p", "ul", "li", "strong", "em", "div", "br"}
    tag_pattern = re.compile(r"</?([a-zA-Z0-9]+)(?:\s[^>]*)?>")

    def replace_tag(match):
        tag = match.group(1).lower()
        if tag in allowed_tags:
            return match.group(0)
        return ""

    html = tag_pattern.sub(replace_tag, html)
    html = re.sub(r"\n{3,}", "\n\n", html).strip()

    if not html:
        return ""

    if not html.startswith("<"):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", html) if p.strip()]
        html = "\n".join(f"<p>{p}</p>" for p in paragraphs)

    return html.strip()


def dedupe_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        key = normalize_key(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def safe_json_like(value):
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return ", ".join(str(x) for x in parsed)
        except Exception:
            return value
    return value


def fallback_section_html(title: str, purpose: str, context_text: str) -> str:
    context_text = normalize_text(context_text)[:1200] if context_text else ""
    if not context_text:
        return (
            f"<p><strong>{title}.</strong> "
            f"This section could not be generated automatically. "
            f"Purpose: {purpose}</p>"
        )
    return (
        f"<p><strong>{title}.</strong> "
        f"This section could not be fully generated automatically. "
        f"Purpose: {purpose}</p>"
        f"<p>{context_text}</p>"
    )


class BriefingSection:
    def __init__(self, title, purpose, content_html, data_sources_used):
        self.title = title
        self.purpose = purpose
        self.content_html = content_html
        self.data_sources_used = data_sources_used

    def __repr__(self):
        return f"BriefingSection(title={self.title!r})"


class BriefingResult:
    def __init__(self, meeting_context, sections, proactive_insights_html, html_path, pdf_path):
        self.meeting_context = meeting_context
        self.sections = sections
        self.proactive_insights_html = proactive_insights_html
        self.html_path = html_path
        self.pdf_path = pdf_path

    def __repr__(self):
        return f"BriefingResult(sections={len(self.sections)}, pdf={self.pdf_path!r})"


def build_structured_context(structured_result) -> str:
    if not structured_result:
        return ""

    lines = []
    for answer in structured_result.answers:
        if answer.is_empty():
            continue

        lines.append(f"[STRUCTURED LABEL: {answer.label}]")
        lines.append(f"[STRUCTURED QUESTION: {answer.question}]")

        for row in answer.data[:MAX_STRUCTURED_ROWS]:
            field_lines = []
            for k, v in list(row.items())[:8]:
                v = safe_json_like(v)
                field_lines.append(f"{k}: {v}")
            row_blob = " | ".join(field_lines)
            lines.append(f"[SOURCE: Structured / {answer.label}] {row_blob}")

        lines.append("")

    return "\n".join(dedupe_preserve_order(lines))


def build_rag_context(rag_result) -> str:
    if not rag_result:
        return ""

    lines = []

    topic_shifts = getattr(rag_result, "topic_shift_signals", []) or getattr(rag_result, "conflict_signals", [])
    if topic_shifts:
        lines.append("TOPIC SHIFT SIGNALS:")
        for signal in topic_shifts[:3]:
            lines.append(str(signal))
        lines.append("")

    chunk_count = 0
    for qr in rag_result.question_results:
        lines.append(f"[RAG QUESTION: {qr.question}]")
        for chunk in qr.chunks[:2]:
            meta = chunk.metadata
            source = meta.get("source_file", "?")
            date_ = meta.get("date", "?")
            tone = meta.get("tone", "?")
            page = meta.get("page_number", "?")
            source_type = meta.get("source_type", "?")
            lines.append(
                f"[SOURCE: RAG / {source} / p{page} / {date_} / tone={tone} / source_type={source_type}] "
                f"{chunk.text[:420]}"
            )
            chunk_count += 1
            if chunk_count >= MAX_RAG_CHUNKS:
                break
        lines.append("")
        if chunk_count >= MAX_RAG_CHUNKS:
            break

    return "\n".join(dedupe_preserve_order(lines))


def build_market_context(market_result) -> str:
    if not market_result:
        return ""

    lines = []
    chunk_count = 0

    for qr in market_result.question_results:
        lines.append(f"[MARKET QUESTION: {qr.question}]")
        for chunk in qr.chunks[:2]:
            meta = chunk.metadata
            source = meta.get("source_file", "?")
            date_ = meta.get("date", "?")
            page = meta.get("page_number", "?")
            conf = getattr(chunk, "confidence_score", None)
            conf_str = f" confidence={conf:.3f}" if conf is not None else ""
            lines.append(
                f"[SOURCE: Market / {source} / p{page} / {date_}{conf_str}] "
                f"{chunk.text[:420]}"
            )
            chunk_count += 1
            if chunk_count >= MAX_MARKET_CHUNKS:
                break
        lines.append("")
        if chunk_count >= MAX_MARKET_CHUNKS:
            break

    return "\n".join(dedupe_preserve_order(lines))


def build_news_context(news_result) -> str:
    if not news_result:
        return ""

    lines = []
    item_count = 0

    for qr in news_result.question_results:
        lines.append(f"[NEWS QUESTION: {qr.question}]")
        for item in qr.items[:2]:
            score = getattr(item, "final_score", None)
            score_str = f" final_score={score:.3f}" if score is not None else ""
            lines.append(
                f"[SOURCE: News / {item.source} / {item.published_date}{score_str}] "
                f"{item.title} — {item.summary[:320]}"
            )
            item_count += 1
            if item_count >= MAX_NEWS_ITEMS:
                break
        lines.append("")
        if item_count >= MAX_NEWS_ITEMS:
            break

    return "\n".join(dedupe_preserve_order(lines))


def assemble_section_context(section, structured_result, rag_result, market_result, news_result) -> str:
    parts = []
    data_sources = section.get("data_sources", [])

    if "structured" in data_sources and structured_result:
        ctx = build_structured_context(structured_result)
        if ctx:
            parts.append("=== STRUCTURED DATA ===\n" + ctx)

    if "rag" in data_sources and rag_result:
        ctx = build_rag_context(rag_result)
        if ctx:
            parts.append("=== DOCUMENT INTELLIGENCE ===\n" + ctx)

    if "market" in data_sources and market_result:
        ctx = build_market_context(market_result)
        if ctx:
            parts.append("=== MARKET RESEARCH ===\n" + ctx)

    if "news" in data_sources and news_result:
        ctx = build_news_context(news_result)
        if ctx:
            parts.append("=== RECENT NEWS ===\n" + ctx)

    context = "\n\n".join(parts)
    return context[:MAX_SECTION_CONTEXT_CHARS]


def build_data_summary(meeting_context, structured_result, rag_result, market_result, news_result) -> str:
    lines = [
        f"- Plan: {meeting_context.plan_name}",
        f"- Strategy: {meeting_context.strategy}",
        f"- Meeting Type: {meeting_context.meeting_type}",
    ]

    if structured_result:
        answered = sum(1 for a in structured_result.answers if not a.is_empty())
        lines.append(f"- Structured answers: {answered}")

    if rag_result:
        total_chunks = sum(len(r.chunks) for r in rag_result.question_results)
        n_signals = len(getattr(rag_result, "topic_shift_signals", []) or getattr(rag_result, "conflict_signals", []))
        lines.append(f"- Document intelligence chunks: {total_chunks}")
        lines.append(f"- Topic shift signals: {n_signals}")

    if market_result:
        total_chunks = sum(len(r.chunks) for r in market_result.question_results)
        lines.append(f"- Market research chunks: {total_chunks}")

    if news_result:
        total_items = sum(len(r.items) for r in news_result.question_results)
        lines.append(f"- News items: {total_items}")

    return "\n".join(lines)


def generate_toc(meeting_context, structured_result, rag_result, market_result, news_result, report_template: dict) -> list:
    log("Agent 6", "Call 1: Generating table of contents from template")

    template_json = json.dumps(report_template, indent=2)
    data_summary = build_data_summary(
        meeting_context,
        structured_result,
        rag_result,
        market_result,
        news_result,
    )

    prompt = f"""You are planning an institutional investment meeting briefing.

Meeting Context:
- Plan: {meeting_context.plan_name}
- Strategy: {meeting_context.strategy}
- Meeting Type: {meeting_context.meeting_type}
- Meeting Date: {meeting_context.meeting_date}
- Manager: {meeting_context.manager_name or "Not specified"}
- Notes: {meeting_context.additional_notes or "None"}

Available Data Summary:
{data_summary}

Reusable Report Template:
{template_json}

Task:
Use the template as the baseline structure for the briefing.

You may:
- keep sections
- remove irrelevant sections
- slightly rename sections
- add up to 2 highly relevant sections
- assign the most useful data sources to each section

Each section object must contain exactly:
- "title"
- "purpose"
- "data_sources"

Valid data_sources:
- structured
- rag
- market
- news

Rules:
- Return 5 to 9 sections
- Keep the structure professional and logically ordered
- Avoid overlapping or duplicate sections
- Use the template as the foundation
- Return ONLY a JSON array

JSON:"""

    try:
        raw = ollama_generate(prompt, timeout=120)
        parsed = extract_json(raw)
    except Exception as e:
        log("Agent 6", f"TOC generation failed error={repr(e)}")
        parsed = None

    if isinstance(parsed, list) and len(parsed) >= 3:
        cleaned = []
        seen_titles = set()

        for item in parsed:
            if not isinstance(item, dict):
                continue

            title = normalize_text(item.get("title", ""))
            purpose = normalize_text(item.get("purpose", ""))
            data_sources = item.get("data_sources", [])

            if not title or not purpose:
                continue

            pretty_title = str(item.get("title", "")).strip()
            pretty_purpose = str(item.get("purpose", "")).strip()
            valid_sources = [s for s in data_sources if s in ALLOWED_DATA_SOURCES]

            if not valid_sources:
                valid_sources = ["structured", "rag"]

            if pretty_title.lower() in seen_titles:
                continue

            seen_titles.add(pretty_title.lower())
            cleaned.append({
                "title": pretty_title,
                "purpose": pretty_purpose,
                "data_sources": valid_sources,
            })

        if len(cleaned) >= 3:
            log("Agent 6", f"TOC generated sections={len(cleaned)}")
            return cleaned

    log("Agent 6", "TOC generation failed, using template defaults")
    return [
        {
            "title": s["title"],
            "purpose": s["purpose"],
            "data_sources": s.get("preferred_data_sources", ["structured", "rag"]),
        }
        for s in report_template.get("default_sections", [])
    ]


def write_section(section, meeting_context, structured_result, rag_result, market_result, news_result, index: int, total_sections: int) -> BriefingSection:
    title = section["title"]
    purpose = section["purpose"]
    data_sources = section["data_sources"]

    log("Agent 6", f"Writing section {index + 1}/{total_sections} title={title}")

    context = assemble_section_context(
        section=section,
        structured_result=structured_result,
        rag_result=rag_result,
        market_result=market_result,
        news_result=news_result,
    )

    if not context:
        context = "No specific evidence available."

    prompt = f"""You are an expert investment management consultant writing one section of a meeting briefing.

Meeting Context:
- Plan: {meeting_context.plan_name}
- Strategy: {meeting_context.strategy}
- Meeting Type: {meeting_context.meeting_type}
- Meeting Date: {meeting_context.meeting_date}
- Manager: {meeting_context.manager_name or "Not specified"}

Section Title:
{title}

Section Purpose:
{purpose}

Assigned Data Sources:
{", ".join(data_sources)}

Evidence:
{context}

Instructions:
- Write a polished, professional section
- Use the section purpose as the organizing logic
- Use only evidence from the provided context
- Mention names, dates, figures, and concrete signals where useful
- Highlight risks, contradictions, pressure points, or opportunities when relevant
- Do not invent evidence
- Do not restate the section title as a heading
- Length: 150 to 260 words
- Include inline source references in parentheses, for example:
  (Source: Structured / Performance vs Benchmark)
  (Source: RAG / 202511-invest-transcript.pdf / p3 / 2025-11-17)
  (Source: Market / pimco_outlook.pdf / p2 / 2026-01-01)
  (Source: News / Pensions & Investments / 2026-02-22)
- Return ONLY valid HTML using <p>, <ul>, <li>, <strong>, <em> tags

HTML:"""

    try:
        raw = ollama_generate(prompt, timeout=180)
        html = sanitize_inline_html(raw)
        if not html:
            html = fallback_section_html(title, purpose, context)
    except Exception as e:
        log("Agent 6", f"Section generation failed title={title} error={repr(e)}")
        html = fallback_section_html(title, purpose, context)

    return BriefingSection(
        title=title,
        purpose=purpose,
        content_html=html,
        data_sources_used=data_sources,
    )


def generate_proactive_insights(meeting_context, structured_result, rag_result, market_result, news_result, sections: list, report_template: dict) -> str:
    log("Agent 6", "Generating proactive insights")

    covered_titles = ", ".join(s.title for s in sections)
    template_titles = ", ".join(
        [s["title"] for s in report_template.get("default_sections", [])] +
        [s["title"] for s in report_template.get("optional_sections", [])]
    )

    parts = []
    if structured_result:
        parts.append(build_structured_context(structured_result)[:1800])
    if rag_result:
        parts.append(build_rag_context(rag_result)[:2000])
    if market_result:
        parts.append(build_market_context(market_result)[:1600])
    if news_result:
        parts.append(build_news_context(news_result)[:1300])

    full_context = "\n\n".join(parts)[:MAX_INSIGHT_CONTEXT_CHARS]

    prompt = f"""You are reviewing an investment meeting briefing and surfacing additional high-value insights.

Meeting Context:
- Plan: {meeting_context.plan_name}
- Strategy: {meeting_context.strategy}
- Meeting Type: {meeting_context.meeting_type}
- Manager: {meeting_context.manager_name or "Not specified"}

Template Universe:
{template_titles}

Already Covered Sections:
{covered_titles}

Available Evidence:
{full_context}

Task:
Identify 3 to 5 proactive insights that:
- are supported by evidence above
- are not redundant with already covered sections
- would materially improve the manager's preparation
- are specific and non-generic

For each insight:
- use a short bold heading
- explain why it matters in 2 to 3 sentences
- include inline source references in parentheses

Return ONLY valid HTML using <div>, <p>, <strong>, <ul>, <li> tags.

HTML:"""

    try:
        raw = ollama_generate(prompt, timeout=180)
        html = sanitize_inline_html(raw)
        if not html:
            html = "<p><strong>No proactive insights generated.</strong></p>"
    except Exception as e:
        log("Agent 6", f"Proactive insight generation failed error={repr(e)}")
        html = "<p><strong>No proactive insights generated.</strong></p>"

    return html


def assemble_html(meeting_context, sections: list, proactive_insights_html: str, toc: list) -> str:
    today = date.today().strftime("%B %d, %Y")

    toc_items = "\n".join(
        f'<li><a href="#section-{i}">{s["title"]}</a></li>'
        for i, s in enumerate(toc)
    )

    section_html_parts = []
    for i, section in enumerate(sections):
        section_html_parts.append(f"""
        <section id="section-{i}" class="briefing-section">
            <h2>{section.title}</h2>
            <div class="section-purpose">{section.purpose}</div>
            <div class="section-body">
                {section.content_html}
            </div>
            <div class="data-sources">
                Sources used: {", ".join(section.data_sources_used)}
            </div>
        </section>
        """)

    sections_html = "\n".join(section_html_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meeting Prep Briefing - {meeting_context.plan_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: Georgia, "Times New Roman", serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #1a1a2e;
            background: #ffffff;
        }}

        .cover {{
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 80px 60px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
            color: white;
            page-break-after: always;
        }}

        .cover-label {{
            font-family: Arial, sans-serif;
            font-size: 9pt;
            font-weight: 600;
            letter-spacing: 3px;
            text-transform: uppercase;
            color: #e94560;
            margin-bottom: 24px;
        }}

        .cover-title {{
            font-size: 32pt;
            font-weight: 400;
            line-height: 1.2;
            margin-bottom: 12px;
        }}

        .cover-subtitle {{
            font-size: 16pt;
            color: #a8b2c1;
            margin-bottom: 48px;
        }}

        .cover-meta {{
            border-top: 1px solid rgba(255,255,255,0.2);
            padding-top: 32px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        .cover-meta-item label {{
            font-family: Arial, sans-serif;
            font-size: 8pt;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #e94560;
            display: block;
            margin-bottom: 4px;
        }}

        .cover-meta-item span {{
            font-size: 11pt;
            color: #e8ecf0;
        }}

        .confidential {{
            margin-top: auto;
            padding-top: 48px;
            font-family: Arial, sans-serif;
            font-size: 8pt;
            color: rgba(255,255,255,0.4);
            letter-spacing: 1px;
        }}

        .toc-page {{
            padding: 60px;
            page-break-after: always;
        }}

        .toc-page h1 {{
            font-family: Arial, sans-serif;
            font-size: 10pt;
            letter-spacing: 3px;
            text-transform: uppercase;
            color: #e94560;
            margin-bottom: 32px;
        }}

        .toc-page ol {{
            list-style: none;
            counter-reset: toc-counter;
        }}

        .toc-page ol li {{
            counter-increment: toc-counter;
            padding: 10px 0;
            border-bottom: 1px solid #e8ecf0;
            display: flex;
            align-items: baseline;
            gap: 12px;
        }}

        .toc-page ol li::before {{
            content: counter(toc-counter, decimal-leading-zero);
            font-family: Arial, sans-serif;
            font-size: 9pt;
            color: #e94560;
            min-width: 28px;
        }}

        .toc-page ol li a {{
            text-decoration: none;
            color: #1a1a2e;
            font-size: 12pt;
        }}

        .content {{
            padding: 0 60px 60px;
        }}

        .briefing-section {{
            padding: 48px 0;
            border-bottom: 1px solid #e8ecf0;
            page-break-inside: avoid;
        }}

        .briefing-section h2 {{
            font-family: Arial, sans-serif;
            font-size: 10pt;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #e94560;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e94560;
        }}

        .section-purpose {{
            font-family: Arial, sans-serif;
            font-size: 9pt;
            color: #566074;
            margin-bottom: 16px;
            font-style: italic;
        }}

        .section-body p {{
            margin-bottom: 12px;
            text-align: justify;
        }}

        .section-body ul {{
            margin: 12px 0 12px 20px;
        }}

        .section-body li {{
            margin-bottom: 6px;
        }}

        .section-body strong {{
            color: #0f3460;
        }}

        .data-sources {{
            margin-top: 16px;
            font-family: Arial, sans-serif;
            font-size: 8pt;
            color: #aab0bc;
            font-style: italic;
        }}

        .insights-section {{
            margin: 48px 0;
            padding: 40px;
            background: #f8f9ff;
            border-left: 4px solid #e94560;
            border-radius: 0 8px 8px 0;
        }}

        .insights-section h2 {{
            font-family: Arial, sans-serif;
            font-size: 10pt;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #e94560;
            margin-bottom: 24px;
        }}

        .insights-section p {{
            margin-bottom: 12px;
        }}

        .insights-section strong {{
            color: #0f3460;
        }}

        .footer {{
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid #e8ecf0;
            font-family: Arial, sans-serif;
            font-size: 8pt;
            color: #aab0bc;
            text-align: center;
        }}

        @media print {{
            .cover {{ min-height: 100vh; }}
            .briefing-section {{ page-break-inside: avoid; }}
            .insights-section {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="cover">
        <div class="cover-label">Confidential Meeting Preparation</div>
        <div class="cover-title">Intelligence Briefing</div>
        <div class="cover-subtitle">{meeting_context.plan_name}</div>
        <div class="cover-meta">
            <div class="cover-meta-item">
                <label>Strategy</label>
                <span>{meeting_context.strategy}</span>
            </div>
            <div class="cover-meta-item">
                <label>Meeting Type</label>
                <span>{meeting_context.meeting_type}</span>
            </div>
            <div class="cover-meta-item">
                <label>Meeting Date</label>
                <span>{meeting_context.meeting_date}</span>
            </div>
            <div class="cover-meta-item">
                <label>Prepared</label>
                <span>{today}</span>
            </div>
            {f'<div class="cover-meta-item"><label>Manager</label><span>{meeting_context.manager_name}</span></div>' if meeting_context.manager_name else ''}
        </div>
        <div class="confidential">
            CONFIDENTIAL - FOR INTERNAL USE ONLY
        </div>
    </div>

    <div class="toc-page">
        <h1>Contents</h1>
        <ol>
            {toc_items}
            <li><a href="#proactive-insights">Proactive Insights</a></li>
        </ol>
    </div>

    <div class="content">
        {sections_html}

        <div id="proactive-insights" class="insights-section">
            <h2>Proactive Insights</h2>
            {proactive_insights_html}
        </div>

        <div class="footer">
            Generated on {today} - Confidential - Internal Use Only
        </div>
    </div>
</body>
</html>"""


def convert_to_pdf(html_path: str, pdf_path: str) -> bool:
    try:
        import pdfkit

        options = {
            "page-size": "Letter",
            "margin-top": "0",
            "margin-right": "0",
            "margin-bottom": "0",
            "margin-left": "0",
            "encoding": "UTF-8",
            "enable-local-file-access": "",
            "print-media-type": "",
            "quiet": "",
        }

        pdfkit.from_file(html_path, pdf_path, options=options)
        log("Agent 6", f"PDF saved path={pdf_path}")
        return True

    except Exception as e:
        log("Agent 6", f"PDF conversion failed error={repr(e)} html_path={html_path}")
        return False


class SynthesisAgent:
    def __init__(self):
        log("Agent 6", "Ready")

    def run(
        self,
        meeting_context,
        structured_result=None,
        rag_result=None,
        market_result=None,
        news_result=None,
        report_template: dict | None = None,
        output_path: str = "./briefing.pdf",
    ) -> BriefingResult:
        if not report_template:
            raise ValueError("report_template is required")

        output_path = Path(output_path)
        html_path = output_path.with_suffix(".html")
        pdf_path = output_path.with_suffix(".pdf")

        log("Agent 6", f"Starting synthesis plan={meeting_context.plan_name} meeting_type={meeting_context.meeting_type}")

        toc = generate_toc(
            meeting_context=meeting_context,
            structured_result=structured_result,
            rag_result=rag_result,
            market_result=market_result,
            news_result=news_result,
            report_template=report_template,
        )
        log("Agent 6", f"TOC titles={[s['title'] for s in toc]}")

        sections = []
        for i, section in enumerate(toc):
            sections.append(
                write_section(
                    section=section,
                    meeting_context=meeting_context,
                    structured_result=structured_result,
                    rag_result=rag_result,
                    market_result=market_result,
                    news_result=news_result,
                    index=i,
                    total_sections=len(toc),
                )
            )

        proactive_insights_html = generate_proactive_insights(
            meeting_context=meeting_context,
            structured_result=structured_result,
            rag_result=rag_result,
            market_result=market_result,
            news_result=news_result,
            sections=sections,
            report_template=report_template,
        )

        html = assemble_html(
            meeting_context=meeting_context,
            sections=sections,
            proactive_insights_html=proactive_insights_html,
            toc=toc,
        )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        log("Agent 6", f"HTML saved path={html_path}")

        pdf_ok = convert_to_pdf(str(html_path), str(pdf_path))

        result = BriefingResult(
            meeting_context=meeting_context,
            sections=sections,
            proactive_insights_html=proactive_insights_html,
            html_path=str(html_path),
            pdf_path=str(pdf_path) if pdf_ok else None,
        )

        log("Agent 6", f"Complete result={result}")
        return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from agent_01_query_decomposition import MeetingContext, QueryDecompositionAgent
    from agent_02_structured_data import StructuredDataAgent
    from agent_03_document_rag import DocumentRAGAgent
    from agent_04_market_intelligence import MarketIntelAgent
    from agent_05_news import NewsAgent

    REPORT_TEMPLATE = {
        "report_name": "Investment Manager Meeting Briefing",
        "default_sections": [
            {
                "title": "Executive Summary",
                "purpose": "Summarize the most important issues for the meeting.",
                "preferred_data_sources": ["structured", "rag", "market", "news"]
            },
            {
                "title": "Client / Plan Overview",
                "purpose": "Describe the pension plan, its scale, constraints, and strategic context.",
                "preferred_data_sources": ["structured", "rag"]
            },
            {
                "title": "Relationship Context",
                "purpose": "Summarize relationship history, prior interactions, and relevant board sentiment.",
                "preferred_data_sources": ["structured", "rag"]
            },
            {
                "title": "Strategy Context",
                "purpose": "Explain the strategy-specific market and portfolio context relevant to the meeting.",
                "preferred_data_sources": ["market", "news", "structured"]
            },
            {
                "title": "Risks and Pressure Points",
                "purpose": "Highlight concerns, conflicts, scrutiny points, and unresolved issues.",
                "preferred_data_sources": ["structured", "rag", "market", "news"]
            },
            {
                "title": "Recommended Talking Points",
                "purpose": "Provide practical talking points and guidance for the meeting.",
                "preferred_data_sources": ["structured", "rag", "market", "news"]
            }
        ],
        "optional_sections": [
            {
                "title": "Competitive Landscape",
                "purpose": "Compare competitors and likely differentiation points.",
                "preferred_data_sources": ["structured", "market", "news"]
            },
            {
                "title": "RFP / Mandate Requirements",
                "purpose": "Summarize requirements, evaluation criteria, and mandate expectations.",
                "preferred_data_sources": ["rag", "structured"]
            },
            {
                "title": "Board Dynamics",
                "purpose": "Summarize decision-makers, known priorities, and likely question patterns.",
                "preferred_data_sources": ["structured", "rag"]
            },
            {
                "title": "Upcoming Dates and Decision Timeline",
                "purpose": "Highlight key dates, deadlines, renewals, and timing risks.",
                "preferred_data_sources": ["structured"]
            }
        ]
    }

    context = MeetingContext(
        plan_name="CalPERS",
        strategy="Core Fixed Income",
        meeting_type="Annual Review",
        meeting_date="2026-03-15",
        manager_name="Apex Capital Management",
        additional_notes="We have managed this mandate for 3 years. Board has raised fee concerns.",
    )

    log("Pipeline", "Step 1: Query Decomposition")
    decomp_agent = QueryDecompositionAgent()
    decomposed = decomp_agent.run(context)

    log("Pipeline", "Step 2: Structured Data")
    struct_agent = StructuredDataAgent()
    struct_result = struct_agent.run(
        questions=decomposed.structured_data,
        plan_name=context.plan_name,
        manager_name=context.manager_name,
    )

    log("Pipeline", "Step 3: Document RAG")
    rag_agent = DocumentRAGAgent()
    rag_result = rag_agent.run(
        questions=decomposed.document_rag,
        plan_name=context.plan_name,
    )

    log("Pipeline", "Step 4: Market Intelligence")
    market_agent = MarketIntelAgent()
    market_result = market_agent.run(
        questions=decomposed.market_intelligence,
        strategy=context.strategy,
    )

    log("Pipeline", "Step 5: News")
    news_agent = NewsAgent(mock_mode=True)
    news_result = news_agent.run(
        questions=decomposed.news,
        plan_name=context.plan_name,
        strategy=context.strategy,
    )

    log("Pipeline", "Step 6: Synthesis")
    synth_agent = SynthesisAgent()
    result = synth_agent.run(
        meeting_context=context,
        structured_result=struct_result,
        rag_result=rag_result,
        market_result=market_result,
        news_result=news_result,
        report_template=REPORT_TEMPLATE,
        output_path="./briefing_calpers_annual_review.pdf",
    )

    print(f"HTML: {result.html_path}")
    print(f"PDF: {result.pdf_path}")
    print(f"Sections: {[s.title for s in result.sections]}")