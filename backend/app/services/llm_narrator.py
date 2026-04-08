"""
LLM Narrator — generates analyst-style commentary from structured insights.

Uses the LSEG Azure OpenAI GPT-5 API with strict guardrails:
- Only references data provided in the insights
- No hallucination of events, dates, or policy actions
- Supports tone control (neutral, bullish, bearish, cautious)
"""

from __future__ import annotations
import hashlib
import json
import logging
import httpx

from ..models.schemas import ChartInsight, Narrative

logger = logging.getLogger(__name__)

# ─── LSEG Azure OpenAI Configuration ─────────────────────────────────────────

LLM_ENDPOINT = (
    "https://a1a-52048-dev-cog-rioai-eus2-1.openai.azure.com"
    "/openai/deployments/gpt-5_2025-08-07/chat/completions"
    "?api-version=2025-01-01-preview"
)
LLM_API_KEY = "f57572c4e8db4f8a8ef2878cabe5fce2"

# Shared async HTTP client — one TCP/TLS connection pool reused across all requests (~100-300ms saved per call)
_http_client = httpx.AsyncClient(timeout=120.0)

# In-memory narrative cache: insight hash → Narrative
# Same chart re-uploaded (or /api/narrative called again) gets instant response.
_narrative_cache: dict[str, Narrative] = {}


async def close_http_client() -> None:
    """Close the shared HTTP client. Called on application shutdown."""
    if not _http_client.is_closed:
        await _http_client.aclose()


# ─── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior financial analyst at a global research firm.
Your task is to generate clear, professional commentary from STRUCTURED DATA ONLY.

STRICT RULES:
1. ONLY reference data, statistics, trends, and anomalies provided in the JSON below.
2. NEVER mention specific events, policy decisions, geopolitical actions, or causes NOT in the data.
3. Use hedging language for interpretive statements: "appears to", "suggests", "indicates".
4. All numbers must come directly from the data provided.
5. Do NOT invent dates or time periods not present in the data.

WRITING STYLE:
- Write like a Bloomberg or Reuters market brief — concise, authoritative, narrative-driven.
- Lead with the most important insight, not raw numbers.
- Embed key figures naturally within sentences. NEVER list raw statistics.
- Describe direction and magnitude in plain language: "climbed sharply", "remained broadly stable", "pulled back modestly".
- Round aggressively: say "roughly 54%" not "53.96%", "nearly doubled" not "increased 181.41%".
- Use comparative language: "outpaced peers", "lagged the group average", "diverged markedly".
- For anomalies, explain significance to an analyst: "an outlier that warrants further investigation" rather than citing z-scores.
- Key takeaways should be actionable observations an analyst can brief to a portfolio manager, not restated statistics.

OUTPUT FORMAT — respond with valid JSON only:
{
  "summary": "2-3 sentence headline-style finding. Lead with the narrative, embed one or two rounded figures.",
  "detailed": "A 4-6 sentence analyst paragraph telling the data story. Emphasize direction, pace, and relative performance. Cite only the most meaningful numbers, rounded and in context.",
  "key_takeaways": ["Actionable observation 1", "Actionable observation 2", "Actionable observation 3"],
  "tone": "the tone you used"
}"""


def _build_user_prompt(insight: ChartInsight, tone: str = "neutral", focus_series: str = "") -> str:
    """Build the user prompt with structured insight data."""

    # Only send stats per series (not raw data points — far too verbose)
    series_summaries = [
        {"name": s.name, "stats": s.stats.model_dump()}
        for s in insight.series
    ]

    data_block = {
        "chart_title": insight.metadata.title,
        "chart_subtitle": insight.metadata.subtitle,
        "chart_type": insight.metadata.chart_type,
        "series": series_summaries,
        # Cap to avoid inflating prompt size / token count
        "trends": [t.model_dump() for t in insight.trends[:8]],
        "anomalies": [a.model_dump() for a in insight.anomalies[:5]],
        "correlations": [c.model_dump() for c in insight.correlations[:4]],
        "confidence": insight.overall_confidence,
    }

    # Compact JSON (no indent/spaces) — ~25-35% fewer prompt tokens vs indent=2
    data_json = json.dumps(data_block, separators=(",", ":"))

    focus_line = f"FOCUS ON SERIES: {focus_series}\n" if focus_series else ""
    return (
        f"Analyze this financial chart data and generate commentary.\n\n"
        f"TONE: {tone}\n"
        f"{focus_line}"
        f"CHART DATA:\n```json\n{data_json}\n```\n\n"
        "Generate the analyst commentary as JSON following the system instructions."
    )


# ─── API Call ─────────────────────────────────────────────────────────────────

async def generate_narrative(
    insight: ChartInsight,
    tone: str = "neutral",
    focus_series: str = "",
) -> Narrative:
    """
    Call LSEG Azure OpenAI GPT-5 to generate narrative from structured insights.
    Returns cached result instantly if the same insight+tone was seen before.
    """
    user_prompt = _build_user_prompt(insight, tone, focus_series)

    # Cache lookup — same chart re-uploaded or /api/narrative called again = zero LLM cost
    cache_key = hashlib.md5(user_prompt.encode()).hexdigest()
    if cache_key in _narrative_cache:
        logger.info("Narrative cache hit")
        return _narrative_cache[cache_key]

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # 1500 tokens: enough for summary(2-3 sent) + detailed(4-6 sent) + 3 takeaways
        # with headroom, while avoiding the 10001 compute-reservation penalty.
        "max_completion_tokens": 1500,
    }

    headers = {
        "api-key": LLM_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        response = await _http_client.post(LLM_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        choice = result["choices"][0]
        content = choice["message"].get("content") or ""
        finish_reason = choice.get("finish_reason", "")

        if finish_reason == "content_filter":
            logger.warning("LLM response filtered by content policy, using fallback")
            return _fallback_narrative(insight, tone)

        # Parse JSON response — handle potential markdown code blocks
        content = content.strip()
        if not content:
            logger.warning(f"LLM returned empty content (finish_reason={finish_reason}), using fallback")
            return _fallback_narrative(insight, tone)

        if content.startswith("```"):
            content = content.split("\n", 1)[1]  # remove first line
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        # Try to extract JSON from within the response if it has surrounding text
        if not content.startswith("{"):
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                content = content[json_start:json_end]

        narrative_data = json.loads(content)

        narrative = Narrative(
                summary=narrative_data.get("summary", ""),
                detailed=narrative_data.get("detailed", ""),
                key_takeaways=narrative_data.get("key_takeaways", []),
                tone=narrative_data.get("tone", tone),
            )
        _narrative_cache[cache_key] = narrative
        return narrative

    except httpx.HTTPStatusError as e:
        logger.error(f"LLM API error: {e.response.status_code} - {e.response.text}")
        return _fallback_narrative(insight, tone)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Raw LLM content (first 500 chars): {content[:500] if content else '(empty)'}")
        return _fallback_narrative(insight, tone)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return _fallback_narrative(insight, tone)


def generate_narrative_sync(
    insight: ChartInsight,
    tone: str = "neutral",
    focus_series: str = "",
) -> Narrative:
    """Synchronous version for testing."""
    import asyncio
    return asyncio.run(generate_narrative(insight, tone, focus_series))


async def stream_narrative(
    insight: ChartInsight,
    tone: str = "neutral",
    focus_series: str = "",
):
    """
    Async generator that streams narrative tokens from the LLM streaming API.
    Yields raw text chunks as they arrive — wire directly to an SSE endpoint.
    Reuses the shared _http_client connection pool (no per-call TLS handshake).
    """
    user_prompt = _build_user_prompt(insight, tone, focus_series)

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": 1500,
    }

    headers = {
        "api-key": LLM_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        async with _http_client.stream("POST", LLM_ENDPOINT, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    except Exception as e:
        logger.error(f"stream_narrative failed: {e}")
        yield "[Streaming error — narrative unavailable]"


# ─── Fallback Narrative (no LLM needed) ──────────────────────────────────────

def _fallback_narrative(insight: ChartInsight, tone: str = "neutral") -> Narrative:
    """Generate a basic narrative without LLM when API fails."""
    title = insight.metadata.title or "this chart"

    # Build summary from stats
    parts = []
    for s in insight.series:
        if s.stats.data_point_count > 0:
            direction = "increased" if s.stats.overall_change_pct > 0 else "decreased"
            parts.append(
                f"{s.name} {direction} by {abs(s.stats.overall_change_pct):.1f}% "
                f"from {s.stats.first_value:.1f} to {s.stats.latest_value:.1f}"
            )

    summary = f"Analysis of {title}. " + ". ".join(parts[:2]) + "." if parts else f"Analysis of {title}."

    # Build takeaways from trends
    takeaways = []
    for t in insight.trends[:5]:
        takeaways.append(
            f"{t.series_name}: {t.direction.value} trend from {t.start_label} to {t.end_label} "
            f"({t.magnitude_pct:+.1f}%)"
        )

    # Add anomaly takeaways
    for a in insight.anomalies[:3]:
        takeaways.append(f"Anomaly: {a.description}")

    return Narrative(
        summary=summary,
        detailed=summary + " " + " ".join(takeaways),
        key_takeaways=takeaways[:5],
        tone=tone,
    )


# ─── Multi-Chart Comparison ──────────────────────────────────────────────────

async def generate_comparison_narrative(
    insights: list[ChartInsight],
    tone: str = "neutral",
) -> Narrative:
    """Generate a comparative narrative across multiple charts."""

    # Build comparison data
    charts_summary = []
    for ins in insights:
        chart_info = {
            "title": ins.metadata.title,
            "series_count": len(ins.series),
            "series": [
                {"name": s.name, "change_pct": s.stats.overall_change_pct, "latest": s.stats.latest_value}
                for s in ins.series
            ],
            "trend_count": len(ins.trends),
            "anomaly_count": len(ins.anomalies),
        }
        charts_summary.append(chart_info)

    user_prompt = f"""Compare these financial charts and identify cross-cutting themes.

TONE: {tone}

CHARTS:
```json
{json.dumps(charts_summary, separators=(',', ':'))}
```

Generate a comparative analyst commentary as JSON following the system instructions."""

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": 1500,
    }

    headers = {
        "api-key": LLM_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        response = await _http_client.post(LLM_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        data = json.loads(content)
        return Narrative(
            summary=data.get("summary", ""),
            detailed=data.get("detailed", ""),
            key_takeaways=data.get("key_takeaways", []),
            tone=data.get("tone", tone),
        )
    except Exception as e:
        logger.error(f"Comparison LLM call failed: {e}")
        return Narrative(
            summary="Comparison analysis unavailable.",
            detailed="Unable to generate comparison at this time.",
            key_takeaways=[],
            tone=tone,
        )

