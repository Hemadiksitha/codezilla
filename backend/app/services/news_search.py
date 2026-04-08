"""
News Search — fetches relevant news headlines for chart breakpoints.

Uses DuckDuckGo HTML search (no API key required) to find
real-world events that may explain trend changes.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


@dataclass
class NewsResult:
    title: str
    snippet: str
    url: str
    date_hint: str  # approximate date from snippet


async def search_news(query: str, max_results: int = 5) -> list[NewsResult]:
    """
    Search DuckDuckGo HTML for news headlines matching a query.
    Returns a list of NewsResult with title, snippet, url.
    """
    results = []
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.post(
                SEARCH_URL,
                data={"q": query, "b": ""},
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            html = resp.text

            # Parse results from DuckDuckGo HTML response
            title_pattern = re.compile(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                re.DOTALL,
            )
            snippet_pattern = re.compile(
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                re.DOTALL,
            )

            titles = title_pattern.findall(html)
            snippets = snippet_pattern.findall(html)

            for i in range(min(len(titles), len(snippets), max_results)):
                url = titles[i][0]
                # DuckDuckGo wraps URLs; extract the real one
                real_url_match = re.search(r'uddg=([^&]+)', url)
                if real_url_match:
                    from urllib.parse import unquote
                    url = unquote(real_url_match.group(1))

                title_text = _strip_html(titles[i][1]).strip()
                snippet_text = _strip_html(snippets[i]).strip()

                # Try to extract a date hint from the snippet
                date_hint = ""
                date_match = re.search(
                    r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+\d{4}\b'
                    r'|\b\d{4}[-/]\d{2}[-/]\d{2}\b'
                    r'|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{4}\b)',
                    snippet_text,
                    re.IGNORECASE,
                )
                if date_match:
                    date_hint = date_match.group(0)

                if title_text:
                    results.append(NewsResult(
                        title=title_text,
                        snippet=snippet_text[:300],
                        url=url,
                        date_hint=date_hint,
                    ))

    except Exception as e:
        logger.warning(f"News search failed for '{query}': {e}")

    return results


def build_search_queries(
    chart_title: str,
    trends: list[dict],
    anomalies: list[dict],
) -> list[str]:
    """
    Build targeted search queries from chart metadata and breakpoints.
    """
    queries = []
    topic = re.sub(r'^\d+\.\s*', '', chart_title)
    topic = re.sub(r'\s*-\s*\d+\s*', ' ', topic)
    topic = topic.strip()

    if not topic:
        return queries

    for t in trends:
        if abs(t.get("magnitude_pct", 0)) > 5:
            direction = t.get("direction", "")
            start = t.get("start_label", "")
            end = t.get("end_label", "")
            period = ""
            if start and end:
                period = f"{start} {end}"
            elif start:
                period = start

            if direction in ("rising", "spike"):
                queries.append(f"{topic} increase {period} reason why")
            elif direction in ("falling", "dip"):
                queries.append(f"{topic} decline {period} reason why")

    for a in anomalies[:2]:
        label = a.get("x_label", "")
        queries.append(f"{topic} {label} unusual change")

    queries.append(f"{topic} major events trends analysis")

    seen = set()
    unique = []
    for q in queries:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique[:5]


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&#x27;', "'").replace('&quot;', '"').replace('&#39;', "'")
    return text
