"""
NOVA-C — FastAPI Backend
Narrative Output from Visual Analytics – Charts

Routes:
  POST /api/upload          — Upload SVG, get full analysis
  POST /api/analyze         — Re-analyze with different parameters
  POST /api/narrative       — Generate/regenerate narrative
  POST /api/compare         — Multi-chart comparison
  GET  /api/charts          — List all analyzed charts
  GET  /api/charts/{id}     — Get specific chart analysis
  GET  /api/demo            — Load all 7 demo charts
  GET  /api/health          — Health check
"""

from __future__ import annotations
import asyncio
import json
import uuid
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

from .services.svg_parser import parse_svg
from .services.axis_calibrator import calibrate_chart
from .services.trend_engine import analyze_chart
from .services import llm_narrator as _llm
from .services.llm_narrator import (
    generate_narrative,
    generate_comparison_narrative,
    stream_narrative,
    _fallback_narrative,
)
from .models.schemas import ChartAnalysisResult, ChartInsight, Narrative, NewsEvent
from .services.news_search import search_news, build_search_queries
from .routers.auth import router as auth_router, require_auth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Demo lazy-cache state ─────────────────────────────────────────────────────

_demo_chart_ids: list[str] = []  # populated on first /api/demo call


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await _llm.close_http_client()


app = FastAPI(
    title="NOVA-C API",
    description="Narrative Output from Visual Analytics – Charts",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth router
app.include_router(auth_router)

# In-memory store for analyzed charts
chart_store: dict[str, ChartAnalysisResult] = {}
# In-memory store for raw SVG content
svg_store: dict[str, bytes] = {}

# Path to demo SVG files
DEMO_SVG_DIR = Path(__file__).parent.parent.parent / "Chart SVGs"
STATIC_DIR = Path(__file__).parent / "static"


# ─── Request/Response Models ──────────────────────────────────────────────────

class NarrativeRequest(BaseModel):
    chart_id: str
    tone: str = "neutral"  # neutral, bullish, bearish, cautious
    focus_series: str = ""


class CompareRequest(BaseModel):
    chart_ids: list[str]
    tone: str = "neutral"


class CompareResponse(BaseModel):
    narrative: Narrative
    chart_ids: list[str]


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    chart_id: str
    question: str
    history: List[ChatMessage] = []


class SelectedSeries(BaseModel):
    name: str
    data_points: list[dict]


class RangeAnalysisRequest(BaseModel):
    chart_id: str
    from_label: str
    to_label: str
    selected_series: List[SelectedSeries]
    chart_title: str = ""
    chart_subtitle: str = ""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "NOVA-C", "charts_loaded": len(chart_store)}


@app.post("/api/upload", response_model=ChartAnalysisResult)
async def upload_chart(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: dict = Depends(require_auth),
):
    """
    Upload an SVG. Parse+analyze runs in a thread pool (~100-200ms).
    Response includes full fallback commentary immediately — no empty screen.
    Background task upgrades narrative to LLM quality; stream via
    GET /api/narrative/{chart_id}/stream.
    """
    if not file.filename or not file.filename.endswith(".svg"):
        raise HTTPException(400, "Only SVG files are accepted")

    content = await file.read()
    chart_id = str(uuid.uuid4())[:8]

    try:
        insight = await asyncio.to_thread(_parse_pipeline_sync, content, chart_id)
    except Exception as e:
        logger.exception(f"Failed to parse SVG: {e}")
        raise HTTPException(500, f"Failed to process chart: {str(e)}")

    # Return fallback narrative immediately — full commentary visible on first load.
    # Background task upgrades it to LLM quality; frontend can poll or stream the upgrade.
    initial_narrative = _fallback_narrative(insight)
    result = ChartAnalysisResult(insight=insight, narrative=initial_narrative)
    chart_store[chart_id] = result
    svg_store[chart_id] = content

    # LLM narrative + news generated concurrently in background
    background_tasks.add_task(_upgrade_chart_background, chart_id, insight)

    return result


@app.post("/api/narrative", response_model=Narrative)
async def regenerate_narrative(request: NarrativeRequest, user: dict = Depends(require_auth)):
    """Regenerate narrative with different tone or focus."""
    if request.chart_id not in chart_store:
        raise HTTPException(404, f"Chart {request.chart_id} not found")

    result = chart_store[request.chart_id]
    narrative = await generate_narrative(
        result.insight,
        tone=request.tone,
        focus_series=request.focus_series,
    )

    # Update stored result
    result.narrative = narrative
    return narrative


@app.post("/api/compare", response_model=CompareResponse)
async def compare_charts(request: CompareRequest, user: dict = Depends(require_auth)):
    """Compare multiple charts and generate cross-cutting narrative."""
    insights = []
    for cid in request.chart_ids:
        if cid not in chart_store:
            raise HTTPException(404, f"Chart {cid} not found")
        insights.append(chart_store[cid].insight)

    narrative = await generate_comparison_narrative(insights, tone=request.tone)
    return CompareResponse(narrative=narrative, chart_ids=request.chart_ids)


@app.post("/api/analyze")
async def analyze_modified(request: dict, user: dict = Depends(require_auth)):
    """
    Re-analyze modified chart data: recompute trends, anomalies, correlations.
    Returns fallback narrative immediately; LLM narrative follows via /api/analyze/narrative.
    """
    from .models.schemas import (
        ChartData, ChartMetadata, SeriesData, AxisInfo, DataPoint as SchemaDP,
    )

    try:
        insight_data = request.get("insight", request)

        # Reconstruct ChartData from the modified insight
        metadata = ChartMetadata(**insight_data.get("metadata", {}))
        x_axis = AxisInfo(**insight_data.get("x_axis", {}))
        y_axis = AxisInfo(**insight_data.get("y_axis", {}))

        series_list = []
        for s in insight_data.get("series", []):
            dps = [SchemaDP(**dp) for dp in s.get("data_points", [])]
            series_list.append(SeriesData(
                name=s.get("name", ""),
                color=s.get("color", ""),
                data_points=dps,
                is_area=metadata.chart_type == "area",
            ))

        chart_data = ChartData(
            chart_id=insight_data.get("chart_id", "modified"),
            metadata=metadata,
            x_axis=x_axis,
            y_axis=y_axis,
            series=series_list,
            plot_area=insight_data.get("plot_area", {}),
            confidence=insight_data.get("overall_confidence", 1.0),
        )

        # Fast: re-run trend engine (no LLM)
        new_insight = analyze_chart(chart_data)
        # Fast: return fallback narrative immediately
        fallback = _fallback_narrative(new_insight)
        # Store for the LLM follow-up
        mod_id = "mod_" + new_insight.chart_id
        chart_store[mod_id] = ChartAnalysisResult(insight=new_insight, narrative=fallback)

        return {
            "insight": new_insight.model_dump(),
            "narrative": fallback.model_dump(),
            "mod_id": mod_id,
        }

    except Exception as e:
        logger.exception(f"Re-analysis failed: {e}")
        raise HTTPException(500, f"Re-analysis failed: {str(e)}")


@app.post("/api/analyze/narrative")
async def analyze_narrative(request: dict, user: dict = Depends(require_auth)):
    """Generate LLM narrative for a previously analyzed modified chart."""
    mod_id = request.get("mod_id", "")
    if mod_id not in chart_store:
        raise HTTPException(404, "Modified chart not found")

    insight = chart_store[mod_id].insight
    narrative = await generate_narrative(insight)
    chart_store[mod_id].narrative = narrative
    return {"narrative": narrative.model_dump()}


@app.post("/api/chat")
async def chat_with_data(request: ChatRequest, user: dict = Depends(require_auth)):
    """Ask a question about a chart using LLM."""
    if request.chart_id not in chart_store:
        raise HTTPException(404, f"Chart {request.chart_id} not found")

    result = chart_store[request.chart_id]
    insight = result.insight

    # Build chart context for the LLM
    series_info = []
    for s in insight.series:
        dp_summary = []
        for dp in s.data_points:
            dp_summary.append({"label": dp.x_label, "value": round(dp.value, 2)})
        series_info.append({
            "name": s.name,
            "data_points": dp_summary,
            "stats": s.stats.model_dump() if s.stats else {},
        })

    chart_context = {
        "title": insight.metadata.title,
        "subtitle": insight.metadata.subtitle,
        "chart_type": insight.metadata.chart_type,
        "source": insight.metadata.source,
        "series": series_info,
        "trends": [t.model_dump() for t in insight.trends],
        "anomalies": [a.model_dump() for a in insight.anomalies],
        "correlations": [c.model_dump() for c in insight.correlations],
    }

    import json as _json
    from .services.llm_narrator import LLM_ENDPOINT, LLM_API_KEY, _http_client

    system_msg = (
        "You are an expert financial data analyst assistant. "
        "You have access to the following chart data. Answer user questions "
        "accurately based ONLY on this data. Be concise but thorough. "
        "If the user asks about a series, provide specific numbers, trends, "
        "peaks, troughs, and changes. If unsure, say so rather than guessing.\n\n"
        f"CHART DATA:\n```json\n{_json.dumps(chart_context, indent=2)}\n```"
    )

    messages = [{"role": "system", "content": system_msg}]
    for msg in request.history[-10:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": request.question})

    payload = {
        "messages": messages,
        "max_completion_tokens": 4000,
    }
    headers = {
        "api-key": LLM_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        response = await _http_client.post(LLM_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Chat LLM call failed: {e}")
        raise HTTPException(500, "Failed to get answer from AI. Please try again.")


@app.get("/api/charts")
async def list_charts(user: dict = Depends(require_auth)):
    """List all analyzed charts."""
    return [
        {
            "chart_id": cid,
            "title": r.insight.metadata.title,
            "chart_type": r.insight.metadata.chart_type,
            "series_count": len(r.insight.series),
            "confidence": r.insight.overall_confidence,
        }
        for cid, r in chart_store.items()
    ]


@app.get("/api/charts/{chart_id}", response_model=ChartAnalysisResult)
async def get_chart(chart_id: str, user: dict = Depends(require_auth)):
    """Get a specific chart's full analysis."""
    if chart_id not in chart_store:
        raise HTTPException(404, f"Chart {chart_id} not found")
    return chart_store[chart_id]


@app.get("/api/charts/{chart_id}/svg")
async def get_chart_svg(chart_id: str, user: dict = Depends(require_auth)):
    """Return the original raw SVG file for a chart."""
    if chart_id not in svg_store:
        raise HTTPException(404, f"SVG for chart {chart_id} not found")
    return Response(content=svg_store[chart_id], media_type="image/svg+xml")


@app.post("/api/range-analysis")
async def range_analysis(request: RangeAnalysisRequest, user: dict = Depends(require_auth)):
    """Return news summary for a selected time range via LLM + news search."""
    import httpx
    from .services.llm_narrator import LLM_ENDPOINT, LLM_API_KEY

    logger.info(f"Range analysis requested: {request.from_label} → {request.to_label} "
                f"on '{request.chart_title}'")

    summaries = []
    for s in request.selected_series:
        pts = s.data_points
        if not pts:
            continue
        vals = [p.get("value", 0) for p in pts]
        first_val, last_val = vals[0], vals[-1]
        change = last_val - first_val
        pct = (change / abs(first_val) * 100) if first_val != 0 else 0
        direction = "rose" if change > 0 else "fell" if change < 0 else "unchanged"
        summaries.append(f"{s.name}: {direction} from {first_val:.1f} to {last_val:.1f} ({pct:+.1f}%)")

    data_summary = "; ".join(summaries)
    headers = {"api-key": LLM_API_KEY, "Content-Type": "application/json"}

    # Resolve abbreviated axis labels to real dates
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resolve_payload = {
                "messages": [{"role": "user", "content": (
                    f"A financial chart titled '{request.chart_title}' has x-axis labels "
                    f"'{request.from_label}' and '{request.to_label}'. "
                    f"On financial charts, 'Apr 13' means April 2013, '21' means 2021, "
                    f"'Oct 5' means October 2005. "
                    f"What are the actual start and end dates? "
                    f"Reply ONLY with: START: [date], END: [date]"
                )}],
                "max_completion_tokens": 8000,
            }
            r1 = await client.post(LLM_ENDPOINT, json=resolve_payload, headers=headers)
            dates_text = ""
            if r1.status_code == 200:
                d1 = r1.json()
                dates_text = (d1.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                logger.info(f"Date resolution: [{dates_text}]")

            if not dates_text or len(dates_text) < 5:
                dates_text = f"{request.from_label} to {request.to_label}"

            # Fetch news articles for the period
            news_events = []
            try:
                period_queries = build_search_queries(
                    f"{request.chart_title} {dates_text}",
                    [], []
                )
                seen = set()
                for q in period_queries[:3]:
                    results = await search_news(q, max_results=3)
                    for r in results:
                        if r.title not in seen:
                            seen.add(r.title)
                            news_events.append({
                                "headline": r.title,
                                "snippet": r.snippet or "",
                                "url": r.url or "",
                                "source": r.date_hint or "",
                            })
                logger.info(f"Range news: {len(news_events)} articles found")
            except Exception as ne:
                logger.warning(f"Range news search failed: {ne}")

            # LLM: summarize news for the period
            news_context = ""
            if news_events:
                headlines = "; ".join([e["headline"] for e in news_events[:6]])
                news_context = f"\nRecent news headlines for this period: {headlines}\n"

            summary_payload = {
                "messages": [{"role": "user", "content": (
                    f"Chart: '{request.chart_title}' ({request.chart_subtitle}).\n"
                    f"Period: {dates_text}.\n"
                    f"Data: {data_summary}.\n"
                    f"{news_context}\n"
                    f"Give a brief 2-3 sentence news summary of what happened during this "
                    f"specific time period that is relevant to this data. Focus on key events, "
                    f"policy changes, or market developments. Be specific with dates."
                )}],
                "max_completion_tokens": 16000,
            }
            r2 = await client.post(LLM_ENDPOINT, json=summary_payload, headers=headers)
            news_summary = ""
            if r2.status_code == 200:
                d2 = r2.json()
                news_summary = (d2.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                logger.info(f"News summary: [{news_summary[:300]}]")

            if not news_summary or len(news_summary) < 10:
                news_summary = "No news summary available for this period."

            return {
                "news_summary": news_summary,
                "news_articles": news_events[:6],
            }

    except Exception as e:
        logger.error(f"Range analysis failed: {e}")
        return {"news_summary": "Could not retrieve news for this period.", "news_articles": []}


@app.get("/api/demo")
async def load_demo(user: dict = Depends(require_auth)):
    """Return demo charts. Served from cache after first load (pre-warmed on startup)."""
    if not DEMO_SVG_DIR.exists():
        raise HTTPException(404, "Demo SVG directory not found")

    # Return cached results if pre-warm already ran
    if _demo_chart_ids:
        results = []
        for cid in _demo_chart_ids:
            if cid in chart_store:
                r = chart_store[cid]
                results.append({
                    "chart_id": cid,
                    "title": r.insight.metadata.title,
                    "chart_type": r.insight.metadata.chart_type,
                    "series_count": len(r.insight.series),
                    "confidence": r.insight.overall_confidence,
                })
        return {"charts": results, "total": len(results), "cached": True}

    svg_files = sorted(DEMO_SVG_DIR.glob("*.svg"))
    if not svg_files:
        raise HTTPException(404, "No SVG files found in demo directory")

    await _load_demo_charts(svg_files)
    results = [
        {
            "chart_id": cid,
            "title": chart_store[cid].insight.metadata.title,
            "chart_type": chart_store[cid].insight.metadata.chart_type,
            "series_count": len(chart_store[cid].insight.series),
            "confidence": chart_store[cid].insight.overall_confidence,
        }
        for cid in _demo_chart_ids
        if cid in chart_store
    ]
    return {"charts": results, "total": len(results)}


@app.get("/api/narrative/{chart_id}/stream")
async def stream_chart_narrative(
    chart_id: str,
    tone: str = Query("neutral"),
    focus_series: str = Query(""),
    user: dict = Depends(require_auth),
):
    """
    Stream narrative tokens as SSE. Frontend can start rendering immediately
    instead of waiting for the full LLM response (~2-8s on average).

    Event format:  data: {"token": "<text chunk>"}\n\n
    Terminal event: data: [DONE]\n\n
    """
    if chart_id not in chart_store:
        raise HTTPException(404, f"Chart {chart_id} not found")

    insight = chart_store[chart_id].insight

    async def event_stream():
        try:
            async for token in stream_narrative(insight, tone=tone, focus_series=focus_series):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            logger.error(f"SSE stream error for {chart_id}: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )


# ─── Serve Frontend ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Core Processing Pipeline ─────────────────────────────────────────────────

def _parse_pipeline_sync(content: bytes, chart_id: str):
    """
    Synchronous parse → calibrate → analyze pipeline.
    Safe to run in a thread pool — lxml/numpy/scipy all release the GIL.
    """
    chart_data = parse_svg(content)
    chart_data.chart_id = chart_id
    chart_data = calibrate_chart(chart_data)
    return analyze_chart(chart_data)


async def _generate_and_store_narrative(chart_id: str, insight: ChartInsight) -> None:
    """Background task: generate narrative and write it into the chart store."""
    try:
        narrative = await generate_narrative(insight)
    except Exception as e:
        logger.error(f"Background narrative failed for {chart_id}: {e}")
        narrative = _fallback_narrative(insight)
    if chart_id in chart_store:
        chart_store[chart_id].narrative = narrative


async def _upgrade_chart_background(chart_id: str, insight: ChartInsight) -> None:
    """Background task: fetch LLM narrative and news concurrently, store both."""
    logger.info(f"Background upgrade starting for {chart_id}")
    try:
        narrative_result, news_result = await asyncio.gather(
            _safe_generate_narrative(insight),
            _fetch_news(insight),
        )
        if chart_id in chart_store:
            chart_store[chart_id].narrative = narrative_result
            chart_store[chart_id].news_events = news_result
            logger.info(f"Background upgrade complete for {chart_id}: {len(news_result)} news events")
    except Exception as e:
        logger.error(f"Background upgrade failed for {chart_id}: {e}")


async def _safe_generate_narrative(insight: ChartInsight) -> Narrative:
    """Generate narrative, falling back on any error."""
    try:
        return await generate_narrative(insight)
    except Exception as e:
        logger.error(f"Narrative generation failed: {e}")
        return _fallback_narrative(insight)


async def _process_svg(content: bytes, chart_id: str) -> ChartAnalysisResult:
    """Full pipeline used by demo loader (parse + narrate + news, all awaited)."""
    insight = await asyncio.to_thread(_parse_pipeline_sync, content, chart_id)
    narrative, news_events = await asyncio.gather(
        generate_narrative(insight),
        _fetch_news(insight),
    )
    return ChartAnalysisResult(insight=insight, narrative=narrative, news_events=news_events)


async def _fetch_news(insight: ChartInsight) -> list[NewsEvent]:
    """Fetch news headlines related to chart trends. Best-effort, never fails."""
    try:
        queries = build_search_queries(
            insight.metadata.title or "",
            [t.model_dump() for t in insight.trends],
            [a.model_dump() for a in insight.anomalies],
        )
        events: list[NewsEvent] = []
        seen_titles = set()
        for q in queries[:4]:
            results = await search_news(q, max_results=3)
            for r in results:
                if r.title not in seen_titles:
                    seen_titles.add(r.title)
                    events.append(NewsEvent(
                        headline=r.title,
                        snippet=r.snippet,
                        url=r.url,
                        date_hint=r.date_hint,
                        search_query=q,
                    ))
        logger.info(f"Fetched {len(events)} news events for '{insight.metadata.title}'")
        return events[:10]
    except Exception as e:
        logger.warning(f"News search failed (non-blocking): {e}")
        return []


async def _fetch_and_store_news(chart_id: str, insight: ChartInsight) -> None:
    """Background task: fetch news and store in chart_store."""
    logger.info(f"Background news fetch starting for {chart_id}")
    try:
        events = await _fetch_news(insight)
        if chart_id in chart_store:
            chart_store[chart_id].news_events = events
            logger.info(f"Stored {len(events)} news events for {chart_id}")
    except Exception as e:
        logger.error(f"Background news fetch failed for {chart_id}: {e}")


async def _load_demo_charts(svg_files: list) -> None:
    """
    Phase 1 — parse all files in parallel threads.
    Phase 2 — all LLM calls concurrently.
    Results cached in chart_store + _demo_chart_ids.
    """
    global _demo_chart_ids

    async def _read_and_parse(svg_path):
        chart_id = str(uuid.uuid4())[:8]
        content = await asyncio.to_thread(svg_path.read_bytes)
        insight = await asyncio.to_thread(_parse_pipeline_sync, content, chart_id)
        return chart_id, svg_path, content, insight

    phase1 = await asyncio.gather(
        *[_read_and_parse(p) for p in svg_files],
        return_exceptions=True,
    )
    pending = [r for r in phase1 if not isinstance(r, Exception)]
    for err in phase1:
        if isinstance(err, Exception):
            logger.error(f"Demo parse error: {err}")

    narratives = await asyncio.gather(
        *[generate_narrative(insight) for _, _, _, insight in pending],
        return_exceptions=True,
    )

    new_ids = []
    for (chart_id, svg_path, content, insight), narrative in zip(pending, narratives):
        if isinstance(narrative, Exception):
            narrative = _fallback_narrative(insight)
        chart_store[chart_id] = ChartAnalysisResult(insight=insight, narrative=narrative)
        svg_store[chart_id] = content
        new_ids.append(chart_id)
        logger.info(f"Demo: {svg_path.name} → {chart_id}")
    _demo_chart_ids = new_ids
