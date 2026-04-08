# NOVA-C — Hackathon MVP Build Plan

## ✅ CURRENT STATUS: MVP COMPLETE & WORKING

**Backend**: FastAPI server at `http://127.0.0.1:8000`
**Frontend**: Served at root `/` — dark-themed dashboard with Chart.js
**LLM**: LSEG Azure OpenAI GPT-5 ✅ generating analyst-quality narratives
**All 7 charts**: Parsing, calibrating, analyzing, narrating — ALL WORKING

### To start the server:
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
cd C:\Users\sn44\IdeaProjects\codezila
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Then open: **http://127.0.0.1:8000**

---

## 🎯 Strategy: Win by Excelling in EVERY Criterion

| Criterion | Weight | Target Score | Our Strategy |
|-----------|--------|-------------|--------------|
| **Innovation & Creativity** | 35% | 5/5 | Confidence scoring, anomaly detection, multi-chart comparison, tone-controlled narratives, auto chart-type detection |
| **User Experience & Usability** | 30% | 5/5 | Beautiful dark dashboard, drag-drop upload, interactive charts with trend overlays, editable commentary, demo mode |
| **Feasibility & Scalability** | 25% | 5/5 | Clean Python+React stack, modular services, Docker-ready, works on all 7 chart types out of the box |
| **Clarity of Presentation** | 10% | 5/5 | One-click demo, PDF/JSON export, compelling storytelling with preloaded charts |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│                 React Frontend               │
│  Upload → Chart Viewer → Insights → Narrative│
└──────────────────┬──────────────────────────┘
                   │ REST API
┌──────────────────▼──────────────────────────┐
│              FastAPI Backend                  │
│                                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  │
│  │SVG Parser│→ │Trend Engine│→ │LLM Narrator│ │
│  └──────────┘  └───────────┘  └──────────┘  │
│       ↓              ↓             ↓         │
│  [ChartData]   [Insights]   [Narrative]      │
└──────────────────────────────────────────────┘
                                    ↓
                        LSEG Azure OpenAI GPT-5
```

## 📊 SVG Chart Types Discovered (all 7 files analyzed)

| # | Chart | Type | Key SVG Elements |
|---|-------|------|-----------------|
| 01 | US unemployment rate | **Multi-line** | 3 `<polyline>` in clip-path, Y: 2-12, X: 2021-2025 |
| 02 | Global crude oil | **Multi-line** | 4 `<polyline>`, Y: 0-120, X: 1970-2020 |
| 03 | Global working age pop | **Stacked area** | `<polyline>` + `<polygon>` fills, Y: 0-6, X: 1960-2020 |
| 04 | China loss-making | **Horizontal bar** | 28 `<rect>` bars, X: 0-60%, Y: region names |
| 05 | Foreign reserves | **Multi-line** | 7 `<polyline>` series, Y: Per cent |
| 06 | Military expenditure | **Log-scale line** | 3 `<polyline>`, Y: 20-1000 (LOG), X: 1990-2025 |
| 07 | US import prices | **Dual-tone +/-** | Alternating `.s9`/`.s10` polylines above/below zero line |

## 🔧 Consistent SVG Structure (parsing rules)

- **Title**: `.s4 > text` — chart title (bold, 37px)
- **Subtitle/Units**: `.s5 > text` — units like "Per cent", "Billions"
- **Source**: `.s1 .s2 > text` — "Source: LSEG Datastream / Fathom Consulting"
- **Data area**: `g[style="clip-path:url(#c0);"]` — contains all data series
- **Series**: `g#n0`, `g#n1`, etc — each series has child `<polyline>` or `<rect>`
- **Y-axis labels**: `text-anchor:end` group → `<text>` with value and y-position
- **X-axis labels**: `text-anchor:middle` group → `<text>` with label and x-position
- **Legend**: text elements after colored polyline samples in legend group
- **Gridlines**: `.s7` group — horizontal dashed lines at major Y values
- **Zero line**: `.s8` group (when present) — baseline for dual-tone charts
- **Metadata comment**: `<!-- <Chart><ImageInfo ... /> -->` at end of file — GUID, GroupName, ChartName, RefreshDate
- **Clip rect**: `<clipPath id="c0"><rect x y width height />` — defines chart plotting area bounds

## 📝 LLM API Details

```
Endpoint: https://a1a-52048-dev-cog-rioai-eus2-1.openai.azure.com/openai/deployments/gpt-5_2025-08-07/chat/completions?api-version=2025-01-01-preview
Method: POST
Headers:
  api-key: f57572c4e8db4f8a8ef2878cabe5fce2
  Content-Type: application/json
```

## 🚀 Build Order (one file at a time)

### Phase 1: Backend Core (files 1-6)
1. **`backend/app/services/svg_parser.py`** — Parse SVG XML → extract polylines, rects, text, metadata
2. **`backend/app/services/axis_calibrator.py`** — Map pixel coords to real values using axis labels
3. **`backend/app/services/chart_detector.py`** — Auto-detect chart type (line/area/bar/dual-tone/log)
4. **`backend/app/services/trend_engine.py`** — Peaks, troughs, momentum, volatility, regime detection
5. **`backend/app/services/anomaly_detector.py`** — Z-score anomaly detection with severity scores
6. **`backend/app/services/llm_narrator.py`** — LLM prompt builder + LSEG API integration

### Phase 2: API Layer (files 7-9)
7. **`backend/app/models/schemas.py`** — Pydantic models for ChartData, Insight, Narrative
8. **`backend/app/main.py`** — FastAPI app with routes: upload, analyze, narrate, compare
9. **`backend/requirements.txt`** — Dependencies

### Phase 3: Frontend (files 10-14)
10. **Frontend scaffold** — React + Vite + Tailwind + dark theme
11. **Upload + Chart Viewer** — Drag-drop SVG, re-render with Recharts
12. **Insights Panel** — Display trends, anomalies, stats as cards
13. **Narrative Editor** — Editable LLM commentary with tone selector
14. **Demo Mode** — Preloaded 7 charts for instant judge engagement

### Phase 4: Polish (files 15-16)
15. **Multi-chart comparison** — Side-by-side analysis + cross-chart narrative
16. **Export + Docker** — PDF/JSON export, docker-compose

## 🏆 Innovation Differentiators (score 5/5)

1. **Confidence scoring** on every extracted value — self-aware about data quality
2. **Auto chart-type detection** — handles all 7 types without user config
3. **Anomaly detection with severity** — "this data point is unusual" markers
4. **Tone-controlled regeneration** — bullish / bearish / neutral / cautious voice
5. **Multi-chart thematic comparison** — cross-cutting narratives across charts
6. **Interactive trend highlighting** — click trend → highlights on chart
7. **Zero hallucination guardrails** — LLM only sees numbers, never invents events
8. **Demo mode** — preloaded with all 7 charts for instant wow-factor

---

**NEXT STEP: Build file #1 — `backend/app/services/svg_parser.py`**
