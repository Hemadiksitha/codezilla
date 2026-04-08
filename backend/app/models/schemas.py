"""Pydantic models for NOVA-C structured chart data and insights."""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─── Enums ───────────────────────────────────────────────────────────────────

class ChartType(str, Enum):
    LINE = "line"
    AREA = "area"
    BAR_HORIZONTAL = "bar_horizontal"
    DUAL_TONE = "dual_tone"  # positive/negative shading (e.g., chart 07)
    LOG_LINE = "log_line"


class TrendDirection(str, Enum):
    RISING = "rising"
    FALLING = "falling"
    FLAT = "flat"
    SPIKE = "spike"
    DIP = "dip"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ─── Raw Parsed Data ─────────────────────────────────────────────────────────

class DataPoint(BaseModel):
    x_label: str
    x_pixel: float
    value: float
    y_pixel: float


class SeriesData(BaseModel):
    name: str
    color: str = ""
    data_points: list[DataPoint] = []
    is_area: bool = False  # has polygon fill


class AxisInfo(BaseModel):
    labels: list[str] = []
    pixel_positions: list[float] = []
    values: list[float] = []  # numeric values for Y-axis


class ChartMetadata(BaseModel):
    title: str = ""
    subtitle: str = ""  # contains units info
    source: str = ""
    chart_type: ChartType = ChartType.LINE
    guid: str = ""
    group_name: str = ""
    chart_name: str = ""
    refresh_date: str = ""


class ChartData(BaseModel):
    """Complete parsed chart data from SVG."""
    chart_id: str = ""
    metadata: ChartMetadata = Field(default_factory=ChartMetadata)
    x_axis: AxisInfo = Field(default_factory=AxisInfo)
    y_axis: AxisInfo = Field(default_factory=AxisInfo)
    series: list[SeriesData] = []
    plot_area: dict = Field(default_factory=dict)  # x, y, width, height
    confidence: float = 1.0  # 0-1 extraction confidence


# ─── Trend / Insight Models ──────────────────────────────────────────────────

class SummaryStats(BaseModel):
    min_value: float = 0
    max_value: float = 0
    mean_value: float = 0
    latest_value: float = 0
    first_value: float = 0
    overall_change_pct: float = 0
    data_point_count: int = 0


class TrendSegment(BaseModel):
    series_name: str
    direction: TrendDirection
    start_label: str
    end_label: str
    start_value: float
    end_value: float
    magnitude: float  # absolute change
    magnitude_pct: float  # percentage change
    duration_points: int
    confidence: float = 1.0


class Anomaly(BaseModel):
    series_name: str
    x_label: str
    value: float
    z_score: float
    severity: float  # 0-1
    description: str = ""


class Correlation(BaseModel):
    series_a: str
    series_b: str
    pearson_r: float
    interpretation: str = ""


class SeriesInsight(BaseModel):
    name: str
    color: str = ""
    stats: SummaryStats = Field(default_factory=SummaryStats)
    data_points: list[DataPoint] = []


class ChartInsight(BaseModel):
    """Full structured insight for a single chart."""
    chart_id: str = ""
    metadata: ChartMetadata = Field(default_factory=ChartMetadata)
    series: list[SeriesInsight] = []
    trends: list[TrendSegment] = []
    anomalies: list[Anomaly] = []
    correlations: list[Correlation] = []
    overall_confidence: float = 1.0
    plot_area: dict = Field(default_factory=dict)
    x_axis: AxisInfo = Field(default_factory=AxisInfo)
    y_axis: AxisInfo = Field(default_factory=AxisInfo)


# ─── Narrative Models ────────────────────────────────────────────────────────

class Narrative(BaseModel):
    summary: str = ""  # 2-3 sentence headline
    detailed: str = ""  # Full analyst paragraph
    key_takeaways: list[str] = []
    tone: str = "neutral"  # neutral, bullish, bearish, cautious


class NewsEvent(BaseModel):
    """A real-world news headline related to chart data."""
    headline: str = ""
    snippet: str = ""
    url: str = ""
    date_hint: str = ""
    search_query: str = ""


class ChartAnalysisResult(BaseModel):
    """Final combined result: insights + narrative."""
    insight: ChartInsight = Field(default_factory=ChartInsight)
    narrative: Narrative = Field(default_factory=Narrative)
    news_events: list[NewsEvent] = []

