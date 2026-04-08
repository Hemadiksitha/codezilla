"""
Trend Engine — fully data-driven trend detection.

Detects peaks, troughs, monotonic regimes (rising/falling/flat),
momentum, volatility, and summary statistics. NO LLM involvement.
"""

from __future__ import annotations
import math
import logging
from typing import Optional

import numpy as np
from scipy.signal import argrelextrema

from ..models.schemas import (
    ChartData, ChartInsight, SeriesInsight, SeriesData,
    SummaryStats, TrendSegment, TrendDirection,
    Anomaly, Correlation, DataPoint,
)

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

FLAT_THRESHOLD_PCT = 2.0   # % change below which a segment is "flat"
MIN_SEGMENT_POINTS = 3     # minimum points to form a trend segment
ANOMALY_Z_THRESHOLD = 2.0  # z-score above which a point is anomalous
PEAK_ORDER = 3             # neighborhood size for local extrema


def analyze_chart(chart: ChartData) -> ChartInsight:
    """
    Run full trend analysis on a calibrated chart.
    Returns structured ChartInsight with stats, trends, anomalies, correlations.
    """
    insight = ChartInsight(
        chart_id=chart.chart_id,
        metadata=chart.metadata,
        overall_confidence=chart.confidence,
        plot_area=chart.plot_area,
        x_axis=chart.x_axis,
        y_axis=chart.y_axis,
    )

    for series in chart.series:
        values = np.array([dp.value for dp in series.data_points])
        labels = [dp.x_label for dp in series.data_points]

        # Summary stats (always computed, even for short series)
        stats = _compute_stats(values, labels)

        series_insight = SeriesInsight(
            name=series.name,
            color=series.color,
            stats=stats,
            data_points=series.data_points,
        )
        insight.series.append(series_insight)

        # Trend segments and anomalies require minimum data points
        if len(series.data_points) < 3:
            logger.info(f"Series '{series.name}' has only {len(series.data_points)} points — included with stats only")
            continue

        trends = _detect_trends(series.name, values, labels)
        insight.trends.extend(trends)

        anomalies = _detect_anomalies(series.name, values, labels)
        insight.anomalies.extend(anomalies)

    # Cross-series correlations
    if len(chart.series) >= 2:
        insight.correlations = _compute_correlations(chart.series)

    return insight


# ─── Summary Statistics ───────────────────────────────────────────────────────

def _compute_stats(values: np.ndarray, labels: list[str]) -> SummaryStats:
    """Compute basic statistics for a series."""
    if len(values) == 0:
        return SummaryStats()

    first_val = float(values[0])
    latest_val = float(values[-1])
    change_pct = ((latest_val - first_val) / abs(first_val) * 100) if first_val != 0 else 0

    return SummaryStats(
        min_value=round(float(np.min(values)), 2),
        max_value=round(float(np.max(values)), 2),
        mean_value=round(float(np.mean(values)), 2),
        latest_value=round(latest_val, 2),
        first_value=round(first_val, 2),
        overall_change_pct=round(change_pct, 2),
        data_point_count=len(values),
    )


# ─── Trend Detection ─────────────────────────────────────────────────────────

def _detect_trends(
    series_name: str,
    values: np.ndarray,
    labels: list[str]
) -> list[TrendSegment]:
    """
    Segment the time series into monotonic regimes:
    rising, falling, or flat.
    """
    if len(values) < MIN_SEGMENT_POINTS:
        return []

    # Compute smoothed first derivative (slope)
    # Use a simple moving average to smooth noise
    window = min(5, len(values) // 3)
    if window < 2:
        window = 2

    smoothed = _moving_average(values, window)
    if len(smoothed) < 3:
        return []

    # Compute slopes between consecutive smoothed points
    slopes = np.diff(smoothed)

    # Classify each slope as rising/falling/flat
    directions = []
    value_range = float(np.max(values) - np.min(values))
    flat_abs_threshold = value_range * (FLAT_THRESHOLD_PCT / 100) if value_range > 0 else 0.01

    for s in slopes:
        if abs(s) < flat_abs_threshold:
            directions.append(TrendDirection.FLAT)
        elif s > 0:
            directions.append(TrendDirection.RISING)
        else:
            directions.append(TrendDirection.FALLING)

    # Merge consecutive same-direction segments
    segments = []
    if not directions:
        return []

    current_dir = directions[0]
    start_idx = 0

    for i in range(1, len(directions)):
        if directions[i] != current_dir:
            # End current segment
            end_idx = min(i, len(values) - 1)
            segments.append((current_dir, start_idx, end_idx))
            current_dir = directions[i]
            start_idx = i

    # Final segment
    end_idx = len(values) - 1
    segments.append((current_dir, start_idx, end_idx))

    # Convert to TrendSegment objects, filtering short segments
    trend_segments = []
    for direction, start, end in segments:
        duration = end - start + 1
        if duration < MIN_SEGMENT_POINTS:
            continue

        start_val = float(values[start])
        end_val = float(values[end])
        magnitude = abs(end_val - start_val)
        magnitude_pct = (magnitude / abs(start_val) * 100) if start_val != 0 else 0

        # Detect spikes/dips (short, high-magnitude segments)
        if duration <= 5 and magnitude_pct > 20:
            if end_val > start_val:
                direction = TrendDirection.SPIKE
            else:
                direction = TrendDirection.DIP

        start_label = labels[start] if start < len(labels) else ""
        end_label = labels[end] if end < len(labels) else ""

        trend_segments.append(TrendSegment(
            series_name=series_name,
            direction=direction,
            start_label=start_label,
            end_label=end_label,
            start_value=round(start_val, 2),
            end_value=round(end_val, 2),
            magnitude=round(magnitude, 2),
            magnitude_pct=round(magnitude_pct, 2),
            duration_points=duration,
            confidence=0.9 if duration >= 5 else 0.7,
        ))

    return trend_segments


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average for smoothing."""
    if len(values) < window:
        return values
    cumsum = np.cumsum(values)
    cumsum = np.insert(cumsum, 0, 0)
    return (cumsum[window:] - cumsum[:-window]) / window


# ─── Anomaly Detection ────────────────────────────────────────────────────────

def _detect_anomalies(
    series_name: str,
    values: np.ndarray,
    labels: list[str]
) -> list[Anomaly]:
    """
    Z-score based anomaly detection.
    Points with |z-score| > threshold are flagged as anomalies.
    """
    if len(values) < 5:
        return []

    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return []

    anomalies = []
    z_scores = (values - mean) / std

    for i, (z, val) in enumerate(zip(z_scores, values)):
        if abs(z) > ANOMALY_Z_THRESHOLD:
            severity = min(1.0, abs(z) / 4.0)  # normalize to 0-1
            direction = "unusually high" if z > 0 else "unusually low"
            label = labels[i] if i < len(labels) else f"point {i}"

            anomalies.append(Anomaly(
                series_name=series_name,
                x_label=label,
                value=round(float(val), 2),
                z_score=round(float(z), 2),
                severity=round(severity, 2),
                description=f"{series_name} is {direction} at {label} "
                            f"(z-score: {z:.1f})",
            ))

    return anomalies


# ─── Cross-Series Correlations ────────────────────────────────────────────────

def _compute_correlations(series_list: list[SeriesData]) -> list[Correlation]:
    """Compute Pearson correlation between all series pairs."""
    correlations = []

    for i in range(len(series_list)):
        for j in range(i + 1, len(series_list)):
            s_a = series_list[i]
            s_b = series_list[j]

            # Align by x_pixel (find overlapping range)
            vals_a = np.array([dp.value for dp in s_a.data_points])
            vals_b = np.array([dp.value for dp in s_b.data_points])

            # Use the shorter length
            min_len = min(len(vals_a), len(vals_b))
            if min_len < 5:
                continue

            # Resample to same length if needed
            if len(vals_a) != len(vals_b):
                # Simple approach: use first min_len points
                vals_a = vals_a[:min_len]
                vals_b = vals_b[:min_len]

            # Pearson correlation
            if np.std(vals_a) == 0 or np.std(vals_b) == 0:
                continue

            r = float(np.corrcoef(vals_a, vals_b)[0, 1])

            # Interpretation
            if abs(r) > 0.8:
                strength = "strongly"
            elif abs(r) > 0.5:
                strength = "moderately"
            elif abs(r) > 0.3:
                strength = "weakly"
            else:
                strength = "negligibly"

            direction = "positively" if r > 0 else "negatively"

            correlations.append(Correlation(
                series_a=s_a.name,
                series_b=s_b.name,
                pearson_r=round(r, 3),
                interpretation=f"{s_a.name} and {s_b.name} are {strength} "
                               f"{direction} correlated (r={r:.2f})",
            ))

    return correlations

