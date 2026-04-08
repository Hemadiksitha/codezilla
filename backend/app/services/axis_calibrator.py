"""
Axis Calibrator — maps pixel coordinates to real data values.

Uses the extracted axis label positions to build linear (or log)
interpolation functions, then converts every data point.
"""

from __future__ import annotations
import math
import logging
from typing import Optional

from ..models.schemas import (
    ChartData, ChartType, AxisInfo, DataPoint, SeriesData,
)

logger = logging.getLogger(__name__)


def calibrate_chart(chart: ChartData) -> ChartData:
    """
    Convert all pixel-based data points to real values using axis info.
    Also assigns x_label to each point via interpolation.
    """
    is_log = chart.metadata.chart_type == ChartType.LOG_LINE
    is_bar = chart.metadata.chart_type == ChartType.BAR_HORIZONTAL

    if is_bar:
        _calibrate_bar_chart(chart)
    else:
        _calibrate_line_chart(chart, is_log=is_log)

    return chart


# ─── Line / Area / Dual-Tone Calibration ─────────────────────────────────────

def _calibrate_line_chart(chart: ChartData, is_log: bool = False) -> None:
    """Calibrate line-type charts (line, area, dual-tone, log-line)."""

    y_axis = chart.y_axis
    x_axis = chart.x_axis

    if len(y_axis.pixel_positions) < 2 or len(y_axis.values) < 2:
        logger.warning("Insufficient Y-axis labels for calibration — attempting fallback")
        # Fallback: use plot area bounds to estimate a value range
        plot_area = chart.plot_area or {}
        pa_y = plot_area.get("y", 0)
        pa_h = plot_area.get("height", 600)
        if len(y_axis.pixel_positions) == 1 and len(y_axis.values) == 1:
            # With one label we can at least assign that value at that pixel
            # and assume the other end maps to 0
            y_axis.pixel_positions.append(pa_y + pa_h)
            y_axis.values.append(0.0)
        else:
            # No axis labels at all — assign pixel positions as raw values so data isn't lost
            for series in chart.series:
                for dp in series.data_points:
                    dp.value = dp.y_pixel
                    dp.x_label = f"{dp.x_pixel:.0f}px"
            return

    # Build Y-axis interpolation
    y_pixels = y_axis.pixel_positions  # sorted top→bottom (small pixel = high value in SVG)
    y_vals = y_axis.values

    # Build X-axis label interpolation
    x_pixels = x_axis.pixel_positions if x_axis.pixel_positions else []
    x_labels = x_axis.labels if x_axis.labels else []

    for series in chart.series:
        for dp in series.data_points:
            # Convert Y pixel to value
            dp.value = _interpolate_y(dp.y_pixel, y_pixels, y_vals, is_log)

            # Assign X label via interpolation
            dp.x_label = _interpolate_x_label(dp.x_pixel, x_pixels, x_labels)


def _interpolate_y(
    pixel_y: float,
    y_pixels: list[float],
    y_values: list[float],
    is_log: bool = False
) -> float:
    """
    Convert a Y pixel position to a real value.
    SVG Y-axis is inverted: smaller pixel = higher on screen = larger value.
    """
    if len(y_pixels) < 2:
        return 0.0

    if is_log:
        return _interpolate_y_log(pixel_y, y_pixels, y_values)

    # Linear interpolation between nearest axis label positions
    # y_pixels and y_values are sorted by pixel position (top to bottom)
    # Clamp to range
    if pixel_y <= y_pixels[0]:
        # Above the top label
        if len(y_pixels) >= 2:
            slope = (y_values[1] - y_values[0]) / (y_pixels[1] - y_pixels[0])
            return y_values[0] + slope * (pixel_y - y_pixels[0])
        return y_values[0]

    if pixel_y >= y_pixels[-1]:
        # Below the bottom label
        if len(y_pixels) >= 2:
            slope = (y_values[-1] - y_values[-2]) / (y_pixels[-1] - y_pixels[-2])
            return y_values[-1] + slope * (pixel_y - y_pixels[-1])
        return y_values[-1]

    # Find the two bracketing labels
    for i in range(len(y_pixels) - 1):
        if y_pixels[i] <= pixel_y <= y_pixels[i + 1]:
            # Linear interpolation
            t = (pixel_y - y_pixels[i]) / (y_pixels[i + 1] - y_pixels[i])
            return y_values[i] + t * (y_values[i + 1] - y_values[i])

    return 0.0


def _interpolate_y_log(
    pixel_y: float,
    y_pixels: list[float],
    y_values: list[float]
) -> float:
    """
    Logarithmic interpolation for log-scale charts.
    The pixel spacing is linear in log-space.
    """
    if len(y_pixels) < 2:
        return 0.0

    # Filter out zero/negative values (can't take log)
    valid = [(p, v) for p, v in zip(y_pixels, y_values) if v > 0]
    if len(valid) < 2:
        return _interpolate_y(pixel_y, y_pixels, y_values, is_log=False)

    log_values = [(p, math.log10(v)) for p, v in valid]
    pixels = [lv[0] for lv in log_values]
    logs = [lv[1] for lv in log_values]

    # Linear interpolation in log-space
    if pixel_y <= pixels[0]:
        if len(pixels) >= 2:
            slope = (logs[1] - logs[0]) / (pixels[1] - pixels[0])
            log_val = logs[0] + slope * (pixel_y - pixels[0])
            return 10 ** log_val
        return 10 ** logs[0]

    if pixel_y >= pixels[-1]:
        if len(pixels) >= 2:
            slope = (logs[-1] - logs[-2]) / (pixels[-1] - pixels[-2])
            log_val = logs[-1] + slope * (pixel_y - pixels[-1])
            return 10 ** log_val
        return 10 ** logs[-1]

    for i in range(len(pixels) - 1):
        if pixels[i] <= pixel_y <= pixels[i + 1]:
            t = (pixel_y - pixels[i]) / (pixels[i + 1] - pixels[i])
            log_val = logs[i] + t * (logs[i + 1] - logs[i])
            return 10 ** log_val

    return 0.0


def _interpolate_x_label(
    pixel_x: float,
    x_pixels: list[float],
    x_labels: list[str]
) -> str:
    """
    Assign an X-axis label to a data point by finding the nearest label
    or interpolating between them (for year-based labels).
    """
    if not x_pixels or not x_labels:
        return f"{pixel_x:.0f}px"

    if len(x_pixels) != len(x_labels):
        # Fallback
        return f"{pixel_x:.0f}px"

    # Check if labels are numeric (years)
    try:
        numeric_labels = [float(l) for l in x_labels]
        is_numeric = True
    except ValueError:
        is_numeric = False

    if is_numeric and len(x_pixels) >= 2:
        # Interpolate between year labels
        if pixel_x <= x_pixels[0]:
            # Extrapolate left
            slope = (numeric_labels[1] - numeric_labels[0]) / (x_pixels[1] - x_pixels[0])
            val = numeric_labels[0] + slope * (pixel_x - x_pixels[0])
            # Return as year or decimal year
            year = int(val)
            frac = val - year
            if frac < 0.05:
                return str(year)
            # Convert fraction to month approximation
            month = int(frac * 12) + 1
            month = max(1, min(12, month))
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            return f"{month_names[month - 1]} {year}"

        if pixel_x >= x_pixels[-1]:
            slope = (numeric_labels[-1] - numeric_labels[-2]) / (x_pixels[-1] - x_pixels[-2])
            val = numeric_labels[-1] + slope * (pixel_x - x_pixels[-1])
            year = int(val)
            frac = val - year
            if frac < 0.05:
                return str(year)
            month = int(frac * 12) + 1
            month = max(1, min(12, month))
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            return f"{month_names[month - 1]} {year}"

        for i in range(len(x_pixels) - 1):
            if x_pixels[i] <= pixel_x <= x_pixels[i + 1]:
                t = (pixel_x - x_pixels[i]) / (x_pixels[i + 1] - x_pixels[i])
                val = numeric_labels[i] + t * (numeric_labels[i + 1] - numeric_labels[i])
                year = int(val)
                frac = val - year
                if frac < 0.05:
                    return str(year)
                month = int(frac * 12) + 1
                month = max(1, min(12, month))
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                return f"{month_names[month - 1]} {year}"

    # Non-numeric: find nearest label
    min_dist = float("inf")
    nearest = x_labels[0] if x_labels else ""
    for px, label in zip(x_pixels, x_labels):
        dist = abs(pixel_x - px)
        if dist < min_dist:
            min_dist = dist
            nearest = label

    return nearest


# ─── Bar Chart Calibration ────────────────────────────────────────────────────

def _calibrate_bar_chart(chart: ChartData) -> None:
    """
    Calibrate horizontal bar chart.
    X-axis has numeric values (bar lengths), Y-axis has category names.
    """
    x_axis = chart.x_axis  # numeric values (0, 10, 20, ...)
    y_axis = chart.y_axis  # category labels

    # For bar charts, X-axis labels are the value labels
    x_pixels = x_axis.pixel_positions
    x_vals = []
    for label in x_axis.labels:
        try:
            x_vals.append(float(label.replace(",", "").replace("%", "")))
        except ValueError:
            x_vals.append(0)

    # Y-axis labels are the category names
    y_pixels = y_axis.pixel_positions
    y_labels = y_axis.labels

    if not chart.series:
        return

    # Bar chart typically has one series with multiple bars
    series = chart.series[0]
    series.name = chart.metadata.title or "Values"

    calibrated_points = []
    for dp in series.data_points:
        # Convert x_pixel (bar end) to value
        if len(x_pixels) >= 2 and len(x_vals) >= 2:
            value = _interpolate_y(dp.x_pixel, x_pixels, x_vals, is_log=False)
        else:
            value = 0

        # Find nearest Y-axis category label
        label = _find_nearest_category(dp.y_pixel, y_pixels, y_labels)

        calibrated_points.append(
            DataPoint(x_label=label, x_pixel=dp.x_pixel, value=value, y_pixel=dp.y_pixel)
        )

    series.data_points = calibrated_points


def _find_nearest_category(
    pixel_y: float,
    y_pixels: list[float],
    y_labels: list[str]
) -> str:
    """Find the nearest category label for a bar's y-position."""
    if not y_pixels or not y_labels:
        return ""

    min_dist = float("inf")
    nearest = y_labels[0]
    for py, label in zip(y_pixels, y_labels):
        dist = abs(pixel_y - py)
        if dist < min_dist:
            min_dist = dist
            nearest = label

    return nearest

