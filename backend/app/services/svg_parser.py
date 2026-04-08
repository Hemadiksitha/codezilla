"""
SVG Parser for LSEG Datastream/Fathom Consulting charts.

Extracts polylines, polygons, rects, axis labels, legend entries,
and metadata from the consistent LSEG SVG structure.
"""

from __future__ import annotations
import re
import logging
from typing import Optional
from lxml import etree

from ..models.schemas import (
    ChartData, ChartMetadata, ChartType,
    SeriesData, DataPoint, AxisInfo,
)

logger = logging.getLogger(__name__)

# Namespace for SVG
SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {"svg": SVG_NS}


def parse_svg(svg_content: str | bytes) -> ChartData:
    """
    Main entry: parse raw SVG content into structured ChartData.
    """
    if isinstance(svg_content, str):
        svg_content = svg_content.encode("utf-8")

    tree = etree.fromstring(svg_content)
    chart = ChartData()

    # 1. Extract metadata from trailing HTML comment
    chart.metadata = _extract_metadata(svg_content, tree)

    # 2. Extract CSS styles (needed for color mapping)
    styles = _extract_styles(svg_content)

    # 3. Extract axis information
    chart.y_axis = _extract_y_axis(tree)
    chart.x_axis = _extract_x_axis(tree)

    # 4. Extract plot area from clip-path
    chart.plot_area = _extract_plot_area(tree)

    # 5. Detect chart type
    chart.metadata.chart_type = _detect_chart_type(tree, chart.metadata, styles)

    # 6. Extract data series
    chart.series = _extract_series(tree, styles)

    # 7. Extract legend names and map to series
    legend_entries = _extract_legend(tree, styles)
    _map_legend_to_series(chart.series, legend_entries, styles)

    # 8. Calculate confidence score
    chart.confidence = _calculate_confidence(chart)

    logger.info(
        f"Parsed chart: '{chart.metadata.title}' | "
        f"Type: {chart.metadata.chart_type} | "
        f"Series: {len(chart.series)} | "
        f"Confidence: {chart.confidence:.2f}"
    )

    return chart


# ─── Metadata Extraction ─────────────────────────────────────────────────────

def _extract_metadata(raw: bytes, tree: etree._Element) -> ChartMetadata:
    """Extract metadata from the HTML comment and SVG text elements."""
    meta = ChartMetadata()

    # Title from .s4 text
    meta.title = _get_text_by_class(tree, "s4")
    # Subtitle/units from .s5 text
    meta.subtitle = _get_text_by_class(tree, "s5")
    # Source from .s1 text
    meta.source = _get_text_by_class(tree, "s1")

    # Extract from trailing <!-- <Chart><ImageInfo .../> --> comment
    raw_str = raw.decode("utf-8", errors="replace")
    comment_match = re.search(
        r'<!--\s*<Chart><ImageInfo\s+(.*?)/></Chart>\s*-->', raw_str, re.DOTALL
    )
    if comment_match:
        attrs_str = comment_match.group(1)
        guid_m = re.search(r'GUID="([^"]*)"', attrs_str)
        group_m = re.search(r'GroupName="([^"]*)"', attrs_str)
        chart_m = re.search(r'ChartName="([^"]*)"', attrs_str)
        refresh_m = re.search(r'RefreshDate="([^"]*)"', attrs_str)
        if guid_m:
            meta.guid = guid_m.group(1)
        if group_m:
            meta.group_name = group_m.group(1)
        if chart_m:
            meta.chart_name = chart_m.group(1)
        if refresh_m:
            meta.refresh_date = refresh_m.group(1)

    return meta


# ─── Style Extraction ─────────────────────────────────────────────────────────

def _extract_styles(raw: bytes) -> dict[str, dict]:
    """Parse CSS from <style> block into {class_name: {prop: value}}."""
    raw_str = raw.decode("utf-8", errors="replace")
    styles = {}

    # Find CDATA content inside <style>
    cdata_match = re.search(r'<style[^>]*>\s*<!\[CDATA\[(.*?)\]\]>', raw_str, re.DOTALL)
    if not cdata_match:
        return styles

    css_text = cdata_match.group(1)

    # Parse each .sN { ... } rule
    for m in re.finditer(r'\.(s\d+)\s*\{([^}]+)\}', css_text):
        class_name = m.group(1)
        props_str = m.group(2)
        props = {}
        for prop_match in re.finditer(r'([\w-]+)\s*:\s*([^;]+)', props_str):
            props[prop_match.group(1).strip()] = prop_match.group(2).strip()
        if class_name not in styles:
            styles[class_name] = {}
        styles[class_name].update(props)

    return styles


def _get_color_for_class(styles: dict, class_name: str) -> str:
    """Get stroke color from a CSS class."""
    if class_name in styles:
        return styles[class_name].get("stroke", styles[class_name].get("fill", ""))
    return ""


# ─── Axis Extraction ──────────────────────────────────────────────────────────

def _extract_y_axis(tree: etree._Element) -> AxisInfo:
    """
    Extract Y-axis labels. Found in text-anchor:end groups.
    These are the groups with class containing text-anchor:end in their CSS.
    In LSEG SVGs, Y-axis labels are in the last axis group, inside
    text elements with explicit y positions and transform attributes.
    """
    axis = AxisInfo()

    # Strategy: find all text elements that look like Y-axis labels.
    # Y-axis labels have text-anchor:end parent and are near the left edge.
    # They use transform="rotate(360 ...)" which is effectively no rotation.
    all_text = tree.findall(f".//{{{SVG_NS}}}text")

    # Collect candidates: text with rotate(360) transform near left side
    y_candidates = []
    # Dynamically determine thresholds from plot area
    plot_area_x = 130.0  # default fallback
    plot_area_y = 0.0
    plot_area_bottom = 720.0
    clip_rects = tree.findall(f".//{{{SVG_NS}}}clipPath/{{{SVG_NS}}}rect")
    for rect in clip_rects:
        plot_area_x = float(rect.get("x", "130"))
        plot_area_y = float(rect.get("y", "0"))
        plot_area_bottom = plot_area_y + float(rect.get("height", "600"))
        break
    y_x_threshold = plot_area_x + 10  # labels are to the left of the plot area

    for t in all_text:
        transform = t.get("transform", "")
        if "rotate(360" not in transform:
            continue
        x = float(t.get("x", "0"))
        y_str = t.get("y", "0")
        text_content = (t.text or "").strip()
        if not text_content:
            continue

        # Y-axis labels are to the left of the plot area
        if x < y_x_threshold:
            # Extract the reference y from transform rotate(360 x,ref_y)
            ref_match = re.search(r'rotate\(360\s+[\d.]+,\s*([\d.]+)\)', transform)
            ref_y = float(ref_match.group(1)) if ref_match else float(y_str)
            # Only include labels within the plot area vertical range
            # (prevents X-axis labels below the chart from leaking in)
            if ref_y < plot_area_bottom + 5:
                y_candidates.append((text_content, ref_y))

    if not y_candidates:
        # Fallback: look for left-aligned text with numeric content
        return axis

    # Sort by pixel position (top to bottom)
    y_candidates.sort(key=lambda c: c[1])

    for label, pixel_y in y_candidates:
        axis.labels.append(label)
        axis.pixel_positions.append(pixel_y)
        try:
            axis.values.append(float(label.replace(",", "").replace("%", "")))
        except ValueError:
            # Non-numeric label (e.g., region name in bar chart)
            axis.values.append(0)

    return axis


def _extract_x_axis(tree: etree._Element) -> AxisInfo:
    """
    Extract X-axis labels. Found in text-anchor:middle groups.
    X-axis labels are at the bottom of the chart.
    """
    axis = AxisInfo()

    all_text = tree.findall(f".//{{{SVG_NS}}}text")

    # Dynamically determine y threshold from plot area
    plot_area_bottom = 610.0  # default fallback
    plot_area_top = 0.0
    clip_rects = tree.findall(f".//{{{SVG_NS}}}clipPath/{{{SVG_NS}}}rect")
    for rect in clip_rects:
        pa_y = float(rect.get("y", "0"))
        pa_h = float(rect.get("height", "600"))
        plot_area_top = pa_y
        plot_area_bottom = pa_y + pa_h
        break
    x_y_min = plot_area_bottom - 5   # labels start near the bottom of the plot area
    x_y_max = plot_area_bottom + 60  # labels extend below the plot area

    # X-axis labels: text-anchor:middle groups, y position near bottom
    x_candidates = []

    for t in all_text:
        y_pos = float(t.get("y", "0"))
        x_pos = float(t.get("x", "0"))
        text_content = (t.text or "").strip()

        if not text_content:
            continue

        transform = t.get("transform", "")

        # X-axis labels are near the bottom of the plot area
        if x_y_min <= y_pos <= x_y_max and "rotate(360" not in transform:
            x_candidates.append((text_content, x_pos))
        # For bar charts: x-axis labels at bottom with rotate(360) transform
        elif (x_y_min + 30) <= y_pos <= x_y_max and "rotate(360" in transform:
            ref_match = re.search(r'rotate\(360\s+([\d.]+)', transform)
            ref_x = float(ref_match.group(1)) if ref_match else x_pos
            x_candidates.append((text_content, ref_x))

    if not x_candidates:
        return axis

    # Sort by pixel position (left to right)
    x_candidates.sort(key=lambda c: c[1])

    for label, pixel_x in x_candidates:
        axis.labels.append(label)
        axis.pixel_positions.append(pixel_x)

    return axis


# ─── Plot Area ────────────────────────────────────────────────────────────────

def _extract_plot_area(tree: etree._Element) -> dict:
    """Extract the clip-path rect that defines the chart plotting area."""
    clip_rects = tree.findall(f".//{{{SVG_NS}}}clipPath/{{{SVG_NS}}}rect")
    for rect in clip_rects:
        return {
            "x": float(rect.get("x", "0")),
            "y": float(rect.get("y", "0")),
            "width": float(rect.get("width", "0")),
            "height": float(rect.get("height", "0")),
        }
    return {"x": 0, "y": 0, "width": 960, "height": 720}


# ─── Chart Type Detection ─────────────────────────────────────────────────────

def _detect_chart_type(
    tree: etree._Element,
    metadata: ChartMetadata,
    styles: dict
) -> ChartType:
    """Auto-detect chart type from SVG structure and metadata."""

    # Check subtitle for log scale
    subtitle = metadata.subtitle.lower()
    if "log scale" in subtitle or "log" in subtitle:
        return ChartType.LOG_LINE

    # Find the clip-path data group
    clip_group = _find_clip_group(tree)
    if clip_group is None:
        return ChartType.LINE

    # Check for <rect> elements inside data series (bar chart)
    rects_in_data = clip_group.findall(f".//{{{SVG_NS}}}rect")
    polylines_in_data = clip_group.findall(f".//{{{SVG_NS}}}polyline")
    polygons_in_data = clip_group.findall(f".//{{{SVG_NS}}}polygon")

    if rects_in_data and not polylines_in_data:
        return ChartType.BAR_HORIZONTAL

    # Check for dual-tone (alternating style classes on polylines in same series group)
    series_groups = clip_group.findall(f".//{{{SVG_NS}}}g[@id]")
    if series_groups:
        first_series = series_groups[0]
        child_groups = first_series.findall(f"{{{SVG_NS}}}g")
        classes = [g.get("class", "") for g in child_groups if g.get("class")]
        # Dual-tone: same series has alternating classes (e.g., s9, s10, s9, s10...)
        unique_classes = list(dict.fromkeys(classes))  # preserve order, remove dupes
        if len(unique_classes) >= 2 and len(classes) > 4:
            # Check if classes alternate between two values
            class_set = set(classes)
            if len(class_set) == 2:
                return ChartType.DUAL_TONE

    # Check for polygons (area chart)
    if polygons_in_data:
        return ChartType.AREA

    return ChartType.LINE


def _find_clip_group(tree: etree._Element) -> Optional[etree._Element]:
    """Find the g element with clip-path:url(#c0)."""
    for g in tree.findall(f".//{{{SVG_NS}}}g"):
        style = g.get("style", "")
        if "clip-path" in style:
            return g
    return None


# ─── Series Extraction ────────────────────────────────────────────────────────

def _extract_series(tree: etree._Element, styles: dict) -> list[SeriesData]:
    """Extract all data series from the clip-path group."""
    clip_group = _find_clip_group(tree)
    if clip_group is None:
        return []

    series_list = []

    # Find all series groups — look for g elements with an id attribute
    # First try direct children with id starting with 'n', then broaden to all id'd groups
    series_groups = [
        g for g in clip_group.findall(f"{{{SVG_NS}}}g")
        if g.get("id", "").startswith("n")
    ]
    # Fallback: if no 'n*' groups found, use all direct child g elements with an id
    if not series_groups:
        series_groups = [
            g for g in clip_group.findall(f"{{{SVG_NS}}}g")
            if g.get("id", "")
        ]
    # Also search recursively for nested series groups
    if not series_groups:
        series_groups = [
            g for g in clip_group.findall(f".//{{{SVG_NS}}}g")
            if g.get("id", "").startswith("n")
        ]

    for sg in series_groups:
        series = SeriesData(name=sg.get("id", "unknown"))

        # Get the main style class for color
        child_groups = sg.findall(f"{{{SVG_NS}}}g")
        main_class = ""

        for cg in child_groups:
            cg_class = cg.get("class", "")

            # Extract polyline points (search recursively to handle deeper nesting)
            polylines = cg.findall(f".//{{{SVG_NS}}}polyline")
            if not polylines:
                polylines = cg.findall(f"{{{SVG_NS}}}polyline")
            for pl in polylines:
                points_str = pl.get("points", "")
                if points_str:
                    if not main_class:
                        main_class = cg_class
                    points = _parse_points(points_str)
                    for x, y in points:
                        series.data_points.append(
                            DataPoint(x_label="", x_pixel=x, value=0, y_pixel=y)
                        )

            # Extract rect elements (bar charts)
            rects = cg.findall(f"{{{SVG_NS}}}rect")
            for rect in rects:
                x = float(rect.get("x", "0"))
                y = float(rect.get("y", "0"))
                width = float(rect.get("width", "0"))
                height = float(rect.get("height", "0"))
                # For horizontal bars, the "value" dimension is width
                series.data_points.append(
                    DataPoint(
                        x_label="", x_pixel=x + width,
                        value=0, y_pixel=y + height / 2
                    )
                )
                if not main_class:
                    main_class = cg_class

            # Check for polygon (area fill)
            polygons = cg.findall(f"{{{SVG_NS}}}polygon")
            if polygons:
                series.is_area = True

        # Set color from style class
        if main_class:
            series.color = _get_color_for_class(styles, main_class)

        if series.data_points:
            # Remove duplicate pixel positions (from multiple polyline segments)
            series.data_points = _deduplicate_points(series.data_points)
            series_list.append(series)

    return series_list


def _parse_points(points_str: str) -> list[tuple[float, float]]:
    """Parse SVG points attribute 'x1,y1 x2,y2 ...' into list of (x, y)."""
    points = []
    for pair in points_str.strip().split():
        parts = pair.split(",")
        if len(parts) == 2:
            try:
                points.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return points


def _deduplicate_points(points: list[DataPoint]) -> list[DataPoint]:
    """Remove truly duplicate points (same x_pixel AND y_pixel), preserving all distinct values."""
    seen = set()
    unique = []
    for p in points:
        key = (round(p.x_pixel, 2), round(p.y_pixel, 2))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return sorted(unique, key=lambda p: p.x_pixel)


# ─── Legend Extraction ────────────────────────────────────────────────────────

def _extract_legend(tree: etree._Element, styles: dict) -> list[dict]:
    """
    Extract legend entries: [{name, color_class}].
    Legend is in a group after the axis groups, containing short polylines
    paired with text labels.
    """
    entries = []

    # Legend text elements are typically at y > 650 and have series names
    # They are siblings to short colored polylines in the legend group
    all_groups = tree.findall(f".//{{{SVG_NS}}}g")

    for g in all_groups:
        children = list(g)
        # Legend groups have pattern: rect, [polyline_g, marker_g, text_g] repeating
        text_elements = g.findall(f".//{{{SVG_NS}}}text")
        polyline_groups = [
            c for c in children
            if c.tag == f"{{{SVG_NS}}}g" and c.findall(f"{{{SVG_NS}}}polyline")
        ]

        # Must have both text and polylines, and text at y > 650
        if not text_elements or not polyline_groups:
            continue

        # Check if texts are in legend area (y > 650)
        legend_texts = [
            t for t in text_elements
            if float(t.get("y", "0")) > 650 and (t.text or "").strip()
        ]

        if len(legend_texts) < 1:
            continue

        # Pair each text with the preceding polyline's class
        for text_el in legend_texts:
            name = (text_el.text or "").strip()
            if not name:
                continue

            # Find the polyline group that precedes this text's parent
            text_parent = text_el.getparent()
            if text_parent is None:
                continue

            # Look backwards from this text parent to find the polyline group
            prev_siblings = []
            for child in children:
                if child is text_parent:
                    break
                prev_siblings.append(child)

            # The color class comes from the polyline group
            color_class = ""
            for ps in reversed(prev_siblings):
                ps_class = ps.get("class", "")
                if ps_class and ps.findall(f"{{{SVG_NS}}}polyline"):
                    color_class = ps_class
                    break
                # Check for rect-based legend markers
                if ps_class and ps.findall(f"{{{SVG_NS}}}polygon"):
                    color_class = ps_class
                    break

            entries.append({"name": name, "color_class": color_class})

    return entries


def _map_legend_to_series(
    series_list: list[SeriesData],
    legend_entries: list[dict],
    styles: dict
):
    """Map legend names to series by matching color classes."""
    if not legend_entries:
        return

    for i, entry in enumerate(legend_entries):
        if i < len(series_list):
            series_list[i].name = entry["name"]
            if entry["color_class"] and not series_list[i].color:
                series_list[i].color = _get_color_for_class(
                    styles, entry["color_class"]
                )


# ─── Confidence Calculation ───────────────────────────────────────────────────

def _calculate_confidence(chart: ChartData) -> float:
    """
    Score 0-1 based on extraction quality:
    - Axis labels found
    - Series data points recovered
    - Metadata completeness
    """
    score = 0.0
    max_score = 0.0

    # Title extracted
    max_score += 1
    if chart.metadata.title:
        score += 1

    # Y-axis labels
    max_score += 2
    if chart.y_axis.labels:
        score += 1
        if len(chart.y_axis.labels) >= 3:
            score += 1

    # X-axis labels
    max_score += 2
    if chart.x_axis.labels:
        score += 1
        if len(chart.x_axis.labels) >= 3:
            score += 1

    # Series extracted
    max_score += 2
    if chart.series:
        score += 1
        if all(len(s.data_points) > 5 for s in chart.series):
            score += 1

    # Series have names
    max_score += 1
    if chart.series and all(
        not s.name.startswith("n") for s in chart.series
    ):
        score += 1

    # Metadata GUID
    max_score += 1
    if chart.metadata.guid:
        score += 1

    return score / max_score if max_score > 0 else 0.0


# ─── Helper ───────────────────────────────────────────────────────────────────

def _get_text_by_class(tree: etree._Element, class_name: str) -> str:
    """Find first text content inside a group with given CSS class."""
    for g in tree.findall(f".//{{{SVG_NS}}}g"):
        if g.get("class") == class_name:
            text_el = g.find(f"{{{SVG_NS}}}text")
            if text_el is not None and text_el.text:
                return text_el.text.strip()
            # Check nested text
            for child in g:
                text_el = child.find(f"{{{SVG_NS}}}text")
                if text_el is not None and text_el.text:
                    return text_el.text.strip()
    return ""

