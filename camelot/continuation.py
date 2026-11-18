"""Cross-page logical table collection (``continuation='auto'``).

This module implements the optional continuation logic that can be enabled
through :func:`camelot.read_pdf`.  It is deliberately kept separate from the
single-page parsers (``stream``/``lattice``) so that the default physical
extraction behaviour is never modified.

The design is evidence based.  Every physical :class:`~camelot.core.Table`
receives a :class:`FragmentContext` describing evidence collected *during page
parsing*:

* original PDF page number and PDF coordinates (left-bottom origin),
* column/row geometry of the fragment,
* the multi-row header detected inside the fragment,
* table-number / "continued" / chapter text found near the fragment but
  outside of its bounding box.

A second stage (:func:`build_logical_tables`) walks the fragments in the
original page order and decides which fragments are continuations of the same
logical table.  It only joins fragments on adjacent selected pages, never
fragments of the same page, and records explicit diagnostics for blocked
edges and ambiguous competition between predecessor fragments.

A third stage (:func:`reassemble_logical_table`) rebuilds every fragment of a
logical table against the column layout established by the first fragment, so
that continuation pages where a whole column is empty (stream) or a vertical
line is interrupted (lattice) keep the canonical columns.  The rebuild uses
the original page text lines and their PDF geometry -- it never relies on the
temporary single-page PDF files, which are deleted before this stage runs.

The grouping strategy is intentionally a *local adjacency* strategy with a
small joint view of one page boundary at a time instead of a global
optimisation over every fragment pair.  Local adjacency keeps the cost linear
in the number of fragments (no Cartesian product of pages/cells), makes the
reason for each accept/reject decision explainable, and matches the document
structure where continuations only ever reach the immediately following
page.  The trade-off is that a fragment separated from its predecessor by a
page that was not selected cannot be re-attached: the missing-page boundary
is reported instead of guessed, which is the conservative choice required to
avoid false merges.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import pandas as pd

from .core import Table


# ---------------------------------------------------------------------------
# Normalisation (used for comparisons only, never to rewrite retained values)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Collapse whitespace/newlines for *comparison* purposes.

    Data values kept inside logical tables are never passed through this
    function; it is only used to compare headers, captions and body text.
    """
    if value is None:
        return ""
    return _WS_RE.sub(" ", str(value)).strip().lower()


# Table number captions, e.g. "Table 3", "Table 3:", "Tbl 3.2", "表 3", "表3-2"
_TABLE_ID_RE = re.compile(
    r"^(?:table|tbl\.?|tab\.?|表)\s*[:：]?\s*"
    r"([0-9]+(?:[.\-][0-9]+)*|[ivxlcdm]+|[一二三四五六七八九十百]+(?:\.[0-9]+)*)",
    re.IGNORECASE,
)
# Continuation markers, e.g. "Table 3 (continued)", "续表 3", "表3（续）"
_CONTINUED_RE = re.compile(
    r"(continued|cont\.?|续\s*表|续表|承前|接上页|接上表)",
    re.IGNORECASE,
)
# Chapter / section switches, e.g. "Chapter 2", "Section 3.1", "第二章", "第三节"
_SECTION_RE = re.compile(
    r"^(?:chapter|section|§|part|第\s*[0-9一二三四五六七八九十百]+\s*[章节]|"
    r"[0-9]+(?:\.[0-9]+)+\s+\S)",
    re.IGNORECASE,
)

# Vertical search window above a fragment for outside captions, in points.
CAPTION_GAP = 72.0
# Horizontal slack when matching a caption line with a table bbox.
CAPTION_X_SLACK = 72.0
# Vertical guard separating a caption line from the table body below it.
ROW_GUARD = 12.0
# Typical table row height, used to widen bbox searches for clipped headers.
ROW_HEIGHT = 20.0
# Column geometry tolerances (points and fraction of column width).
COL_MATCH_TOL_PTS = 6.0
COL_MATCH_TOL_FRAC = 0.35


# ---------------------------------------------------------------------------
# Evidence objects
# ---------------------------------------------------------------------------


@dataclass
class CaptionInfo:
    """A text line found outside a fragment, near its top edge.

    Attributes
    ----------
    text : str
        Original text of the line (not normalised).
    bbox : tuple of float
        ``(x0, y0, x1, y1)`` in the page PDF coordinate space.
    kind : str
        One of ``"table_id"``, ``"continued"``, ``"section"``,
        ``"other"``.
    table_id : str or None
        Normalised table identifier extracted from the line, if any.
    """

    text: str
    bbox: tuple[float, float, float, float]
    kind: str
    table_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON friendly representation."""
        return {
            "text": self.text,
            "bbox": list(self.bbox),
            "kind": self.kind,
            "table_id": self.table_id,
        }


@dataclass
class FragmentContext:
    """Cross-page evidence attached to every physical :class:`Table`.

    The object is exposed as ``table.fragment_context`` whenever the PDF was
    read with ``continuation='auto'``.  All coordinates use the original PDF
    page number and its left-bottom PDF coordinate system; temporary
    single-page file paths are never stored here.

    Attributes
    ----------
    page : int
        Original PDF page number (1-based).
    order : int
        Physical table order on the page (1-based).
    flavor : str
        Parser flavor that produced the fragment (``"stream"`` or
        ``"lattice"``).
    bbox : tuple of float
        Fragment bounding box ``(x0, y0, x1, y1)`` in page coordinates.
    pdf_size : tuple of float
        ``(width, height)`` of the original PDF page.
    col_anchors : list of float
        Sorted column boundary x coordinates, including the outer edges.
    col_centers : list of float
        Center x coordinate of every physical column.
    row_bands : list of tuple
        ``(y0, y1)`` of every physical row, top to bottom.
    n_cols, n_rows : int
        Physical fragment dimensions.
    header_depth : int
        Number of leading physical rows recognised as the (possibly
        multi-row) header.
    header_rows : list of list of str
        Raw cell text of the header rows.
    header_normalized : list of str
        Normalised non-empty header cell texts (comparison only).
    caption : CaptionInfo or None
        Nearest table-number / continued caption above the fragment.
    continuation_marker : bool
        Whether a continuation wording was found near the fragment.
    section_heading : CaptionInfo or None
        Chapter/section heading found between this fragment and the page top
        or the previous fragment.
    """

    page: int
    order: int
    flavor: str
    bbox: tuple[float, float, float, float]
    pdf_size: tuple[float, float]
    col_anchors: list[float]
    col_centers: list[float]
    row_bands: list[tuple[float, float]]
    n_cols: int
    n_rows: int
    header_depth: int = 1
    header_rows: list[list[str]] = field(default_factory=list)
    header_normalized: list[str] = field(default_factory=list)
    caption: CaptionInfo | None = None
    continuation_marker: bool = False
    section_heading: CaptionInfo | None = None

    @property
    def key(self) -> tuple[int, int]:
        """Identity key ``(page, order)`` of the fragment."""
        return (self.page, self.order)

    @property
    def table_id(self) -> str | None:
        """Normalised table id of the nearby caption, if any."""
        return self.caption.table_id if self.caption is not None else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON friendly representation of the evidence."""
        return {
            "page": self.page,
            "order": self.order,
            "flavor": self.flavor,
            "bbox": list(self.bbox),
            "pdf_size": list(self.pdf_size),
            "col_anchors": list(self.col_anchors),
            "col_centers": list(self.col_centers),
            "row_bands": [list(b) for b in self.row_bands],
            "n_cols": self.n_cols,
            "n_rows": self.n_rows,
            "header_depth": self.header_depth,
            "header_rows": [list(r) for r in self.header_rows],
            "header_normalized": list(self.header_normalized),
            "caption": self.caption.to_dict() if self.caption else None,
            "continuation_marker": self.continuation_marker,
            "section_heading": (
                self.section_heading.to_dict() if self.section_heading else None
            ),
        }


# ---------------------------------------------------------------------------
# Stage 1 -- per-page evidence collection
# ---------------------------------------------------------------------------


def _line_bold(textline: Any) -> bool:
    """Return True when a pdfminer text line is rendered in bold."""
    chars = [obj for obj in getattr(textline, "_objs", []) if hasattr(obj, "fontname")]
    if not chars:
        return False
    bold = sum("bold" in (getattr(c, "fontname", "") or "").lower() for c in chars)
    return bold * 2 >= len(chars)


def _classify_caption(text: str) -> tuple[str, str | None]:
    """Classify an outside text line."""
    stripped = text.strip()
    id_match = _TABLE_ID_RE.match(stripped)
    table_id = id_match.group(1) if id_match else None
    if _CONTINUED_RE.search(stripped):
        return "continued", table_id
    if id_match:
        return "table_id", table_id
    if _SECTION_RE.match(stripped):
        return "section", None
    return "other", None


def _find_outside_lines(
    bbox: tuple[float, float, float, float],
    textlines: list[Any],
) -> list[Any]:
    """Return text lines above ``bbox`` inside the caption search window.

    Stream fragments can include the caption inside the detected bbox, so
    the upper bound is the table top plus the gap window rather than a
    strict outside test.
    """
    x0, _y0, x1, y1 = bbox
    candidates = []
    for tl in textlines:
        if not tl.get_text().strip():
            continue
        cx = (tl.x0 + tl.x1) / 2.0
        if tl.y1 <= y1 + CAPTION_GAP:
            if (
                x0 - CAPTION_X_SLACK <= cx <= x1 + CAPTION_X_SLACK
                or (tl.x0 <= x1 + CAPTION_X_SLACK and tl.x1 >= x0 - CAPTION_X_SLACK)
            ):
                candidates.append(tl)
    candidates.sort(key=lambda t: (-t.y1, t.x0))
    return candidates


def _detect_header(
    table: Any, horizontal_text: list[Any]
) -> tuple[int, list[list[str]]]:
    """Detect contiguous bold header rows at the top of a fragment.

    Stream fragments can contain outside caption lines above the actual
    table header, so the scan starts at the first bold physical row rather
    than the first physical row.  Falls back to a single header row when no
    bold row is present.
    """
    data = table.data
    if not data:
        return 0, []

    def row_is_caption_band(band_lines):
        if not band_lines:
            return False
        for tl in band_lines:
            kind, _ = _classify_caption(tl.get_text().strip())
            if kind in ("table_id", "continued", "section"):
                return True
        return False

    # Group the page text lines into horizontal y-bands ourselves: the
    # parser's row boundaries are joined coordinates that need not contain
    # any individual text line center.
    text_bands = _row_bands_from_textlines(
        [tl for tl in horizontal_text if tl.get_text().strip()]
    )
    any_bold = any(_line_bold(tl) for tl in horizontal_text)
    depth = 0
    first_bold = None
    if any_bold:
        for i, band_lines in enumerate(text_bands[: max(3, len(data))]):
            if (
                all(_line_bold(tl) for tl in band_lines)
                and not row_is_caption_band(band_lines)
            ):
                first_bold = i
                break
        if first_bold is not None:
            for band_lines in text_bands[
                first_bold : first_bold + 3
            ]:
                if (
                    all(_line_bold(tl) for tl in band_lines)
                    and not row_is_caption_band(band_lines)
                ):
                    depth += 1
                else:
                    break

    if depth == 0:
        return 1, [data[0]]

    # Map the detected bold text bands back to physical data rows by
    # comparing normalised cell content (text bands and data rows share the
    # same top-to-bottom order).
    header_rows: list[list[str]] = []
    bold_texts = [
        sorted(normalize_text(tl.get_text()) for tl in text_bands[first_bold + h])
        for h in range(depth)
    ]
    for row in data:
        row_texts = sorted(normalize_text(v) for v in row if normalize_text(v))
        if row_texts in bold_texts:
            header_rows.append(row)
        if len(header_rows) == depth:
            break
    if len(header_rows) != depth:
        # Fallback: assume the bold bands are the leading non-caption rows.
        header_rows = data[:depth]
    return len(header_rows), header_rows


def build_fragment_context(
    table: Any,
    horizontal_text: list[Any],
    vertical_text: list[Any],
) -> FragmentContext:
    """Collect cross-page evidence for one physical ``table``.

    Called from the page parser worker, so everything returned must be
    picklable and must not reference the temporary single-page PDF path.
    """
    bbox = tuple(float(v) for v in table._bbox)
    cols = [(float(a), float(b)) for a, b in table.cols]
    col_centers = [(a + b) / 2.0 for a, b in cols]
    # Boundary anchors: outer edges plus the shared boundary between columns.
    anchors = [cols[0][0]]
    anchors.extend((cols[i - 1][1] + cols[i][0]) / 2.0 for i in range(1, len(cols)))
    anchors.append(cols[-1][1])
    row_bands = [(float(a), float(b)) for a, b in table.rows]

    header_depth, header_rows = _detect_header(table, horizontal_text)
    header_norm = [
        normalize_text(cell)
        for row in header_rows
        for cell in row
        if normalize_text(cell)
    ]

    # Determine the y level of the actual table top: the first bold header
    # row when bold text exists, otherwise the top physical row.  Captions
    # sit above that level and may themselves be inside the stream bbox.
    top_band = table.rows[0]
    bold_band_y = None
    if any(_line_bold(tl) for tl in horizontal_text):
        for i in range(min(header_depth, len(table.rows))):
            band = table.rows[i]
            band_lines = [
                tl
                for tl in horizontal_text
                if band[0] <= (tl.y0 + tl.y1) / 2.0 <= band[1]
                and tl.get_text().strip()
            ]
            if band_lines and all(_line_bold(tl) for tl in band_lines):
                bold_band_y = (band[0] + band[1]) / 2.0
                break
    header_y = bold_band_y if bold_band_y is not None else (
        top_band[0] + top_band[1]
    ) / 2.0

    caption = None
    continuation_marker = False
    section_heading = None
    for tl in _find_outside_lines(bbox, horizontal_text):
        text = tl.get_text().strip()
        kind, table_id = _classify_caption(text)
        # Skip lines that are actually table rows (at/below header level).
        if (tl.y0 + tl.y1) / 2.0 <= header_y + 2:
            continue
        caption_center = (tl.x0 + tl.x1) / 2.0
        horizontally_related = (
            bbox[0] - CAPTION_X_SLACK <= caption_center <= bbox[2] + CAPTION_X_SLACK
            or (
                tl.x0 >= bbox[0] - CAPTION_X_SLACK
                and tl.x1 <= bbox[2] + CAPTION_X_SLACK
            )
        )
        info = CaptionInfo(
            text=text,
            bbox=(float(tl.x0), float(tl.y0), float(tl.x1), float(tl.y1)),
            kind=kind,
            table_id=table_id,
        )
        if kind == "section" and section_heading is None:
            section_heading = info
        elif kind in ("table_id", "continued") and caption is None:
            if horizontally_related:
                caption = info
                if kind == "continued" or _CONTINUED_RE.search(text):
                    continuation_marker = True
        if caption is not None and section_heading is not None:
            break

    return FragmentContext(
        page=int(table.page),
        order=int(table.order),
        flavor=str(table.flavor),
        bbox=bbox,
        pdf_size=(float(table.pdf_size[0]), float(table.pdf_size[1])),
        col_anchors=anchors,
        col_centers=col_centers,
        row_bands=row_bands,
        n_cols=len(cols),
        n_rows=len(row_bands),
        header_depth=header_depth,
        header_rows=header_rows,
        header_normalized=header_norm,
        caption=caption,
        continuation_marker=continuation_marker,
        section_heading=section_heading,
    )


# ---------------------------------------------------------------------------
# Stage 1b -- splitting physical fragments merged on the same page
# ---------------------------------------------------------------------------


def _row_bands_from_textlines(textlines):
    """Group page text lines into horizontal row bands by y0 proximity."""
    items = [tl for tl in textlines if tl.get_text().strip()]
    items.sort(key=lambda tl: -tl.y0)
    bands: list[list[Any]] = []
    current: list[Any] = []
    current_y = None
    for tl in items:
        if current_y is None or math.isclose(tl.y0, current_y, abs_tol=3.0):
            current.append(tl)
            current_y = tl.y0 if current_y is None else current_y
        else:
            bands.append(current)
            current = [tl]
            current_y = tl.y0
    if current:
        bands.append(current)
    return bands


def _split_points_from_gaps(merged: Any, page_lines: list[Any]) -> list[float]:
    """Detect two tall, clearly separated column groups without captions.

    This conservative fallback only fires when both sides of the wide gap
    contain text on most row bands: ordinary inter-column whitespace inside
    one table never reaches that coverage.
    """
    bbox = merged._bbox
    bands = [
        [
            tl
            for tl in band
            if tl.get_text().strip()
            and bbox[0] - 2 <= (tl.x0 + tl.x1) / 2 <= bbox[2] + 2
            and bbox[1] - 2 <= (tl.y0 + tl.y1) / 2 <= bbox[3] + 2
        ]
        for band in _row_bands_from_textlines(page_lines)
    ]
    bands = [band for band in bands if band]
    if len(bands) < 4:
        return []

    occupied = sorted(
        (tl.x0, tl.x1) for band in bands for tl in band
    )
    intervals: list[tuple[float, float]] = []
    for x0, x1 in occupied:
        if intervals and x0 <= intervals[-1][1] + 2:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], x1))
        else:
            intervals.append((x0, x1))
    if len(intervals) < 2:
        return []

    for left, right in zip(intervals, intervals[1:]):
        gap = right[0] - left[1]
        if gap < 30.0:
            continue
        left_rows = sum(
            any(left[0] <= (tl.x0 + tl.x1) / 2 <= left[1] for tl in band)
            for band in bands
        )
        right_rows = sum(
            any(right[0] <= (tl.x0 + tl.x1) / 2 <= right[1] for tl in band)
            for band in bands
        )
        coverage = min(left_rows, right_rows) / len(bands)
        if coverage >= 0.75:
            return [(left[1] + right[0]) / 2.0]
    return []


def _caption_centers(
    merged: Any, page_lines: list[Any]
) -> list[float]:
    """Return horizontal centers of table-id/continued captions above.

    Stream's detected bbox can extend above the real table (its padding
    includes the caption), so caption membership is decided by text
    classification, horizontal alignment and proximity rather than strict
    bbox containment.
    """
    bbox = merged._bbox

    # The table header is the highest bold non-caption line cluster; the
    # real table top is its upper edge.  Captions are bold too but sit on a
    # higher y level.
    def horizontally_aligned(tl):
        center = (tl.x0 + tl.x1) / 2.0
        return (
            bbox[0] - CAPTION_X_SLACK <= center <= bbox[2] + CAPTION_X_SLACK
        )

    caption_kinds = ("table_id", "continued")
    bold_body = [
        tl
        for tl in page_lines
        if tl.get_text().strip()
        and horizontally_aligned(tl)
        and _line_bold(tl)
        and _classify_caption(tl.get_text().strip())[0] not in caption_kinds
    ]
    if bold_body:
        body_top = max(tl.y1 for tl in bold_body)
    else:
        body_ys = [
            tl.y1
            for tl in page_lines
            if tl.get_text().strip()
            and horizontally_aligned(tl)
            and _classify_caption(tl.get_text().strip())[0]
            not in caption_kinds
        ]
        body_top = max(body_ys) if body_ys else bbox[3]

    captions = []
    for tl in page_lines:
        text = tl.get_text().strip()
        if not text:
            continue
        kind, _table_id = _classify_caption(text)
        if kind not in caption_kinds or not horizontally_aligned(tl):
            continue
        center_y = (tl.y0 + tl.y1) / 2.0
        # Caption sits strictly above the bold header, within one window.
        if body_top < center_y <= body_top + CAPTION_GAP:
            captions.append(tl)
    captions.sort(key=lambda tl: tl.x0)
    return [(tl.x0 + tl.x1) / 2.0 for tl in captions]


def _column_starts(textlines, x_left, x_right):
    """Infer stable column start x positions inside one horizontal group.

    All row bands vote on column start positions by clustering x0 values.
    Rows with an empty leading cell therefore do not shrink the column
    layout.
    """
    in_group = [
        tl
        for tl in textlines
        if x_left - 2 <= (tl.x0 + tl.x1) / 2 < x_right - 2
    ]
    if not in_group:
        return []
    rows_grouped = _row_bands_from_textlines(in_group)
    widest_count = max(len(band) for band in rows_grouped)

    # Collect every observed x0, then cluster values that are close.
    observed = sorted(tl.x0 for tl in in_group)
    clusters: list[list[float]] = []
    for x in observed:
        if clusters and abs(x - clusters[-1][-1]) <= 8.0:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    starts = [sum(c) / len(c) for c in clusters]
    if len(starts) > widest_count:
        # Keep one representative per distinct column: merge clusters that
        # belong together by preferring the widest rows' x0 values.
        widest_x = sorted(
            {round(tl.x0, 3) for band in rows_grouped if len(band) == widest_count for tl in band}
        )
        starts = widest_x
    return starts


def _rebuild_split_table(
    merged: Any,
    page_lines: list[Any],
    x_left: float,
    x_right: float,
    new_order: int,
    header_floor_y: float,
) -> Table:
    """Rebuild one independent Table from a horizontal slice of a fragment.

    Caption lines are excluded by their text classification rather than a
    hard y cutoff, so a header sitting close to the caption level is kept.
    Columns are rebuilt from x-position clusters inside the group, which
    keeps the correct number of columns even when some leading cells are
    empty.
    """
    in_range = [
        tl
        for tl in page_lines
        if tl.get_text().strip()
        and (tl.y0 + tl.y1) / 2.0 <= header_floor_y + 2
        and x_left - 2 <= (tl.x0 + tl.x1) / 2 < x_right - 2
        and _classify_caption(tl.get_text().strip())[0]
        not in ("table_id", "continued", "section")
    ]
    rows_grouped = [r for r in _row_bands_from_textlines(in_range) if r]
    if not rows_grouped:
        raise RuntimeError("empty split fragment")

    # Row bands from the actual text line extents.
    row_tuples = []
    for band in rows_grouped:
        top = max(tl.y1 for tl in band)
        bottom = min(tl.y0 for tl in band)
        row_tuples.append((bottom, top))
    for i in range(len(row_tuples) - 1):
        top_y = row_tuples[i][1]
        next_y = row_tuples[i + 1][0]
        mid = (top_y + next_y) / 2.0
        row_tuples[i] = (row_tuples[i][0], mid)
        row_tuples[i + 1] = (mid, row_tuples[i + 1][1])

    starts = _column_starts(in_range, x_left, x_right)
    # Calibrate against the widest row band: its distinct x0 values give
    # the authoritative column count, clustering absorbs small drift.
    widest = max(rows_grouped, key=lambda band: len(band))
    widest_starts = sorted({round(tl.x0, 3) for tl in widest})
    if len(widest_starts) > len(starts):
        starts = widest_starts
    all_x0 = [tl.x0 for tl in in_range]
    all_x1 = [tl.x1 for tl in in_range]
    col_left = min(all_x0) - 3
    col_right = max(all_x1) + 3
    boundaries = [col_left]
    boundaries.extend(
        (prev + nxt) / 2.0 for prev, nxt in zip(starts, starts[1:])
    )
    boundaries.append(col_right)
    cols = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    table = Table(cols, row_tuples)
    table.set_all_edges()
    table.page = merged.page
    table.order = new_order
    table.flavor = merged.flavor
    table.filename = merged.filename
    table.pdf_size = merged.pdf_size
    table._bbox = (col_left, row_tuples[-1][0], col_right, row_tuples[0][1])
    table._page_horizontal_text = merged._page_horizontal_text
    table._page_vertical_text = merged._page_vertical_text

    # Assign text into cells using center coordinates; preserve raw text.
    for r_idx, band in enumerate(rows_grouped):
        for tl in sorted(band, key=lambda t: t.x0):
            cx = (tl.x0 + tl.x1) / 2.0
            c_idx = min(
                range(len(cols)),
                key=lambda j: abs(cx - (cols[j][0] + cols[j][1]) / 2.0),
            )
            table.cells[r_idx][c_idx].text = tl.get_text()

    data = table.data
    table.df = pd.DataFrame(data)
    table.shape = table.df.shape
    from .utils import compute_whitespace

    table.whitespace = compute_whitespace(data)
    table.accuracy = merged.accuracy
    table.textlines = list(merged.textlines)
    table._text = list(getattr(merged, "_text", []))
    return table


def _header_row_gap_cuts(
    page_lines: list[Any], header_floor_y: float
) -> list[float]:
    """Find boundaries from the bold header row's largest x gaps.

    Header rows are the most regular row on a page: every column has a
    bold label.  The largest horizontal gaps between consecutive header
    labels therefore mark the boundaries between side-by-side tables.
    """
    # Header lines are bold and sit strictly below the caption floor.
    # Captions are bold too but one guard band above the table body.
    header_lines_all = [
        tl
        for tl in page_lines
        if tl.get_text().strip()
        and _line_bold(tl)
        and (tl.y0 + tl.y1) / 2.0 < header_floor_y - ROW_GUARD / 2.0
    ]
    if not header_lines_all:
        return []
    levels: dict[float, list[Any]] = {}
    for tl in header_lines_all:
        key = round(tl.y0, 1)
        for existing in levels:
            if math.isclose(existing, key, abs_tol=4.0):
                levels[existing].append(tl)
                break
        else:
            levels[key] = [tl]
    top_y = max(levels, key=lambda k: len(levels[k]))
    header_row = sorted(levels[top_y], key=lambda tl: tl.x0)
    if len(header_row) < 4:
        return []
    # Side-by-side tables each have their own regular column pitch.  Scan
    # every possible split of the header labels and score how well each
    # side keeps a constant inter-column distance: the true boundary is the
    # split where both sides have the lowest pitch variance.
    x0s = [tl.x0 for tl in header_row]
    if len(x0s) < 4:
        return []

    def pitch_variance(values):
        if len(values) < 2:
            return float("inf")
        diffs = [b - a for a, b in zip(values, values[1:])]
        mean = sum(diffs) / len(diffs)
        return sum((d - mean) ** 2 for d in diffs) / len(diffs)

    best_cut = None
    best_score = float("inf")
    for k in range(2, len(x0s) - 1):
        # The two groups are disjoint (x0s[:k] and x0s[k:]); each needs at
        # least two columns to have a measurable column pitch.
        left_pitch = pitch_variance(x0s[:k])
        right_pitch = pitch_variance(x0s[k:])
        score = left_pitch + right_pitch
        if score < best_score:
            best_score = score
            # Boundary must clear the last left label's right edge, while
            # staying before the first right label's left edge.  When the
            # two labels nearly touch, split the inter-label space.
            last_left = header_row[k - 1]
            first_right = header_row[k]
            best_cut = max(last_left.x1, (last_left.x1 + first_right.x0) / 2.0)
    if best_cut is None:
        return []
    return [best_cut]


def _gap_near(
    page_lines: list[Any],
    header_floor_y: float,
    anchor: float,
    search_half_width: float,
) -> float | None:
    """Find the x position of an empty column gap near ``anchor``.

    The search works per row band: candidate cuts are positions where none
    of the row bands contain text.  This avoids merging adjacent columns of
    two independent tables into a single horizontal interval.
    """
    body = [
        tl
        for tl in page_lines
        if tl.get_text().strip()
        and (tl.y0 + tl.y1) / 2.0 <= header_floor_y + 2
        and anchor - search_half_width <= (tl.x0 + tl.x1) / 2
        <= anchor + search_half_width
    ]
    bands = [b for b in _row_bands_from_textlines(body) if b]
    if not bands:
        return None

    # Candidate cuts: every midpoint between distinct text x0/x1 edges.
    edges = sorted({x for tl in body for x in (tl.x0, tl.x1)})
    candidates = [
        (a + b) / 2.0 for a, b in zip(edges, edges[1:])
    ]
    scored = []
    for cut in candidates:
        if abs(cut - anchor) > search_half_width:
            continue
        # A valid cut has no text line straddling it on any row band.
        straddled = any(
            tl.x0 < cut < tl.x1 for band in bands for tl in band
        )
        if straddled:
            continue
        nearest_left = [
            tl.x1 for band in bands for tl in band if tl.x1 <= cut
        ]
        nearest_right = [
            tl.x0 for band in bands for tl in band if tl.x0 >= cut
        ]
        if not nearest_left or not nearest_right:
            continue
        gap = min(nearest_right) - max(nearest_left)
        # Balance: the boundary between two tables leaves columns on both
        # sides, whereas an interior column gap starves one side.
        left_cols = len(
            {
                round(tl.x0, 1)
                for band in bands
                for tl in band
                if (tl.x0 + tl.x1) / 2 < cut
            }
        )
        right_cols = len(
            {
                round(tl.x0, 1)
                for band in bands
                for tl in band
                if (tl.x0 + tl.x1) / 2 >= cut
            }
        )
        balance = min(left_cols, right_cols)
        scored.append((balance, gap, -abs(cut - anchor), cut))
    if not scored:
        return None
    # Most balanced partition first, then widest gap, then anchor proximity.
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return scored[0][3]


def _derive_pieces(
    page_tables: list[Any],
    page_lines: list[Any],
    union_bbox: tuple[float, float, float, float],
    merged: Any,
) -> list[tuple[Any, list[list[str]], FragmentContext]] | None:
    """Derive independent fragments from a page's physical tables.

    Returns None when the physical tables already represent independent
    fragments.  Otherwise returns one ``(table, matrix, context)`` tuple per
    derived side table.  The physical tables are never mutated: rebuilt
    pieces are constructed as separate objects and only used by the logical
    layer.
    """
    caption_centers = _caption_centers(merged, page_lines)
    groups: list[tuple[float, float]] = []
    if len(caption_centers) >= 2:
        all_caption_ys = [
            (tl.y0 + tl.y1) / 2.0
            for tl in page_lines
            if tl.get_text().strip()
            and _classify_caption(tl.get_text().strip())[0]
            in ("table_id", "continued")
        ]
        caption_floor = max(all_caption_ys) if all_caption_ys else union_bbox[3]
        anchor_cuts = []
        header_cuts = _header_row_gap_cuts(
            page_lines, caption_floor - ROW_GUARD
        )
        n_boundaries = len(caption_centers) - 1
        if len(header_cuts) == n_boundaries:
            anchor_cuts = list(header_cuts)
        else:
            for left_center, right_center in zip(
                caption_centers, caption_centers[1:]
            ):
                anchor = (left_center + right_center) / 2.0
                cut = None
                for hc in header_cuts:
                    if hc >= left_center and hc <= right_center + CAPTION_X_SLACK:
                        cut = hc
                        break
                if cut is None and header_cuts:
                    cut = min(header_cuts, key=lambda hc: abs(hc - anchor))
                if cut is None:
                    half = max(160.0, (right_center - left_center))
                    cut = _gap_near(
                        page_lines, caption_floor - ROW_GUARD, anchor, half
                    )
                if cut is None:
                    cut = anchor
                anchor_cuts.append(cut)
        body_extent = [
            tl
            for tl in page_lines
            if tl.get_text().strip()
            and (tl.y0 + tl.y1) / 2.0 <= caption_floor - ROW_GUARD / 2.0
        ]
        page_x0 = min(tl.x0 for tl in body_extent) - 6
        page_x1 = max(tl.x1 for tl in body_extent) + 6
        cuts = [page_x0] + anchor_cuts + [page_x1]
        groups = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    elif merged.fragment_context.n_cols >= 4:
        bold_ys = [
            (tl.y0 + tl.y1) / 2.0
            for tl in page_lines
            if tl.get_text().strip() and _line_bold(tl)
        ]
        if bold_ys:
            header_anchor = max(bold_ys) + ROW_HEIGHT
        else:
            header_anchor = union_bbox[3] + ROW_HEIGHT
        header_cuts = _header_row_gap_cuts(page_lines, header_anchor)
        if header_cuts:
            body_extent = [tl for tl in page_lines if tl.get_text().strip()]
            page_x0 = min(tl.x0 for tl in body_extent) - 6
            page_x1 = max(tl.x1 for tl in body_extent) + 6
            cuts = [page_x0] + header_cuts + [page_x1]
            groups = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
        else:
            gap_points = _split_points_from_gaps(merged, page_lines)
            if gap_points:
                cuts = [union_bbox[0]] + gap_points + [union_bbox[2]]
                groups = [
                    (cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)
                ]

    if len(groups) < 2:
        return None

    all_caption_ys = [
        (tl.y0 + tl.y1) / 2.0
        for tl in page_lines
        if tl.get_text().strip()
        and _classify_caption(tl.get_text().strip())[0]
        in ("table_id", "continued")
    ]
    caption_floor = max(all_caption_ys) if all_caption_ys else union_bbox[3]
    bold_below_caption = [
        tl
        for tl in page_lines
        if tl.get_text().strip()
        and _line_bold(tl)
        and (tl.y0 + tl.y1) / 2.0 < caption_floor - ROW_GUARD / 2.0
    ]
    if bold_below_caption:
        header_floor_y = max(tl.y1 for tl in bold_below_caption) + 1.0
    else:
        header_floor_y = caption_floor - ROW_GUARD

    backing = page_tables[0]
    pieces = []
    for i, (x_left, x_right) in enumerate(groups):
        try:
            piece = _rebuild_split_table(
                backing,
                page_lines,
                x_left,
                x_right,
                i + 1,
                header_floor_y,
            )
        except RuntimeError:
            continue
        # The rebuilt piece keeps the original physical table's page/order
        # so provenance still points at the real extraction, but carries its
        # own cells built from original page geometry.
        piece.page = backing.page
        piece.order = backing.order
        horizontal = list(backing._page_horizontal_text or [])
        vertical = list(backing._page_vertical_text or [])
        ctx = build_fragment_context(piece, horizontal, vertical)
        pieces.append((piece, piece.data, ctx))
    if len(pieces) < 2:
        return None
    return pieces


# ---------------------------------------------------------------------------
# Stage 2 -- adjacency decisions
# ---------------------------------------------------------------------------


@dataclass
class EdgeEvidence:
    """Evaluated relationship between a predecessor and a candidate fragment."""

    predecessor: tuple[int, int]
    candidate: tuple[int, int]
    strength: str  # "strong" | "medium" | "weak" | "none"
    decision: str  # "adopted" | "rejected"
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON friendly representation."""
        return {
            "predecessor": list(self.predecessor),
            "candidate": list(self.candidate),
            "strength": self.strength,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "details": self.details,
        }


def _affine_column_map(
    prev: FragmentContext, cur: FragmentContext
) -> tuple[float, float, float, list[tuple[int, int, float]]]:
    """Map ``cur`` column centers onto ``prev`` coordinates.

    Returns ``(scale, delta, match_fraction, pairs)`` where each pair is
    ``(prev_col_index, cur_col_index, distance)``.
    """
    p_x0, _p_y0, p_x1, _p_y1 = prev.bbox
    c_x0, _c_y0, c_x1, _c_y1 = cur.bbox
    p_width = max(p_x1 - p_x0, 1e-6)
    c_width = max(c_x1 - c_x0, 1e-6)
    scale = c_width / p_width
    delta = c_x0 - scale * p_x0

    def to_cur(x: float) -> float:
        return scale * x + delta

    pairs: list[tuple[int, int, float]] = []
    used = set()
    for i, center in enumerate(prev.col_centers):
        target = to_cur(center)
        best_j, best_d = None, None
        for j, cur_center in enumerate(cur.col_centers):
            if j in used:
                continue
            d = abs(cur_center - target)
            if best_d is None or d < best_d:
                best_j, best_d = j, d
        if best_j is not None:
            tol = max(
                COL_MATCH_TOL_PTS,
                COL_MATCH_TOL_FRAC * (p_x1 - p_x0) / max(prev.n_cols, 1),
            )
            if best_d <= tol:
                used.add(best_j)
                pairs.append((i, best_j, round(best_d, 3)))
    match_fraction = len(pairs) / max(prev.n_cols, 1)
    return scale, delta, match_fraction, pairs


def _headers_compatible(prev: FragmentContext, cur: FragmentContext) -> bool:
    """Compare multi-row headers as normalised token multisets."""
    if not prev.header_normalized or not cur.header_normalized:
        return False
    return sorted(prev.header_normalized) == sorted(cur.header_normalized)


def evaluate_edge(
    prev: FragmentContext,
    cur: FragmentContext,
    page_gap: int,
    prev_fragment_key: tuple | None = None,
    cur_fragment_key: tuple | None = None,
) -> EdgeEvidence:
    """Evaluate whether ``cur`` is a continuation fragment of ``prev``.

    Hard blockers (distinct table ids, chapter/section switch, missing
    intermediate pages, incompatible geometry) always win over positive
    signals.  Explicit captions and continuation wording produce strong
    edges; header plus geometry agreement without captions produces a
    medium edge; geometry alone is never enough.
    """
    prev_key = prev_fragment_key if prev_fragment_key is not None else prev.key
    cur_key = cur_fragment_key if cur_fragment_key is not None else cur.key
    reasons: list[str] = []
    details: dict[str, Any] = {}
    strength = "none"

    if page_gap > 1:
        reasons.append(
            f"missing intermediate pages: page {prev.page} and page "
            f"{cur.page} are not adjacent in the selected page list"
        )
        return EdgeEvidence(prev_key, cur_key, "none", "rejected", reasons, details)

    # Same-page fragments are independent (side-by-side / stacked tables).
    if prev.page == cur.page:
        reasons.append("fragments on the same page are never merged")
        return EdgeEvidence(prev_key, cur_key, "none", "rejected", reasons, details)

    # Explicit chapter / section switch between the fragments.
    if cur.section_heading is not None:
        reasons.append(
            f"section heading before candidate: {cur.section_heading.text!r}"
        )
        return EdgeEvidence(prev_key, cur_key, "none", "rejected", reasons, details)

    # Distinct explicit table ids always block the merge.
    if prev.table_id and cur.table_id and prev.table_id != cur.table_id:
        reasons.append(
            f"distinct table numbers: {prev.table_id!r} vs {cur.table_id!r}"
        )
        return EdgeEvidence(prev_key, cur_key, "none", "rejected", reasons, details)

    scale, delta, match_fraction, pairs = _affine_column_map(prev, cur)
    details["column_match_fraction"] = round(match_fraction, 3)
    details["column_pairs"] = pairs
    details["scale"] = round(scale, 4)
    details["translation_x"] = round(delta, 3)
    geometry_ok = match_fraction >= 0.8 and (
        cur.n_cols == prev.n_cols or cur.n_cols >= prev.n_cols - 1
    )
    if not geometry_ok:
        reasons.append(
            "column geometry mismatch: "
            f"matched {len(pairs)}/{prev.n_cols} columns, "
            f"{prev.n_cols} vs {cur.n_cols} physical columns"
        )

    headers_ok = _headers_compatible(prev, cur)
    details["headers_compatible"] = headers_ok
    if not headers_ok:
        reasons.append("multi-row header text does not match")

    same_id = bool(prev.table_id) and prev.table_id == cur.table_id
    details["same_table_id"] = same_id
    details["candidate_continuation_marker"] = cur.continuation_marker

    # An explicit table id agreement is decisive on its own: a continuation
    # page may have lost a whole column (empty column / interrupted ruling
    # line), so column geometry must not be allowed to veto the merge.  The
    # canonical column layout is restored later from the first fragment.
    if same_id:
        strength = "strong"
        reasons.append(f"same explicit table number {prev.table_id!r}")
        if cur.continuation_marker:
            reasons.append("continuation caption present on candidate")
        if geometry_ok:
            reasons.append("column geometry agrees")
        else:
            reasons.append(
                "column count reduced on continuation page; canonical "
                "layout to be restored from the first fragment"
            )
    elif geometry_ok and cur.continuation_marker and headers_ok:
        strength = "strong"
        reasons.append("continuation caption present on candidate")
        reasons.append("header text and column geometry agree")
    elif geometry_ok and headers_ok:
        strength = "medium"
        reasons.append("header text and column geometry agree without conflict")
    else:
        strength = "weak" if geometry_ok or headers_ok else "none"
        reasons.append("insufficient positive evidence for a continuation")

    return EdgeEvidence(prev_key, cur_key, strength, "rejected", reasons, details)


_STRENGTH_ORDER = {"strong": 3, "medium": 2, "weak": 1, "none": 0}


@dataclass
class LogicalTable:
    """A logical table assembled from one or more physical fragments.

    Attributes
    ----------
    logical_id : str
        Stable identifier ``"logical-table-N"`` in page/order order.
    df : pandas.DataFrame
        Assembled data matrix.  Columns are integer indexed; the first
        fragment header rows are kept as ordinary data rows while repeated
        headers on continuation fragments are removed.
    fragments : list of tuple
        Fragment keys ``(page, order, sub)`` in page order.  ``sub`` is 0
        for a whole physical table and 1, 2, ... for side-table pieces
        derived from one physical fragment.  Each fragment belongs to
        exactly one logical table, and every physical table backs at least
        one fragment.  Use :attr:`row_sources` for the physical table
        ``(page, order)`` of each data row.
    row_sources : list of RowSource
        Provenance of every logical data row, in logical row order.
    removed_headers : list of dict
        Audit records of repeated header rows removed from continuation
        fragments.
    diagnostics : list of dict
        Diagnostics recorded while building this logical table (ambiguity,
        blocked edges, column layout repair, ...).
    columns : list of dict
        Canonical column layout: boundary coordinates in the first
        fragment page coordinate system.
    """

    logical_id: str
    df: pd.DataFrame
    fragments: list[tuple[int, int, int]]
    row_sources: list["RowSource"]
    removed_headers: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    columns: list[dict[str, Any]]

    def __repr__(self) -> str:  # noqa D105
        return (
            f"<LogicalTable {self.logical_id} shape={self.df.shape} "
            f"fragments={len(self.fragments)}>"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON friendly summary (without the full data frame)."""
        return {
            "logical_id": self.logical_id,
            "shape": list(self.df.shape),
            "fragments": [list(k) for k in self.fragments],
            "removed_headers": list(self.removed_headers),
            "diagnostics": list(self.diagnostics),
            "columns": list(self.columns),
        }

    @property
    def structure(self):
        """Always raise: logical tables have no single-page grid evidence.

        The structural cell view is defined for one physical lattice table
        drawn on one page.  A logical table may join fragments from several
        pages with different column layouts, so no single grid (and no single
        page/order pair) describes it.  Inspect the physical fragments
        instead: ``read_pdf(..., continuation='auto')`` still returns every
        physical table, each of which exposes ``Table.structure``.
        """
        from .structure import StructureUnavailableError

        raise StructureUnavailableError(
            "Logical tables have no structural cell view: it is only available"
            " for single-page physical lattice tables. Inspect the physical"
            " fragments returned by read_pdf() instead."
        )


@dataclass
class RowSource:
    """Provenance for one logical data row.

    Attributes
    ----------
    logical_row : int
        Row index inside :attr:`LogicalTable.df`.
    page : int
        Original PDF page number (1-based).
    order : int
        Physical table order on the page (1-based).
    physical_row : int
        Row index inside the physical fragment's ``df``/``cells``.
    cells : list of dict
        Non-empty cells only, each with ``column`` (logical column index),
        ``text`` (original retained value) and ``bbox`` in the page's PDF
        coordinates.
    """

    logical_row: int
    page: int
    order: int
    physical_row: int
    cells: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """JSON friendly representation."""
        return {
            "logical_row": self.logical_row,
            "page": self.page,
            "order": self.order,
            "physical_row": self.physical_row,
            "cells": [
                {
                    "column": c["column"],
                    "text": c["text"],
                    "bbox": list(c["bbox"]),
                }
                for c in self.cells
            ],
        }


@dataclass
class LogicalFragment:
    """A unit of logical grouping, derived without mutating physical tables.

    Normally one physical :class:`~camelot.core.Table` produces one
    ``LogicalFragment``.  When a single physical fragment actually contains
    two independent side-by-side tables, the continuation layer derives two
    fragments from the original page text instead; both still point back to
    the same physical table via :attr:`source_page`/:attr:`source_order`, so
    every physical table remains attributable and the returned
    :class:`~camelot.core.TableList` is never modified.

    Attributes
    ----------
    key : tuple
        Fragment identity ``(page, order, sub)`` where ``sub`` is 0 for a
        whole physical table and 1, 2, ... for derived side-table pieces.
    source_page, source_order : int
        Physical table (``table.page``/``table.order``) backing this
        fragment.
    context : FragmentContext
        Cross-page evidence used for grouping decisions.
    table : camelot.core.Table
        The backing physical table (its own ``df``/``cells`` are never
        rewritten).  For derived pieces it is still the original table;
        :attr:`derived` is True and :attr:`derived_matrix` holds the
        rebuilt content.
    derived : bool
        Whether this fragment was rebuilt from page text rather than read
        directly from the physical table.
    derived_matrix : list or None
        Rebuilt cell matrix for derived fragments, else None.
    """

    key: tuple[int, int, int]
    source_page: int
    source_order: int
    context: FragmentContext
    table: Any
    derived: bool = False
    derived_matrix: list[list[str]] | None = None

    @property
    def page(self) -> int:
        """PDF page number of the backing physical table."""
        return self.source_page

    @property
    def order(self) -> int:
        """Order of the backing physical table on the page."""
        return self.source_order


def build_logical_fragments(
    tables: list[Any],
) -> list[LogicalFragment]:
    """Build the logical fragment layer over untouched physical tables.

    One fragment is created per physical table, except when page-level
    evidence proves that a physical fragment merges independent
    side-by-side tables; in that case several derived fragments are created
    for the same physical table.  The physical tables themselves are never
    modified.
    """
    fragments: list[LogicalFragment] = []
    pages: dict[int, list[Any]] = {}
    for table in tables:
        pages.setdefault(table.page, []).append(table)

    for page in sorted(pages):
        page_tables = sorted(pages[page], key=lambda t: t.order)
        page_lines = list(page_tables[0]._page_horizontal_text or [])
        union_bbox = (
            min(t._bbox[0] for t in page_tables),
            min(t._bbox[1] for t in page_tables),
            max(t._bbox[2] for t in page_tables),
            max(t._bbox[3] for t in page_tables),
        )

        class _Union:
            def __init__(self, bbox, n_cols):
                self._bbox = bbox
                self.fragment_context = type(
                    "Ctx", (), {"n_cols": n_cols}
                )()

        merged = _Union(
            union_bbox,
            max(t.fragment_context.n_cols for t in page_tables),
        )
        derived = _derive_pieces(page_tables, page_lines, union_bbox, merged)

        if derived is None:
            for table in page_tables:
                fragments.append(
                    LogicalFragment(
                        key=(table.page, table.order, 0),
                        source_page=table.page,
                        source_order=table.order,
                        context=table.fragment_context,
                        table=table,
                    )
                )
        else:
            backing = page_tables[0]
            for sub, (piece, matrix, ctx) in enumerate(derived, start=1):
                fragments.append(
                    LogicalFragment(
                        key=(backing.page, backing.order, sub),
                        source_page=backing.page,
                        source_order=backing.order,
                        context=ctx,
                        table=piece,
                        derived=True,
                        derived_matrix=matrix,
                    )
                )
    return fragments


def build_logical_tables(
    tables: list[Any],
) -> tuple[list[LogicalTable], list[dict[str, Any]]]:
    """Group physical ``tables`` into logical tables.

    The physical tables are only read.  A derived fragment layer
    (:func:`build_logical_fragments`) handles side-by-side tables that a
    single physical fragment merged, so the returned :class:`TableList`
    stays identical to the default extraction while every physical table is
    still represented by at least one fragment.

    The function only inspects boundaries between adjacent selected pages.
    Each fragment keeps at most one predecessor and one successor; competing
    edges of equal strength produce an ambiguity diagnostic instead of an
    arbitrary choice.  Fragments that cannot be merged are still returned,
    each as its own single-fragment logical table, so no physical table is
    ever lost.

    Returns
    -------
    logical_tables : list of LogicalTable
    global_diagnostics : list of dict
        Boundary-level diagnostics (ambiguities and explicit block
        decisions), deterministic and JSON friendly.
    """
    fragments = build_logical_fragments(tables)
    contexts: dict[tuple[int, int, int], FragmentContext] = {
        f.key: f.context for f in fragments
    }
    ordered_keys = sorted(contexts.keys())
    pages: dict[int, list[tuple[int, int, int]]] = {}
    for key in ordered_keys:
        pages.setdefault(key[0], []).append(key)
    sorted_pages = sorted(pages)

    # predecessor key -> successor key, and the reverse map
    successor: dict[tuple, tuple] = {}
    predecessor: dict[tuple, tuple] = {}
    global_diagnostics: list[dict[str, Any]] = []

    for page_idx in range(1, len(sorted_pages)):
        prev_page, cur_page = sorted_pages[page_idx - 1], sorted_pages[page_idx]
        page_gap = cur_page - prev_page
        for cur_key in pages[cur_page]:
            cur = contexts[cur_key]
            edges = [
                evaluate_edge(
                    contexts[pk],
                    cur,
                    page_gap,
                    prev_fragment_key=pk,
                    cur_fragment_key=cur_key,
                )
                for pk in pages[prev_page]
            ]
            mergeable = [e for e in edges if e.strength in ("strong", "medium")]
            blockers = [
                e
                for e in edges
                if e.reasons
                and e.strength in ("none", "weak")
                and any(
                    "table number" in r or "section heading" in r or "missing" in r
                    for r in e.reasons
                )
            ]

            if not mergeable:
                for edge in blockers:
                    global_diagnostics.append(
                        {
                            "type": "blocked_edge",
                            "candidate": list(cur_key),
                            "edge": edge.to_dict(),
                        }
                    )
                continue

            strengths = {e.strength for e in mergeable}
            top_strength = max(strengths, key=lambda s: _STRENGTH_ORDER[s])
            top = [e for e in mergeable if e.strength == top_strength]
            if len(top) > 1:
                global_diagnostics.append(
                    {
                        "type": "ambiguity",
                        "candidate": list(cur_key),
                        "reason": (
                            f"{len(top)} competing {top_strength} predecessor "
                            "fragments; no merge performed"
                        ),
                        "candidates": [e.to_dict() for e in mergeable],
                    }
                )
                continue

            chosen = top[0]
            chosen.decision = "adopted"
            rejected = [e for e in mergeable if e is not chosen]
            pred_fragment_key = chosen.predecessor
            for edge in rejected:
                edge.decision = "rejected"
                edge.reasons.append(
                    f"losing against stronger/unique predecessor "
                    f"{list(chosen.predecessor)}"
                )
            successor[pred_fragment_key] = cur_key
            predecessor[cur_key] = pred_fragment_key
            if rejected:
                global_diagnostics.append(
                    {
                        "type": "resolved_competition",
                        "candidate": list(cur_key),
                        "adopted": chosen.to_dict(),
                        "rejected": [e.to_dict() for e in rejected],
                    }
                )

    # Build chains in the original fragment ordering.
    groups: list[list[tuple]] = []
    group_of: dict[tuple, int] = {}
    for key in ordered_keys:
        if key in predecessor:
            gi = group_of[predecessor[key]]
            groups[gi].append(key)
        else:
            gi = len(groups)
            groups.append([key])
        group_of[key] = gi

    fragments_by_key = {f.key: f for f in fragments}
    logical_tables: list[LogicalTable] = []
    for idx, group_keys in enumerate(groups):
        group_fragments = [fragments_by_key[k] for k in group_keys]
        lt, local_diags = reassemble_logical_table(
            idx + 1, group_fragments, global_diagnostics
        )
        logical_tables.append(lt)
        global_diagnostics.extend(local_diags)

    return logical_tables, global_diagnostics


# ---------------------------------------------------------------------------
# Stage 3 -- canonical column reassembly
# ---------------------------------------------------------------------------


def _canonical_affine(
    first: FragmentContext, cur: FragmentContext
) -> tuple[float, float]:
    """Affine map from first-fragment x coordinates to ``cur`` page coords."""
    f_x0, _f_y0, f_x1, _f_y1 = first.bbox
    c_x0, _c_y0, c_x1, _c_y1 = cur.bbox
    f_width = max(f_x1 - f_x0, 1e-6)
    scale = (c_x1 - c_x0) / f_width
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    delta = c_x0 - scale * f_x0
    return scale, delta


def _reassemble_fragment_matrix(
    table: Any,
    first_ctx: FragmentContext,
    cur_ctx: FragmentContext,
) -> tuple[list[list[str]], list[list[list[dict[str, Any]]]], bool]:
    """Rebuild one fragment against the canonical column layout.

    Returns the raw text matrix, per-row/per-cell contributing text line
    geometry (for provenance), and whether the physical column layout had to
    be repaired (missing column or interrupted line).
    """
    scale, delta = _canonical_affine(first_ctx, cur_ctx)
    canonical_bounds = [
        scale * x + delta for x in first_ctx.col_anchors
    ]
    n_cols = first_ctx.n_cols

    repaired = cur_ctx.n_cols != n_cols
    if not repaired:
        # Even with equal column counts, a lattice fragment may have lost an
        # interior vertical segment: detect it through the measured anchors.
        mapped = [scale * x + delta for x in first_ctx.col_anchors]
        for anchor in mapped[1:-1]:
            if all(abs(anchor - a) > COL_MATCH_TOL_PTS for a in cur_ctx.col_anchors):
                repaired = True
                break

    horizontal = list(getattr(table, "_page_horizontal_text", []) or [])
    vertical = list(getattr(table, "_page_vertical_text", []) or [])
    bbox = cur_ctx.bbox

    # Caption lines are excluded by their text classification, not by a y
    # window: a split table's header can sit close to the caption level and
    # must be preserved.
    caption_lines = {
        id(tl)
        for tl in horizontal
        if tl.get_text().strip()
        and _classify_caption(tl.get_text().strip())[0]
        in ("table_id", "continued")
    }

    def in_bbox(tl: Any) -> bool:
        cx = (tl.x0 + tl.x1) / 2.0
        cy = (tl.y0 + tl.y1) / 2.0
        # Generous vertical window: a rebuilt fragment bbox can clip the
        # top header band.  Horizontal containment stays strict.
        return (
            bbox[0] - 2 <= cx <= bbox[2] + 2
            and bbox[1] - ROW_HEIGHT <= cy <= bbox[3] + ROW_HEIGHT
        )

    textlines = [
        tl
        for tl in horizontal
        if in_bbox(tl) and id(tl) not in caption_lines
    ]
    textlines.extend(
        tl
        for tl in vertical
        if in_bbox(tl) and id(tl) not in caption_lines
    )

    # Rebuild row bands from the actual text lines instead of reusing the
    # physical parser's row_bands: an empty continuation column can leave
    # physical bands with no text, which would otherwise produce blank
    # matrix rows and misalign row provenance.
    actual_bands = _row_bands_from_textlines(textlines)

    matrix: list[list[str]] = []
    geometry: list[list[list[dict[str, Any]]]] = []
    for band_lines in actual_bands:
        row_values = ["" for _ in range(n_cols)]
        row_geo: list[list[dict[str, Any]]] = [[] for _ in range(n_cols)]
        band_lines = [tl for tl in band_lines if tl.get_text()]
        band_lines.sort(key=lambda tl: (tl.x0, tl.y0))
        for tl in band_lines:
            cx = (tl.x0 + tl.x1) / 2.0
            col_idx = None
            for j in range(n_cols):
                if canonical_bounds[j] <= cx <= canonical_bounds[j + 1]:
                    col_idx = j
                    break
            if col_idx is None:
                # Fall back to the nearest canonical column instead of
                # dropping the text.
                col_idx = min(
                    range(n_cols),
                    key=lambda j: abs(
                        cx - (canonical_bounds[j] + canonical_bounds[j + 1]) / 2.0
                    ),
                )
            raw = tl.get_text()
            # Preserve the original text; only join fragments inside a cell.
            row_values[col_idx] = "".join([row_values[col_idx], raw])
            row_geo[col_idx].append(
                {
                    "bbox": (
                        float(tl.x0),
                        float(tl.y0),
                        float(tl.x1),
                        float(tl.y1),
                    ),
                    "text": raw,
                }
            )
        matrix.append(row_values)
        geometry.append(row_geo)
    return matrix, geometry, repaired


def _union_bbox(boxes: list[tuple[float, float, float, float]]):
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _row_is_caption(
    row: list[str], ctx: FragmentContext, table: Any
) -> bool:
    """Return True when a physical row is an outside caption.

    Stream fragments can include the table-number caption as a leading
    physical row.  It is recognised through the FragmentContext caption
    evidence rather than cell content, so body text equal to a header word
    is never affected.
    """
    if ctx.caption is None:
        return False
    joined = normalize_text(" ".join(v for v in row if v))
    if not joined:
        return False
    caption_text = normalize_text(ctx.caption.text)
    if joined == caption_text or joined in caption_text or caption_text in joined:
        # Only treat as caption when the row sits at the very top of the
        # fragment (captions never appear in the middle of a table).
        first_non_empty = next(
            (
                i
                for i, r in enumerate(table.data)
                if any(v.strip() for v in r)
            ),
            0,
        )
        try:
            row_index = table.data.index(list(row))
        except ValueError:
            row_index = 0
        return row_index <= first_non_empty
    return False


def _is_repeated_header(
    first_ctx: FragmentContext,
    matrix: list[list[str]],
) -> int:
    """Return the number of leading rows that repeat the first header.

    Handles wrapped continuation headers: a logical header row may occupy
    several physical rows because a label wraps onto a second visual line.
    The leading physical rows are consumed while their normalised tokens
    are still explainable by the first fragment's header token set.  A body
    row that merely equals one header word never qualifies on its own, and
    empty leading cells are allowed.
    """
    depth = first_ctx.header_depth
    if depth == 0 or not matrix:
        return 0
    header_tokens = set(first_ctx.header_normalized)
    if not header_tokens:
        return 0

    def column_texts(row):
        return [normalize_text(c) for c in row]

    # Reference header as one normalised string per logical header row and
    # column.  Wrapped continuation labels are prefixes/subsets of these.
    reference = [
        [normalize_text(c) for c in row]
        for row in first_ctx.header_rows
    ]

    def tokens_of(text):
        return set(text.split()) if text else set()

    consumed = 0
    logical_row = 0
    limit = min(len(matrix), depth * 2)
    for i in range(limit):
        if logical_row >= depth:
            break
        cand = column_texts(matrix[i])
        ref = reference[logical_row]
        if not any(cand):
            consumed += 1
            continue
        # Every non-empty candidate cell must be explainable by the current
        # reference header row (wrapped pieces are token subsets); body rows
        # contain data tokens that never satisfy this.
        explainable = True
        for j, text in enumerate(cand):
            if not text:
                continue
            ref_tokens = tokens_of(ref[j]) if j < len(ref) else set()
            if not ref_tokens:
                explainable = False
                break
            if not tokens_of(text) <= ref_tokens:
                explainable = False
                break
        if not explainable:
            break
        consumed += 1
        # A logical header row is complete once all its non-empty reference
        # cells have been seen; wrapped visual lines may precede that.
        seen_cells = {
            j
            for j, text in enumerate(cand)
            if text and j < len(ref) and tokens_of(text) == tokens_of(ref[j])
        }
        required = {j for j, text in enumerate(ref) if text}
        if required <= seen_cells or i + 1 >= limit:
            logical_row += 1

    if logical_row >= depth and consumed >= depth:
        return consumed
    return 0


def _fragment_matrix(fragment: LogicalFragment) -> list[list[str]]:
    """Return the working matrix of a fragment (derived or physical)."""
    if fragment.derived:
        return [list(row) for row in fragment.derived_matrix]
    return [list(row) for row in fragment.table.data]


def reassemble_logical_table(
    index: int,
    group_fragments: list[LogicalFragment],
    global_diagnostics: list[dict[str, Any]],
) -> tuple[LogicalTable, list[dict[str, Any]]]:
    """Assemble the canonical data frame and row provenance for one group."""
    first_fragment = group_fragments[0]
    first_ctx = first_fragment.context
    logical_id = f"logical-table-{index}"

    all_rows: list[list[str]] = []
    row_sources: list[RowSource] = []
    removed_headers: list[dict[str, Any]] = []
    local_diags: list[dict[str, Any]] = []

    canonical_columns = [
        {
            "column": j,
            "x0": first_ctx.col_anchors[j],
            "x1": first_ctx.col_anchors[j + 1],
            "reference_page": first_ctx.page,
        }
        for j in range(first_ctx.n_cols)
    ]

    logical_row = 0
    for fragment in group_fragments:
        ctx = fragment.context
        table = fragment.table
        geometry = None
        if fragment.derived:
            # Rebuilt side-table piece: its matrix already excludes captions.
            matrix = _fragment_matrix(fragment)
        elif fragment is first_fragment:
            # First physical fragment: keep its extraction, drop caption rows
            # that stream may have included inside the bbox.
            matrix = [
                list(row)
                for row in table.data
                if not _row_is_caption(row, ctx, table)
            ]
        else:
            matrix, geometry, repaired = _reassemble_fragment_matrix(
                table, first_ctx, ctx
            )
            if repaired:
                local_diags.append(
                    {
                        "type": "column_layout_repaired",
                        "logical_id": logical_id,
                        "fragment": [ctx.page, ctx.order],
                        "reason": (
                            "continuation fragment missed columns of the "
                            "canonical first-fragment layout; text reassigned "
                            "using original page geometry"
                        ),
                        "canonical_columns": first_ctx.n_cols,
                        "physical_columns": ctx.n_cols,
                    }
                )

        # Repeated headers only occur on continuation fragments; the first
        # fragment's own header is retained as data.
        skip_rows = 0
        if fragment is not first_fragment:
            skip_rows = _is_repeated_header(first_ctx, matrix)
            if skip_rows:
                for h in range(skip_rows):
                    removed_headers.append(
                        {
                            "logical_id": logical_id,
                            "page": ctx.page,
                            "order": ctx.order,
                            "physical_row": h,
                            "values": list(matrix[h]),
                        }
                    )

        for physical_row, values in enumerate(matrix):
            if physical_row < skip_rows:
                continue
            cells: list[dict[str, Any]] = []
            for col_idx, value in enumerate(values):
                if not value:
                    continue
                if geometry is not None:
                    pieces = geometry[physical_row][col_idx]
                    bbox = _union_bbox([p["bbox"] for p in pieces])
                    if bbox is None:
                        continue
                else:
                    # Map the working row back to a physical table row so the
                    # provenance references the original physical extraction.
                    phys_row = _physical_row_index(
                        table, values, physical_row, fragment
                    )
                    if phys_row is None or col_idx >= len(table.cells[phys_row]):
                        continue
                    cell = table.cells[phys_row][col_idx]
                    bbox = (cell.x1, cell.y1, cell.x2, cell.y2)
                cells.append(
                    {
                        "column": col_idx,
                        "text": value,
                        "bbox": bbox,
                    }
                )
            row_sources.append(
                RowSource(
                    logical_row=logical_row,
                    page=ctx.page,
                    order=ctx.order,
                    physical_row=physical_row,
                    cells=cells,
                )
            )
            all_rows.append(list(values))
            logical_row += 1

    width = first_ctx.n_cols
    for row in all_rows:
        if len(row) < width:
            row.extend([""] * (width - len(row)))

    df = pd.DataFrame(all_rows, columns=list(range(width)))
    lt = LogicalTable(
        logical_id=logical_id,
        df=df,
        fragments=[_fragment_ref(f) for f in group_fragments],
        row_sources=row_sources,
        removed_headers=removed_headers,
        diagnostics=local_diags,
        columns=canonical_columns,
    )
    return lt, local_diags


def _fragment_ref(fragment: LogicalFragment) -> tuple[int, int, int]:
    """Return the full fragment key ``(page, order, sub)``."""
    return fragment.key


def _physical_row_index(
    table: Any,
    values: list[str],
    working_row: int,
    fragment: LogicalFragment,
) -> int | None:
    """Map a working matrix row back to a physical table row index.

    For derived fragments the rebuilt rows follow the same top-to-bottom
    order as the physical rows that contribute text, so the working index is
    used directly; for ordinary physical fragments the matrix index already
    is the physical index.
    """
    if working_row < len(table.cells):
        return working_row
    return None
