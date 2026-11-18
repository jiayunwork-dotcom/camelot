"""Structural (merged cell) view of single-page lattice tables.

The lattice parser runs two independent stages:

1. line / joint detection decides the grid boundaries and the per-cell edge
   flags (:meth:`camelot.core.Table.set_edges` and
   :meth:`~camelot.core.Table.set_border`);
2. text assignment (:func:`camelot.utils.get_table_index`) decides which grid
   cell owns each extracted text line, before the optional ``shift_text`` /
   ``copy_text`` adjustments.

Neither stage exposes merged cells: after extraction only a flat matrix is
left, so an empty slot cannot be distinguished from a slot covered by a
title that spans several rows and/or columns.

This module derives that missing description from the *same* evidence, without
changing the flat matrix, ``Table.cells``, the parsing report or the default
exports.

How seams are classified
------------------------
Every interior grid seam (between two horizontally or vertically adjacent
grid cells) is measured against the line segments actually found inside the
table bounding box.  Let ``coverage`` be the fraction of the seam length
covered by the union of detected line segments aligned with that seam (both
segment orientation and the seam coordinate are matched within
``joint_tol``):

* ``coverage <= OPEN_SEAM_COVERAGE`` and both adjacent lattice edge flags are
  open (no boundary was recorded for either cell) -> **open** seam;
* ``coverage >= SOLID_SEAM_COVERAGE`` and at least one adjacent cell recorded
  the boundary -> **solid** seam, and a drawn separator is never crossed by a
  region;
* a partly covered seam, or a seam whose coverage and flags disagree (no
  pixels where the lattice recorded a boundary, or a mostly covered seam the
  lattice did not record), is **ambiguous** / **conflicting** and reported.
  These seams never join cells, so no separator is silently swallowed.

Only open seams join grid cells.  Ambiguous and conflicting seams never join
cells and are reported as :class:`StructureDiagnostic` records (page, table
order, grid positions and retained texts).  Regions are the connected
components of grid cells under open seams; a component is accepted only when
it exactly fills its bounding rectangle.  Ragged components (an L shape, for
example) are reported with every member position and are kept as separate
cells, so no rectangular interpretation is silently chosen.

Cost
----
The work is linear in the grid: ``O(R * C)`` seams and a union-find over
``R * C`` cells.  Line segments are indexed by coordinate once
(``O(S log S)``) and only segments near a given seam are inspected.  No
enumeration of candidate rectangles (or of cell subsets) is performed, so the
cost grows with the number of rows/columns, not with combinations of them.
"""

from __future__ import annotations

import bisect
import html as _html
from dataclasses import dataclass
from dataclasses import field
from typing import Any


# A seam with at most this fraction covered by a line is considered open.
OPEN_SEAM_COVERAGE = 0.15
# A seam with at least this fraction covered by a line is a drawn separator.
SOLID_SEAM_COVERAGE = 0.5
# Small tolerance used when clipping line segments to a seam.
SEAM_COORD_TOLERANCE = 2.0


# ---------------------------------------------------------------------------
# Evidence collection / seam classification
# ---------------------------------------------------------------------------


class StructureUnavailableError(NotImplementedError):
    """Raised when the structural view cannot be derived for a table."""


def _union_length(intervals: list[tuple[float, float]]) -> float:
    """Length of the union of one-dimensional intervals."""
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo <= cur_hi + 1e-6:
            cur_hi = max(cur_hi, hi)
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    total += cur_hi - cur_lo
    return total


@dataclass
class TextSource:
    """One original text fragment that belongs to a structural cell.

    Attributes
    ----------
    row, col : int
        Pre-``shift_text`` grid position the text assignment stage gave the
        fragment.
    text : str
        Text as extracted (before ``shift_text`` / ``copy_text``).
    bbox : tuple
        ``(x0, y0, x1, y1)`` of the text line in PDF coordinates.
    direction : str
        ``"horizontal"`` or ``"vertical"`` text direction.
    """

    row: int
    col: int
    text: str
    bbox: tuple[float, float, float, float]
    direction: str = "horizontal"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON friendly representation."""
        return {
            "row": self.row,
            "col": self.col,
            "text": self.text,
            "bbox": [float(v) for v in self.bbox],
            "direction": self.direction,
        }


@dataclass
class StructuralCell:
    """A cell of the structural view (merged or not).

    Attributes
    ----------
    row, col : int
        Top-left anchor, zero based, in the table's grid.
    rowspan, colspan : int
        Number of grid rows / columns covered by the cell.
    page : int
        Original PDF page number of the physical table.
    order : int
        Table order on that page.
    bbox : tuple
        ``(x0, y0, x1, y1)`` in PDF coordinates (left-bottom origin).
    text : str
        One retained copy of the original extracted text for the cell.
    sources : list of TextSource
        Every original text fragment assigned to the cell.  Two fragments
        carrying the same string from two positions remain two sources.
    """

    row: int
    col: int
    rowspan: int
    colspan: int
    page: int
    order: int
    bbox: tuple[float, float, float, float]
    text: str
    sources: list[TextSource] = field(default_factory=list)

    @property
    def is_merged(self) -> bool:
        """Whether the cell covers more than one grid position."""
        return self.rowspan > 1 or self.colspan > 1

    @property
    def anchor(self) -> tuple[int, int]:
        """Top-left grid position covered by the cell."""
        return (self.row, self.col)

    @property
    def positions(self) -> list[tuple[int, int]]:
        """All grid positions covered by the cell, in reading order."""
        return [
            (r, c)
            for r in range(self.row, self.row + self.rowspan)
            for c in range(self.col, self.col + self.colspan)
        ]

    def covers(self, row: int, col: int) -> bool:
        """Whether the cell covers the given grid position."""
        return (
            self.row <= row < self.row + self.rowspan
            and self.col <= col < self.col + self.colspan
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON friendly representation."""
        return {
            "row": self.row,
            "col": self.col,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
            "page": self.page,
            "order": self.order,
            "bbox": [float(v) for v in self.bbox],
            "text": self.text,
            "is_merged": self.is_merged,
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass
class StructureDiagnostic:
    """A place where the structure evidence is not conclusive.

    Attributes
    ----------
    kind : str
        One of ``"partial_seam"``, ``"seam_conflict"``,
        ``"non_rectangular_component"`` or ``"text_assignment_mismatch"``.
    page : int
        Original PDF page number.
    order : int
        Table order on that page.
    positions : list of tuple
        Grid positions affected by the question.
    message : str
        Human readable explanation.
    details : dict
        Extra evidence (coverage, orientation, bounding rectangle, texts...).
    """

    kind: str
    page: int
    order: int
    positions: list[tuple[int, int]]
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON friendly representation."""
        out = {
            "kind": self.kind,
            "page": self.page,
            "order": self.order,
            "grid_positions": [[r, c] for r, c in self.positions],
            "message": self.message,
        }
        out.update(self.details)
        return out


class TableStructure:
    """Structural cells and diagnostics for one physical lattice table.

    Instances are created lazily by :attr:`camelot.core.Table.structure`.
    ``cells`` covers every grid position exactly once: merged regions appear
    once at their top-left anchor, plain and standalone empty cells appear as
    ``rowspan == colspan == 1`` entries.
    """

    def __init__(
        self,
        page: int,
        order: int,
        shape: tuple[int, int],
        cells: list[StructuralCell],
        diagnostics: list[StructureDiagnostic],
        seams: list[dict[str, Any]],
    ) -> None:
        self.page = page
        self.order = order
        self.shape = shape
        self.cells = cells
        self.diagnostics = diagnostics
        self.seams = seams
        self._by_position: dict[tuple[int, int], StructuralCell] = {}
        for cell in cells:
            for position in cell.positions:
                self._by_position[position] = cell

    def __repr__(self) -> str:  # noqa D105
        merged = sum(1 for cell in self.cells if cell.is_merged)
        return (
            f"<TableStructure page={self.page} order={self.order} "
            f"shape={self.shape} merged={merged} "
            f"diagnostics={len(self.diagnostics)}>"
        )

    def __iter__(self):
        """Iterate over the structural cells in reading order."""
        return iter(self.cells)

    def __len__(self) -> int:
        """Number of structural cells (merged regions count once)."""
        return len(self.cells)

    @property
    def merged_cells(self) -> list[StructuralCell]:
        """Only the cells that cover more than one grid position."""
        return [cell for cell in self.cells if cell.is_merged]

    def cell_at(self, row: int, col: int) -> StructuralCell:
        """Return the structural cell covering a grid position."""
        return self._by_position[(row, col)]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON friendly representation."""
        return {
            "page": self.page,
            "order": self.order,
            "shape": list(self.shape),
            "cells": [cell.to_dict() for cell in self.cells],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "seams": list(self.seams),
        }


# ---------------------------------------------------------------------------
# Evidence collection / seam classification
# ---------------------------------------------------------------------------


def _clip(lo: float, hi: float, a: float, b: float) -> tuple[float, float] | None:
    """Clip ``[a, b]`` to ``[lo, hi]``; return None when empty."""
    x0 = max(lo, min(a, b))
    x1 = min(hi, max(a, b))
    if x1 - x0 <= 1e-6:
        return None
    return (x0, x1)


class _SegmentIndex:
    """Index line segments by their fixed coordinate for fast seam queries."""

    def __init__(self, segments, fixed_index, lo_index, hi_index):
        items = []
        for seg in segments:
            fixed = float(seg[fixed_index])
            lo = min(float(seg[lo_index]), float(seg[hi_index]))
            hi = max(float(seg[lo_index]), float(seg[hi_index]))
            items.append((fixed, lo, hi))
        items.sort(key=lambda item: item[0])
        self._items = items
        self._coords = [item[0] for item in items]

    def intervals_near(self, coord, tolerance):
        """Yield ``(lo, hi)`` intervals whose fixed coordinate is near coord."""
        left = bisect.bisect_left(self._coords, coord - tolerance)
        right = bisect.bisect_right(self._coords, coord + tolerance)
        return [(lo, hi) for _fixed, lo, hi in self._items[left:right]]


def _table_segments(table, bbox):
    """Return table line segments clipped to the table bounding box."""
    evidence = getattr(table, "_structure_evidence", None) or {}
    segments = getattr(table, "_segments", None)
    if evidence.get("vertical_segments") is not None:
        vertical = evidence["vertical_segments"]
        horizontal = evidence["horizontal_segments"]
    elif segments is not None:
        vertical, horizontal = segments
    else:
        raise StructureUnavailableError(
            "Table has no detected line segments; the structural cell view is "
            "only available for physical tables parsed with flavor='lattice'."
        )
    x0, y0, x1, y1 = bbox
    pad = SEAM_COORD_TOLERANCE
    v_keep = [
        seg
        for seg in vertical
        if x0 - pad <= float(seg[0]) <= x1 + pad
        and max(float(seg[1]), float(seg[3])) >= y0 - pad
        and min(float(seg[1]), float(seg[3])) <= y1 + pad
    ]
    h_keep = [
        seg
        for seg in horizontal
        if y0 - pad <= float(seg[1]) <= y1 + pad
        and max(float(seg[0]), float(seg[2])) >= x0 - pad
        and min(float(seg[0]), float(seg[2])) <= x1 + pad
    ]
    return v_keep, h_keep


def _classify_seam(coverage: float, flag_upper: bool, flag_lower: bool) -> str:
    """Classify one seam from line coverage and the lattice edge flags."""
    flags_open = not flag_upper and not flag_lower
    if coverage <= OPEN_SEAM_COVERAGE:
        # No line pixels: only an open seam when the lattice also recorded no
        # boundary on either side.  A recorded boundary without pixels is a
        # contradiction and must not be swallowed.
        return "open" if flags_open else "conflict"
    if coverage >= SOLID_SEAM_COVERAGE:
        # A clearly drawn separator always wins over the flags.
        if flags_open:
            # The line covers most of the seam but the lattice did not record
            # the boundary (typically a short break in the line): ambiguous.
            return "ambiguous"
        return "solid"
    # Partly covered seam: a short break in a line.
    return "ambiguous"


def _seam_text(table, positions):
    """Return the pre-shift texts retained at the given grid positions.

    Uses the evidence recorded during parsing, not the flat matrix, so the
    diagnostics do not change when ``shift_text``/``copy_text`` run.
    """
    raw = getattr(table, "_raw_text_cells", None) or []
    texts: dict[tuple[int, int], list[str]] = {}
    for entry in raw:
        key = (int(entry["row"]), int(entry["col"]))
        texts.setdefault(key, []).append(entry["text"])
    out = []
    for r, c in positions:
        value = "".join(texts.get((r, c), []))
        out.append({"row": r, "col": c, "text": value})
    return out


def _diagnostic_for_seam(table, orientation, row, col, coverage, kind):
    if orientation == "vertical":
        positions = [(row, col - 1), (row, col)]
        where = f"vertical seam between columns {col - 1} and {col} of row {row}"
    else:
        positions = [(row - 1, col), (row, col)]
        where = f"horizontal seam between rows {row - 1} and {row} of column {col}"
    if kind == "conflict":
        message = (
            f"Line evidence and lattice edge flags disagree at the {where}; "
            "the cells are kept separate and their texts retained."
        )
        code = "seam_conflict"
    elif coverage >= SOLID_SEAM_COVERAGE:
        message = (
            f"A line covers {coverage:.0%} of the {where}, but the lattice "
            "recorded no boundary for the adjacent cells. The seam is not "
            "treated as a merge; the cells are kept separate and their texts "
            "retained."
        )
        code = "partial_seam"
    else:
        message = (
            f"The {where} is only partly covered by a line "
            f"({coverage:.0%}); it is not treated as a merge. The cells are "
            "kept separate and their texts retained."
        )
        code = "partial_seam"
    return StructureDiagnostic(
        kind=code,
        page=table.page,
        order=table.order,
        positions=positions,
        message=message,
        details={
            "orientation": orientation,
            "coverage": round(float(coverage), 4),
            "grid_positions": [[r, c] for r, c in positions],
            "texts": _seam_text(table, positions),
        },
    )


def _raw_text_sources(table):
    """Return the pre-shift text assignment evidence recorded during parsing."""
    raw = getattr(table, "_raw_text_cells", None)
    if raw is None:
        raise StructureUnavailableError(
            "Table has no recorded text-assignment evidence; the structural "
            "cell view is only available for physical tables parsed with "
            "flavor='lattice'."
        )
    return raw


def _join_sources(sources: list[TextSource]) -> str:
    """Keep one copy of the retained text of a region.

    Sources are kept in reading order.  A source whose whole text repeats the
    whole text of an earlier source is not appended again, so the same string
    written at two positions contributes one copy while both sources remain
    listed.  Repetition *inside* one fragment is never removed.
    """
    kept: list[TextSource] = []
    seen: set[str] = set()
    for source in sources:
        key = source.text.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(source)
    return "".join(source.text for source in kept)


def _build_cells(table, components):
    """Turn accepted rectangular components into StructuralCells."""
    raw = table._raw_text_cells
    by_position: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for entry in raw:
        by_position.setdefault((int(entry["row"]), int(entry["col"])), []).append(
            entry
        )

    cells: list[StructuralCell] = []
    for component in components:
        rows = [r for r, _c in component]
        cols = [c for _r, c in component]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        selected = []
        for position in sorted(component):
            for entry in by_position.get(position, []):
                selected.append(
                    TextSource(
                        row=int(entry["row"]),
                        col=int(entry["col"]),
                        text=entry["text"],
                        bbox=tuple(float(v) for v in entry["bbox"]),
                        direction=entry.get("direction", "horizontal"),
                    )
                )
        text = _join_sources(selected)
        bbox = (
            float(table.cols[c0][0]),
            float(table.rows[r1][1]),
            float(table.cols[c1][1]),
            float(table.rows[r0][0]),
        )
        cells.append(
            StructuralCell(
                row=r0,
                col=c0,
                rowspan=r1 - r0 + 1,
                colspan=c1 - c0 + 1,
                page=int(table.page),
                order=int(table.order),
                bbox=bbox,
                text=text,
                sources=selected,
            )
        )
    cells.sort(key=lambda cell: (cell.row, cell.col))
    return cells


def _scan_vertical_seams(table, index, tolerance):
    """Classify every vertical interior seam of the table.

    Returns a list of ``(row, col, coverage, kind)`` tuples.  ``col`` is the
    boundary index: the seam lies between columns ``col - 1`` and ``col``.
    """
    found = []
    for row in range(len(table.rows)):
        y_lo = float(min(table.rows[row]))
        y_hi = float(max(table.rows[row]))
        height = y_hi - y_lo
        if height <= 0:
            continue
        for col in range(1, len(table.cols)):
            x = float(table.cols[col][0])
            intervals = []
            for lo, hi in index.intervals_near(x, tolerance):
                clipped = _clip(y_lo, y_hi, lo, hi)
                if clipped is not None:
                    intervals.append(clipped)
            coverage = _union_length(intervals) / height
            flag_upper = bool(table.cells[row][col - 1].right)
            flag_lower = bool(table.cells[row][col].left)
            found.append(
                (row, col, coverage, _classify_seam(coverage, flag_upper, flag_lower))
            )
    return found


def _scan_horizontal_seams(table, index, tolerance):
    """Classify every horizontal interior seam of the table.

    Returns a list of ``(row, col, coverage, kind)`` tuples.  ``row`` is the
    boundary index: the seam lies between rows ``row - 1`` and ``row``.
    """
    found = []
    for row in range(1, len(table.rows)):
        y = float(table.rows[row][0])
        for col in range(len(table.cols)):
            x_lo = float(min(table.cols[col]))
            x_hi = float(max(table.cols[col]))
            width = x_hi - x_lo
            if width <= 0:
                continue
            intervals = []
            for lo, hi in index.intervals_near(y, tolerance):
                clipped = _clip(x_lo, x_hi, lo, hi)
                if clipped is not None:
                    intervals.append(clipped)
            coverage = _union_length(intervals) / width
            flag_upper = bool(table.cells[row - 1][col].bottom)
            flag_lower = bool(table.cells[row][col].top)
            found.append(
                (row, col, coverage, _classify_seam(coverage, flag_upper, flag_lower))
            )
    return found


def _components_to_regions(table, components):
    """Accept exact rectangles and report ragged components."""
    diagnostics: list[StructureDiagnostic] = []
    accepted: list[list[tuple[int, int]]] = []
    for members in components:
        rows = {r for r, _c in members}
        cols = {c for _r, c in members}
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(members) == len(rows) * len(cols):
            accepted.append(sorted(members))
            continue
        member_set = set(members)
        missing = [
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if (r, c) not in member_set
        ]
        diagnostics.append(
            StructureDiagnostic(
                kind="non_rectangular_component",
                page=table.page,
                order=table.order,
                positions=sorted(members),
                message=(
                    "Missing separators connect the listed grid positions into "
                    "a non-rectangular (for example L-shaped) region. More "
                    "than one rectangular interpretation is possible, so no "
                    "merge is created and every cell keeps its own text."
                ),
                details={
                    "bounding_box": [r0, c0, r1, c1],
                    "missing_positions": [[r, c] for r, c in missing],
                    "texts": _seam_text(table, sorted(members)),
                },
            )
        )
        accepted.extend([[(r, c)] for r, c in sorted(members)])
    return accepted, diagnostics


def _text_mismatch_diagnostics(table, cells, tolerance):
    """Report text fragments assigned outside the structural cell geometry."""
    diagnostics: list[StructureDiagnostic] = []
    for cell in cells:
        for source in cell.sources:
            x0, y0, x1, y1 = source.bbox
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            bx0, by0, bx1, by1 = cell.bbox
            inside = (
                bx0 - tolerance <= cx <= bx1 + tolerance
                and by0 - tolerance <= cy <= by1 + tolerance
            )
            if inside:
                continue
            diagnostics.append(
                StructureDiagnostic(
                    kind="text_assignment_mismatch",
                    page=table.page,
                    order=table.order,
                    positions=[(source.row, source.col)],
                    message=(
                        "A text fragment was assigned to this cell by the "
                        "text stage but its centre lies outside the cell "
                        "geometry. Its text is retained in the cell."
                    ),
                    details={
                        "text": source.text,
                        "text_bbox": [float(v) for v in source.bbox],
                        "cell_bbox": [float(v) for v in cell.bbox],
                    },
                )
            )
    return diagnostics


def build_table_structure(table) -> TableStructure:
    """Compute the structural view for a physical lattice table.

    Parameters
    ----------
    table : camelot.core.Table
        A physical table produced by the lattice parser.  Its ``cols``,
        ``rows``, edge flags, line segments and pre-shift text assignment
        evidence are used read-only.

    Returns
    -------
    TableStructure

    Raises
    ------
    StructureUnavailableError
        If the table was not produced by the lattice parser or lacks the
        line/text evidence recorded during page parsing.
    """
    flavor = getattr(table, "flavor", None)
    if flavor != "lattice":
        raise StructureUnavailableError(
            "The structural cell view is only available for physical tables "
            f"parsed with flavor='lattice' (got flavor={flavor!r})."
        )
    evidence = getattr(table, "_structure_evidence", None)
    if not evidence:
        raise StructureUnavailableError(
            "Table has no lattice parsing evidence (it may be a logical or "
            "rebuilt table); the structural cell view is unavailable."
        )
    _raw_text_sources(table)

    joint_tol = float(evidence.get("joint_tol", SEAM_COORD_TOLERANCE))
    tolerance = max(SEAM_COORD_TOLERANCE, joint_tol)
    n_rows = len(table.rows)
    n_cols = len(table.cols)
    bbox = tuple(float(v) for v in table._bbox)

    vertical, horizontal = _table_segments(table, bbox)
    v_index = _SegmentIndex(vertical, fixed_index=0, lo_index=1, hi_index=3)
    h_index = _SegmentIndex(horizontal, fixed_index=1, lo_index=0, hi_index=2)

    parent = {(r, c): (r, c) for r in range(n_rows) for c in range(n_cols)}

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    diagnostics: list[StructureDiagnostic] = []
    seams: list[dict[str, Any]] = []

    for row, col, coverage, kind in _scan_vertical_seams(
        table, v_index, tolerance
    ):
        seams.append(
            {
                "orientation": "vertical",
                "row": row,
                "col": col,
                "coverage": round(float(coverage), 4),
                "kind": kind,
            }
        )
        if kind == "open":
            union((row, col - 1), (row, col))
        elif kind in ("ambiguous", "conflict"):
            diagnostics.append(
                _diagnostic_for_seam(
                    table, "vertical", row, col, coverage, kind
                )
            )

    for row, col, coverage, kind in _scan_horizontal_seams(
        table, h_index, tolerance
    ):
        seams.append(
            {
                "orientation": "horizontal",
                "row": row,
                "col": col,
                "coverage": round(float(coverage), 4),
                "kind": kind,
            }
        )
        if kind == "open":
            union((row - 1, col), (row, col))
        elif kind in ("ambiguous", "conflict"):
            diagnostics.append(
                _diagnostic_for_seam(
                    table, "horizontal", row, col, coverage, kind
                )
            )

    # Group grid positions into components; accept only exact rectangles.
    grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for position in parent:
        grouped.setdefault(find(position), []).append(position)

    components, component_diagnostics = _components_to_regions(
        table, grouped.values()
    )
    diagnostics.extend(component_diagnostics)

    cells = _build_cells(table, components)
    diagnostics.extend(_text_mismatch_diagnostics(table, cells, tolerance))

    return TableStructure(
        page=int(table.page),
        order=int(table.order),
        shape=(n_rows, n_cols),
        cells=cells,
        diagnostics=diagnostics,
        seams=seams,
    )


# ---------------------------------------------------------------------------
# HTML rendering (also used by Table.to_html(keep_merges=True))
# ---------------------------------------------------------------------------


def render_structure_html(
    structure: TableStructure,
    classes: str | None = None,
    border: int = 1,
    escape: bool = True,
) -> str:
    """Render a :class:`TableStructure` as an HTML table with real spans."""
    anchors = {(cell.row, cell.col): cell for cell in structure.cells}
    n_rows, n_cols = structure.shape
    class_attr = f' class="{_html.escape(classes)}"' if classes else ""
    lines = [f'<table border="{int(border)}"{class_attr}>']
    for row in range(n_rows):
        lines.append("<tr>")
        for col in range(n_cols):
            cell = anchors.get((row, col))
            if cell is None:
                continue
            attrs = []
            if cell.rowspan > 1:
                attrs.append(f'rowspan="{cell.rowspan}"')
            if cell.colspan > 1:
                attrs.append(f'colspan="{cell.colspan}"')
            attr_text = (" " + " ".join(attrs)) if attrs else ""
            value = cell.text.strip()
            if escape:
                value = _html.escape(value)
            lines.append(f"<td{attr_text}>{value}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def write_structure_excel(
    structure: TableStructure, path, sheet_name: str
) -> None:
    """Write a :class:`TableStructure` to an xlsx file with real merges."""
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = sheet_name[:31]
    for cell in structure.cells:
        value = cell.text.strip()
        sheet.cell(
            row=cell.row + 1,
            column=cell.col + 1,
            value=value if value != "" else None,
        )
        if cell.is_merged:
            sheet.merge_cells(
                start_row=cell.row + 1,
                start_column=cell.col + 1,
                end_row=cell.row + cell.rowspan,
                end_column=cell.col + cell.colspan,
            )
    workbook.save(path)
