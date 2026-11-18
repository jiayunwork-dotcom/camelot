"""Independent PDF generator for cross-page continuation tests.

The generator renders multi-page reports with reportlab and writes the ground
truth (cell matrices, fragment membership, placement coordinates, captions)
to a JSON sidecar.  It never imports camelot or any of the continuation
recognition code: expected results come exclusively from what this module
placed on the pages.

Two flavors of PDF are produced with the *same* semantic layout:

* ``stream`` PDFs contain only precisely positioned text (no ruling lines);
* ``lattice`` PDFs additionally draw ruling lines around every cell, with
  one deliberately interrupted vertical segment.

Layout coverage (see ``build_report``):

1. A table spanning three pages with a two-row header, shifted horizontally
   on continuation pages, with a wrapped header line and a wider page.
2. Two independent tables on the same page, each continuing on the next
   page.
3. Adjacent tables with identical headers but distinct table numbers.
4. A chapter switch between two same-shaped tables.
5. A body row containing a header word and a row with an empty first cell.
6. A continuation page where one whole column is empty (stream then loses a
   column on its own; lattice additionally gets an interrupted line).
7. An ambiguity page where two equally strong predecessor fragments compete
   for one continuation fragment.

The page numbering deliberately leaves a selectable gap so tests can verify
that missing intermediate pages block merging.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from dataclasses import field

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_SIZE = 9
LINE_GAP = 1.6
ROW_HEIGHT = 18
TOP_MARGIN = 50
BOTTOM_MARGIN = 45
CAPTION_GAP_PT = 26
TABLE_X_GAP = 28
DEFAULT_WIDTH, DEFAULT_HEIGHT = A4


@dataclass
class PlacedCell:
    """One placed cell with its exact PDF coordinates."""

    row: int
    col: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class FragmentSpec:
    """A physical fragment placed on a page."""

    logical: str
    page: int
    order: int
    x: float
    top_y: float
    col_widths: list[float]
    page_width: float
    page_height: float
    matrix: list[list[str]] = field(default_factory=list)
    cells: list[PlacedCell] = field(default_factory=list)
    caption: str | None = None
    section_heading: str | None = None
    continued_caption: bool = False
    interrupted_col: int | None = None
    bbox: tuple[float, float] | None = None


class ReportBuilder:
    """Render a report page by page and accumulate ground truth."""

    def __init__(self, path: str, flavor: str):
        assert flavor in ("stream", "lattice")
        self.path = path
        self.flavor = flavor
        self.c = canvas.Canvas(path, pagesize=A4)
        self.page_width, self.page_height = A4
        self.page_no = 0
        self.fragments: list[FragmentSpec] = []
        self.logicals: dict[str, dict] = {}
        self._start_page()

    # -- page management -------------------------------------------------

    def _start_page(self, width: float | None = None, height: float | None = None):
        if self.page_no:
            self.c.showPage()
        self.page_no += 1
        self.page_width = width or DEFAULT_WIDTH
        self.page_height = height or DEFAULT_HEIGHT
        self.c.setPageSize((self.page_width, self.page_height))

    def new_page(self, width: float | None = None, height: float | None = None):
        """Start a new page (numbering stays contiguous in the PDF)."""
        self._start_page(width, height)

    def save(self):
        self.c.showPage()
        self.c.save()

    # -- rendering --------------------------------------------------------

    def _draw_text(self, text, x, y, bold=False):
        if text == "":
            width = 0.0
        else:
            font = FONT_BOLD if bold else FONT
            self.c.setFont(font, FONT_SIZE)
            self.c.drawString(x, y, text)
            width = self.c.stringWidth(text, font, FONT_SIZE)
        return width

    def draw_section_heading(self, text: str, x: float, top_y: float) -> float:
        """Draw a bold chapter heading and return the new cursor y."""
        self.c.setFont(FONT_BOLD, FONT_SIZE + 3)
        self.c.drawString(x, top_y, text)
        self.c.setFont(FONT, FONT_SIZE)
        return top_y - ROW_HEIGHT - 6

    def draw_fragment(
        self,
        logical: str,
        matrix: list[list[str]],
        col_widths: list[float],
        x: float,
        top_y: float,
        header_depth: int,
        caption: str | None = None,
        continued_caption: bool = False,
        section_heading: str | None = None,
        x_shift: float = 0.0,
        interrupted_col: int | None = None,
        page_width: float | None = None,
        page_height: float | None = None,
        wrapped_header: bool = False,
    ) -> FragmentSpec:
        """Render one physical table fragment and record its truth."""
        if page_width is not None or page_height is not None:
            self.new_page(page_width, page_height)
        c = self.c
        width = sum(col_widths)
        x += x_shift
        cursor_y = top_y

        if section_heading:
            cursor_y = self.draw_section_heading(section_heading, x, cursor_y)

        if caption:
            c.setFont(FONT_BOLD, FONT_SIZE)
            c.drawString(x, cursor_y, caption)
            cursor_y -= CAPTION_GAP_PT
            c.setFont(FONT, FONT_SIZE)

        spec = FragmentSpec(
            logical=logical,
            page=self.page_no,
            order=sum(f.page == self.page_no for f in self.fragments) + 1,
            x=x,
            top_y=cursor_y,
            col_widths=list(col_widths),
            page_width=self.page_width,
            page_height=self.page_height,
            caption=caption,
            section_heading=section_heading,
            continued_caption=continued_caption,
            interrupted_col=interrupted_col,
        )

        table_top = cursor_y
        for r, row in enumerate(matrix):
            is_header = r < header_depth
            row_baseline = cursor_y - ROW_HEIGHT + 4
            for col, value in enumerate(row):
                cell_x = x + sum(col_widths[:col])
                w = col_widths[col]
                # Header text may wrap into two physical visual lines on
                # continuation pages; the data model still carries one cell.
                display = value
                if (
                    is_header
                    and wrapped_header
                    and r == header_depth - 1
                    and " " in value
                ):
                    parts = value.split(" ", 1)
                    display = parts[0]
                bold = is_header
                text_w = self._draw_text(display, cell_x + 3, row_baseline, bold)
                if display and wrapped_header and is_header and " " in value:
                    second = value.split(" ", 1)[1]
                    self._draw_text(
                        second,
                        cell_x + 3,
                        row_baseline - FONT_SIZE - LINE_GAP,
                        bold,
                    )
                    w2 = self.c.stringWidth(second, FONT_BOLD, FONT_SIZE)
                    text_w = max(text_w, w2)
                spec.cells.append(
                    PlacedCell(
                        row=r,
                        col=col,
                        text=value,
                        x0=cell_x + 3,
                        y0=row_baseline - 2,
                        x1=cell_x + 3 + text_w,
                        y1=row_baseline + FONT_SIZE,
                    )
                )
            spec.matrix.append(list(row))
            cursor_y -= ROW_HEIGHT

        table_bottom = cursor_y
        spec.bbox = (x, table_bottom, x + width, table_top)

        if self.flavor == "lattice":
            self._draw_rules(spec, interrupted_col)

        self.fragments.append(spec)
        self.logicals.setdefault(
            logical,
            {
                "logical_id": logical,
                "header_depth": header_depth,
                "n_cols": len(col_widths),
                "fragments": [],
                "full_matrix": [],
            },
        )
        self.logicals[logical]["fragments"].append([spec.page, spec.order])
        return spec

    def _draw_rules(self, spec: FragmentSpec, interrupted_col: int | None):
        """Draw lattice ruling lines; optionally interrupt one vertical."""
        c = self.c
        c.setLineWidth(0.7)
        x0, y0, x1, y1 = spec.bbox
        n_rows = len(spec.matrix)
        n_cols = len(spec.col_widths)
        # Horizontal rules (full width), including row separators.
        for r in range(n_rows + 1):
            y = y1 - r * ROW_HEIGHT
            c.line(x0, y, x1, y)
        # Vertical rules.
        for col in range(n_cols + 1):
            x = x0 + sum(spec.col_widths[:col])
            if (
                interrupted_col is not None
                and col == interrupted_col
                and 0 < col < n_cols
            ):
                # Interrupted interior line: keep short stubs at the top and
                # bottom so text in the column can still be detected but the
                # segment does not cross the middle data rows.
                stub = ROW_HEIGHT * 0.9
                c.line(x, y1, x, y1 - stub)
                c.line(x, y0, x, y0 + stub)
            else:
                c.line(x, y0, x, y1)

    # -- output -----------------------------------------------------------

    def truth(self) -> dict:
        return {
            "flavor": self.flavor,
            "pdf_pages": self.page_no,
            "fragments": [
                {
                    "logical": f.logical,
                    "page": f.page,
                    "order": f.order,
                    "bbox": list(f.bbox),
                    "col_widths": list(f.col_widths),
                    "matrix": f.matrix,
                    "caption": f.caption,
                    "continued_caption": f.continued_caption,
                    "section_heading": f.section_heading,
                    "interrupted_col": f.interrupted_col,
                    "cells": [
                        {
                            "row": cell.row,
                            "col": cell.col,
                            "text": cell.text,
                            "bbox": [cell.x0, cell.y0, cell.x1, cell.y1],
                        }
                        for cell in f.cells
                    ],
                }
                for f in self.logicals_ordered()
            ],
            "logicals": {
                lid: {
                    "header_depth": meta["header_depth"],
                    "n_cols": meta["n_cols"],
                    "fragments": meta["fragments"],
                }
                for lid, meta in self.logicals.items()
            },
        }

    def logicals_ordered(self) -> list[FragmentSpec]:
        return sorted(self.fragments, key=lambda f: (f.page, f.order))


# ---------------------------------------------------------------------------
# Scenario layout
# ---------------------------------------------------------------------------


def _data_rows(prefix, n, n_cols, start_idx=0, empty_col=None, header_words=()):
    """Build deterministic body rows with an empty cell and a header word."""
    rows = []
    for i in range(n):
        idx = start_idx + i
        row = [f"{prefix}-r{idx}-c{col}" for col in range(n_cols)]
        if empty_col is not None:
            row[empty_col] = ""
        if i == 1:
            # row whose first cell is empty
            row[0] = ""
        if i == 2 and header_words:
            # body text equal to a header word must not be removed
            row[1] = header_words[0]
        rows.append(row)
    return rows


def build_report(path: str, flavor: str) -> dict:
    """Build the full clear-layout report and return its ground truth."""
    b = ReportBuilder(path, flavor)

    # --- Logical A: three-page, two-row header, shift + page width change.
    header_a = [["Region A", "Metric A", "Unit A"], ["Name A", "Value A", "Scale A"]]
    widths_a = [150, 120, 110]
    a_rows_1 = _data_rows("A", 4, 3, header_words=("Value A",))
    b.draw_fragment(
        "A",
        header_a + a_rows_1,
        widths_a,
        x=60,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=2,
        caption="Table 1: Regional metrics",
    )
    # Page 2: shifted horizontally, wrapped header.
    b.new_page()
    a_rows_2 = _data_rows("A", 4, 3, start_idx=4)
    b.draw_fragment(
        "A",
        header_a + a_rows_2,
        widths_a,
        x=60,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=2,
        caption="Table 1 (continued)",
        continued_caption=True,
        x_shift=35,
        wrapped_header=True,
    )
    # Page 3: wider page + further shift.
    b.new_page(width=DEFAULT_WIDTH + 90)
    a_rows_3 = _data_rows("A", 4, 3, start_idx=8)
    b.draw_fragment(
        "A",
        header_a + a_rows_3,
        widths_a,
        x=60,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=2,
        caption="Table 1 (continued)",
        continued_caption=True,
        x_shift=60,
    )

    # --- Logical B and C: two independent tables on one page, both
    # continuing on the next page.
    b.new_page()  # page 4
    header_b = [["Item B", "Qty B", "Price B"]]
    header_c = [["Item C", "Qty C", "Price C"]]
    widths_bc = [140, 100, 110]
    b_rows = _data_rows("B", 4, 3, header_words=("Qty B",))
    b.draw_fragment(
        "B",
        header_b + b_rows,
        widths_bc,
        x=50,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
        caption="Table 2: Inventory one",
    )
    c_rows = _data_rows("C", 4, 3)
    b.draw_fragment(
        "C",
        header_c + c_rows,
        widths_bc,
        x=355,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
        caption="Table 3: Inventory two",
    )
    b.new_page()  # page 5
    b.draw_fragment(
        "B",
        header_b + _data_rows("B", 4, 3, start_idx=4),
        widths_bc,
        x=50,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
        caption="Table 2 (continued)",
        continued_caption=True,
    )
    b.draw_fragment(
        "C",
        header_c + _data_rows("C", 4, 3, start_idx=4),
        widths_bc,
        x=355,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
        caption="Table 3 (continued)",
        continued_caption=True,
    )

    # --- Logical D / E: same header shape, distinct table numbers ->
    # must never merge.
    b.new_page()  # page 6
    shared_header = [["Code", "Amount", "Note"]]
    widths_de = [130, 120, 130]
    b.draw_fragment(
        "D",
        shared_header + _data_rows("D", 4, 3, header_words=("Amount",)),
        widths_de,
        x=60,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
        caption="Table 4: Alpha ledger",
    )
    b.new_page()  # page 7
    b.draw_fragment(
        "E",
        shared_header + _data_rows("E", 4, 3),
        widths_de,
        x=60,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
        caption="Table 5: Beta ledger",
    )

    # --- Logical F / G: chapter switch blocks an otherwise identical shape.
    b.new_page()  # page 8
    header_fg = [["Key", "Value", "Remark"]]
    b.draw_fragment(
        "F",
        header_fg + _data_rows("F", 4, 3),
        widths_de,
        x=60,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
        caption="Table 6: Chapter one figures",
    )
    b.new_page()  # page 9
    b.draw_fragment(
        "G",
        header_fg + _data_rows("G", 4, 3),
        widths_de,
        x=60,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
        caption="Table 7: Chapter two figures",
        section_heading="Chapter 2: New scope",
    )

    # --- Logical H: continuation page with one whole empty column.
    # Stream loses the column on a single-page extraction; lattice also has
    # an interrupted vertical line.
    b.new_page()  # page 10
    header_h = [["Product H", "Stock H", "Bin H"]]
    b.draw_fragment(
        "H",
        header_h + _data_rows("H", 4, 3, header_words=("Stock H",)),
        widths_bc,
        x=60,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
        caption="Table 8: Warehouse stock",
    )
    b.new_page()  # page 11
    empty_rows = _data_rows("H", 4, 3, start_idx=4, empty_col=1)
    b.draw_fragment(
        "H",
        header_h + empty_rows,
        widths_bc,
        x=60,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
        caption="Table 8 (continued)",
        continued_caption=True,
        interrupted_col=2 if flavor == "lattice" else None,
    )

    # --- Logical I: isolated page used to test missing-page selection.
    # Placed on page 13; page 12 stays blank.  Tests select pages 10,11,13
    # and verify the gap blocks the H/I merge.
    b.new_page()  # page 12 blank
    b.new_page()  # page 13
    header_i = [["Product I", "Stock I", "Bin I"]]
    b.draw_fragment(
        "I",
        header_i + _data_rows("I", 4, 3),
        widths_bc,
        x=60,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
        caption="Table 9: Separate stock",
    )

    b.save()
    truth = b.truth()
    # Logical full matrices (headers once, body concatenated).
    for lid, meta in truth["logicals"].items():
        full = []
        for page, order in meta["fragments"]:
            frag = next(
                f
                for f in truth["fragments"]
                if f["page"] == page and f["order"] == order
            )
            start = 0 if not full else meta["header_depth"]
            full.extend(frag["matrix"][start:])
        meta["full_matrix"] = full
    return truth


def build_ambiguity_report(path: str, flavor: str) -> dict:
    """Two equal predecessors competing for one continuation fragment."""
    b = ReportBuilder(path, flavor)
    # Deliberately no captions/table ids: geometry + header are equal, so
    # auto must report ambiguity instead of picking one.
    header_p = [["Label", "Amount", "Detail"]]
    widths = [150, 120, 130]
    b.draw_fragment(
        "P1",
        header_p + _data_rows("P1", 4, 3),
        widths,
        x=50,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
    )
    b.draw_fragment(
        "P2",
        header_p + _data_rows("P2", 4, 3),
        widths,
        x=375,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
    )
    b.new_page()
    # Single fragment, horizontally centered between the two predecessors
    # (same width, same relative column structure), same header.  The
    # predecessor tables keep a clear horizontal gap so the only ambiguity
    # is which one the continuation belongs to -- not where columns split.
    b.draw_fragment(
        "Q",
        header_p + _data_rows("Q", 4, 3),
        widths,
        x=215,
        top_y=b.page_height - TOP_MARGIN,
        header_depth=1,
    )
    b.save()
    truth = b.truth()
    for lid, meta in truth["logicals"].items():
        full = []
        for page, order in meta["fragments"]:
            frag = next(
                f
                for f in truth["fragments"]
                if f["page"] == page and f["order"] == order
            )
            start = 0 if not full else meta["header_depth"]
            full.extend(frag["matrix"][start:])
        meta["full_matrix"] = full
    truth["ambiguity_expected"] = {
        "candidate": [2, 1],
        "predecessors": [[1, 1], [1, 2]],
    }
    return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", help="Directory for generated PDFs and truth")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for flavor in ("stream", "lattice"):
        pdf_path = os.path.join(args.outdir, f"continuation_{flavor}.pdf")
        truth = build_report(pdf_path, flavor)
        with open(
            os.path.join(args.outdir, f"continuation_{flavor}_truth.json"),
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(truth, fh, ensure_ascii=False, indent=2)

        amb_path = os.path.join(args.outdir, f"ambiguity_{flavor}.pdf")
        amb_truth = build_ambiguity_report(amb_path, flavor)
        with open(
            os.path.join(args.outdir, f"ambiguity_{flavor}_truth.json"),
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(amb_truth, fh, ensure_ascii=False, indent=2)
    print("generated in", args.outdir)


if __name__ == "__main__":
    main()
