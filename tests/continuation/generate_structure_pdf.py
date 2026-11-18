"""Independent generator for the structural-cell test PDF.

The generator only uses reportlab and never imports camelot: the expected
merged regions, standalone empty cells, duplicate-source region, short line
break and L-shaped ambiguity come exclusively from what was drawn here.

The generated PDF and its ground-truth JSON are committed under
``tests/files`` so the test suite does not depend on reportlab at run time.
"""

from __future__ import annotations

import json
import os
import sys

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


FONT = "Helvetica"
FONT_SIZE = 8


def build(path: str, truth_path: str) -> dict:
    """Render the structural sample PDF and write its ground truth."""
    page_width, page_height = A4
    x0 = 30.0
    top = 790.0
    col_widths = [88.0] * 6
    n_rows = 9
    n_cols = 6
    row_height = 72.0

    x_edges = [x0]
    for width in col_widths:
        x_edges.append(x_edges[-1] + width)
    y_edges = [top]
    for _ in range(n_rows):
        y_edges.append(y_edges[-1] - row_height)

    # Grid seams that were NOT drawn: (row, col) for vertical boundaries
    # between columns col-1 and col, (row, col) for horizontal boundaries
    # between rows row-1 and row.
    v_missing = set()
    h_missing = set()

    def merge_rect(r0, c0, nrows, ncols):
        for row in range(r0, r0 + nrows):
            for col in range(c0 + 1, c0 + ncols):
                v_missing.add((row, col))
        for row in range(r0 + 1, r0 + nrows):
            for col in range(c0, c0 + ncols):
                h_missing.add((row, col))

    merge_rect(0, 0, 1, 5)  # title spanning five columns
    merge_rect(1, 0, 2, 1)  # row span
    merge_rect(1, 1, 1, 2)  # header column span
    merge_rect(2, 1, 1, 2)  # header column span
    merge_rect(1, 4, 1, 2)  # notes column span
    merge_rect(3, 1, 2, 2)  # 2x2 region
    merge_rect(3, 4, 2, 2)  # 2x2 region with two source positions
    merge_rect(8, 0, 1, 6)  # footer spanning the table

    # L-shaped ambiguity: the right edge of (6, 1) and the bottom edge of
    # (6, 1) are both missing, while the rest of the L is closed.
    v_missing.add((6, 2))
    h_missing.add((7, 1))

    pdf = canvas.Canvas(path, pagesize=A4)
    pdf.setLineWidth(0.8)

    for row in range(n_rows):
        y_top = y_edges[row]
        y_bottom = y_edges[row + 1]
        for col in range(n_cols + 1):
            x = x_edges[col]
            if col in (0, n_cols):
                pdf.line(x, y_bottom, x, y_top)
            elif (row, col) in v_missing:
                continue
            elif (row, col) == (5, 2):
                # Short break: a gap of ~14% at the top of the boundary.
                pdf.line(x, y_bottom, x, y_bottom + 0.86 * (y_top - y_bottom))
            else:
                pdf.line(x, y_bottom, x, y_top)

    for row in range(n_rows + 1):
        y = y_edges[row]
        for col in range(n_cols):
            if row in (0, n_rows):
                pdf.line(x_edges[col], y, x_edges[col + 1], y)
            elif (row, col) in h_missing:
                continue
            else:
                pdf.line(x_edges[col], y, x_edges[col + 1], y)

    pdf.setFont(FONT, FONT_SIZE)

    def text_at(row, col, value):
        cx = (x_edges[col] + x_edges[col + 1]) / 2.0
        cy = (y_edges[row] + y_edges[row + 1]) / 2.0
        pdf.drawCentredString(cx, cy - 3, value)

    def text_in_rect(row, col, nrows, ncols, value):
        cx = (x_edges[col] + x_edges[col + ncols]) / 2.0
        cy = (y_edges[row] + y_edges[row + nrows]) / 2.0
        pdf.drawCentredString(cx, cy - 3, value)

    text_in_rect(0, 0, 1, 5, "Regional Enrollment Summary")
    # (0, 5) is a standalone empty cell inside the grid.
    text_at(1, 0, "Region")
    text_in_rect(1, 1, 1, 2, "Coverage")
    text_in_rect(2, 1, 1, 2, "Lives")
    text_at(1, 3, "Plan")
    text_at(2, 3, "Alpha")
    text_at(2, 4, "10")
    text_at(2, 5, "ok")
    text_in_rect(1, 4, 1, 2, "Notes")
    text_at(3, 0, "North")
    text_in_rect(3, 1, 2, 2, "Combined Block")
    text_at(3, 3, "Beta")
    # The same string is written at two source positions inside the 2x2
    # region anchored at (3, 4).
    dup_y = (y_edges[3] + y_edges[5]) / 2.0 - 3
    pdf.drawString(x_edges[4] + 6, dup_y, "DUP")
    pdf.drawRightString(x_edges[6] - 6, dup_y, "DUP")
    text_at(4, 0, "Mid")
    text_at(4, 3, "Gamma")
    text_at(5, 0, "Break")
    text_at(5, 1, "Left")
    text_at(5, 2, "Right")
    text_at(5, 4, "Tail")
    # (5, 3) and (5, 5) are standalone empty cells.
    text_at(6, 0, "Conflict")
    text_at(6, 1, "L-A")
    text_at(6, 2, "L-B")
    text_at(6, 3, "x")
    text_at(6, 4, "y")
    text_at(6, 5, "z")
    text_at(7, 0, "South")
    text_at(7, 1, "L-C")
    text_at(7, 2, "L-D")
    text_at(7, 3, "p")
    text_at(7, 4, "q")
    text_at(7, 5, "r")
    text_in_rect(8, 0, 1, 6, "Source: generated sample")

    pdf.showPage()
    pdf.save()

    truth = {
        "generator": "tests/continuation/generate_structure_pdf.py",
        "page_size": [page_width, page_height],
        "shape": [n_rows, n_cols],
        "x_edges": x_edges,
        "y_edges": y_edges,
        "regions": [
            {
                "row": 0,
                "col": 0,
                "rowspan": 1,
                "colspan": 5,
                "text": "Regional Enrollment Summary",
                "sources": 1,
                "kind": "merged",
            },
            {
                "row": 0,
                "col": 5,
                "rowspan": 1,
                "colspan": 1,
                "text": "",
                "sources": 0,
                "kind": "empty",
            },
            {
                "row": 1,
                "col": 0,
                "rowspan": 2,
                "colspan": 1,
                "text": "Region",
                "sources": 1,
                "kind": "merged",
            },
            {
                "row": 1,
                "col": 1,
                "rowspan": 1,
                "colspan": 2,
                "text": "Coverage",
                "sources": 1,
                "kind": "merged",
            },
            {
                "row": 1,
                "col": 3,
                "rowspan": 1,
                "colspan": 1,
                "text": "Plan",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 1,
                "col": 4,
                "rowspan": 1,
                "colspan": 2,
                "text": "Notes",
                "sources": 1,
                "kind": "merged",
            },
            {
                "row": 2,
                "col": 1,
                "rowspan": 1,
                "colspan": 2,
                "text": "Lives",
                "sources": 1,
                "kind": "merged",
            },
            {
                "row": 2,
                "col": 3,
                "rowspan": 1,
                "colspan": 1,
                "text": "Alpha",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 2,
                "col": 4,
                "rowspan": 1,
                "colspan": 1,
                "text": "10",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 2,
                "col": 5,
                "rowspan": 1,
                "colspan": 1,
                "text": "ok",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 3,
                "col": 0,
                "rowspan": 1,
                "colspan": 1,
                "text": "North",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 3,
                "col": 1,
                "rowspan": 2,
                "colspan": 2,
                "text": "Combined Block",
                "sources": 1,
                "kind": "merged",
            },
            {
                "row": 3,
                "col": 3,
                "rowspan": 1,
                "colspan": 1,
                "text": "Beta",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 3,
                "col": 4,
                "rowspan": 2,
                "colspan": 2,
                "text": "DUP",
                "sources": 2,
                "kind": "merged",
            },
            {
                "row": 4,
                "col": 0,
                "rowspan": 1,
                "colspan": 1,
                "text": "Mid",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 4,
                "col": 3,
                "rowspan": 1,
                "colspan": 1,
                "text": "Gamma",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 5,
                "col": 0,
                "rowspan": 1,
                "colspan": 1,
                "text": "Break",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 5,
                "col": 1,
                "rowspan": 1,
                "colspan": 1,
                "text": "Left",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 5,
                "col": 2,
                "rowspan": 1,
                "colspan": 1,
                "text": "Right",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 5,
                "col": 3,
                "rowspan": 1,
                "colspan": 1,
                "text": "",
                "sources": 0,
                "kind": "empty",
            },
            {
                "row": 5,
                "col": 4,
                "rowspan": 1,
                "colspan": 1,
                "text": "Tail",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 5,
                "col": 5,
                "rowspan": 1,
                "colspan": 1,
                "text": "",
                "sources": 0,
                "kind": "empty",
            },
            {
                "row": 6,
                "col": 0,
                "rowspan": 1,
                "colspan": 1,
                "text": "Conflict",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 6,
                "col": 1,
                "rowspan": 1,
                "colspan": 1,
                "text": "L-A",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 6,
                "col": 2,
                "rowspan": 1,
                "colspan": 1,
                "text": "L-B",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 6,
                "col": 3,
                "rowspan": 1,
                "colspan": 1,
                "text": "x",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 6,
                "col": 4,
                "rowspan": 1,
                "colspan": 1,
                "text": "y",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 6,
                "col": 5,
                "rowspan": 1,
                "colspan": 1,
                "text": "z",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 7,
                "col": 0,
                "rowspan": 1,
                "colspan": 1,
                "text": "South",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 7,
                "col": 1,
                "rowspan": 1,
                "colspan": 1,
                "text": "L-C",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 7,
                "col": 2,
                "rowspan": 1,
                "colspan": 1,
                "text": "L-D",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 7,
                "col": 3,
                "rowspan": 1,
                "colspan": 1,
                "text": "p",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 7,
                "col": 4,
                "rowspan": 1,
                "colspan": 1,
                "text": "q",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 7,
                "col": 5,
                "rowspan": 1,
                "colspan": 1,
                "text": "r",
                "sources": 1,
                "kind": "cell",
            },
            {
                "row": 8,
                "col": 0,
                "rowspan": 1,
                "colspan": 6,
                "text": "Source: generated sample",
                "sources": 1,
                "kind": "merged",
            },
        ],
        "expected_diagnostics": {
            "partial_seam": [
                {
                    "orientation": "vertical",
                    "row": 5,
                    "col": 2,
                    "between": [[5, 1], [5, 2]],
                }
            ],
            "non_rectangular_component": {
                "positions": [[6, 1], [6, 2], [7, 1]],
                "bounding_box": [6, 1, 7, 2],
            },
        },
        "short_break": {"row": 5, "left_col": 1, "right_col": 2},
        "duplicate_region": {
            "row": 3,
            "col": 4,
            "rowspan": 2,
            "colspan": 2,
            "text": "DUP",
        },
    }

    with open(truth_path, "w", encoding="utf-8") as handle:
        json.dump(truth, handle, ensure_ascii=False, indent=2)
    return truth


def main(argv: list[str]) -> int:
    """Write the PDF and truth JSON into the given directory."""
    if len(argv) != 2:
        print("usage: generate_structure_pdf.py OUTPUT_DIR", file=sys.stderr)
        return 2
    outdir = argv[1]
    os.makedirs(outdir, exist_ok=True)
    build(
        os.path.join(outdir, "structure_sample.pdf"),
        os.path.join(outdir, "structure_sample_truth.json"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
