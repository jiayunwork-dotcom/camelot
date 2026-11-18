"""Tests for the structural (merged cell) view of lattice tables.

The sample PDF is generated independently by
``tests/continuation/generate_structure_pdf.py`` (reportlab only, no camelot)
and committed together with its ground-truth JSON, so these tests never
depend on reportlab at run time.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

import camelot
from camelot.core import Table
from camelot.structure import StructureUnavailableError


SAMPLE = "structure_sample.pdf"
TRUTH = "structure_sample_truth.json"
FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")


@pytest.fixture(scope="module")
def truth():
    with open(os.path.join(FILES_DIR, TRUTH), encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def sample_path():
    return os.path.join(FILES_DIR, SAMPLE)


@pytest.fixture(scope="module")
def continuation_paths():
    """Generated multi-page continuation PDFs (shared with tests/continuation)."""
    from .continuation.conftest import _ensure_pdfs

    return _ensure_pdfs()


@pytest.fixture(scope="module")
def table(sample_path):
    return camelot.read_pdf(sample_path)[0]


def _normalise(text):
    return "".join(str(text).split())


def test_view_covers_every_grid_position(table, truth):
    structure = table.structure
    assert structure.shape == tuple(truth["shape"])
    assert structure.page == table.page == 1
    assert structure.order == table.order == 1
    covered = {}
    for cell in structure:
        for position in cell.positions:
            assert position not in covered, f"{position} covered twice"
            covered[position] = cell
    expected = {
        (row, col)
        for row in range(truth["shape"][0])
        for col in range(truth["shape"][1])
    }
    assert set(covered) == expected


def test_regions_match_generator_truth(table, truth):
    structure = table.structure
    by_anchor = {(cell.row, cell.col): cell for cell in structure}
    for region in truth["regions"]:
        anchor = (region["row"], region["col"])
        cell = by_anchor[anchor]
        assert (cell.rowspan, cell.colspan) == (
            region["rowspan"],
            region["colspan"],
        ), region
        assert _normalise(cell.text) == _normalise(region["text"]), region
        assert len(cell.sources) == region["sources"], region
        assert cell.page == table.page and cell.order == table.order


def test_spanning_cells_have_geometry_and_anchor(table, truth):
    merged = table.structure.merged_cells
    assert len(merged) == sum(
        1 for region in truth["regions"] if region["kind"] == "merged"
    )
    horizontal = [c for c in merged if c.rowspan == 1 and c.colspan == 5]
    assert [c.anchor for c in horizontal] == [(0, 0)]
    assert horizontal[0].bbox == (
        table.cols[0][0],
        table.rows[0][1],
        table.cols[4][1],
        table.rows[0][0],
    )
    vertical = [c for c in merged if c.rowspan == 2 and c.colspan == 1]
    assert [c.anchor for c in vertical] == [(1, 0)]
    both = [c for c in merged if c.rowspan == 2 and c.colspan == 2]
    assert sorted(c.anchor for c in both) == [(3, 1), (3, 4)]
    assert both[0].bbox == (
        min(table.cols[1]),
        table.rows[4][1],
        max(table.cols[2]),
        table.rows[3][0],
    )


def test_standalone_empty_cells(table, truth):
    anchors = {
        (region["row"], region["col"])
        for region in truth["regions"]
        if region["kind"] == "empty"
    }
    assert anchors == {(0, 5), (5, 3), (5, 5)}
    for anchor in anchors:
        cell = table.structure.cell_at(*anchor)
        assert cell.anchor == anchor
        assert (cell.rowspan, cell.colspan) == (1, 1)
        assert cell.text.strip() == ""
        assert cell.sources == []


def test_duplicate_text_keeps_two_sources_one_value(table, truth):
    duplicate = truth["duplicate_region"]
    cell = table.structure.cell_at(duplicate["row"], duplicate["col"])
    assert (cell.rowspan, cell.colspan) == (
        duplicate["rowspan"],
        duplicate["colspan"],
    )
    assert cell.text.strip() == duplicate["text"]
    assert sorted((source.row, source.col) for source in cell.sources) == [
        (4, 4),
        (4, 5),
    ]
    assert all(source.text.strip() == "DUP" for source in cell.sources)


def test_short_break_is_not_merged(table, truth):
    break_info = truth["short_break"]
    left = table.structure.cell_at(break_info["row"], break_info["left_col"])
    right = table.structure.cell_at(break_info["row"], break_info["right_col"])
    assert left.anchor != right.anchor
    assert left.text.strip() == "Left"
    assert right.text.strip() == "Right"
    partial = [
        d
        for d in table.structure.diagnostics
        if d.kind == "partial_seam"
    ]
    assert len(partial) == 1
    diagnostic = partial[0]
    assert diagnostic.page == table.page
    assert diagnostic.order == table.order
    assert diagnostic.positions == [
        tuple(position)
        for position in truth["expected_diagnostics"]["partial_seam"][0]["between"]
    ]
    assert 0.0 < diagnostic.details["coverage"] < 1.0


def test_l_shape_is_reported_and_text_retained(table, truth):
    expected = truth["expected_diagnostics"]["non_rectangular_component"]
    diagnostics = [
        d
        for d in table.structure.diagnostics
        if d.kind == "non_rectangular_component"
    ]
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.page == table.page
    assert diagnostic.order == table.order
    assert sorted(diagnostic.positions) == [
        tuple(position) for position in expected["positions"]
    ]
    assert diagnostic.details["bounding_box"] == expected["bounding_box"]
    for row, col, text in [
        (6, 1, "L-A"),
        (6, 2, "L-B"),
        (7, 1, "L-C"),
        (7, 2, "L-D"),
    ]:
        cell = table.structure.cell_at(row, col)
        assert (cell.rowspan, cell.colspan) == (1, 1)
        assert cell.text.strip() == text


def test_flat_matrix_and_report_are_unchanged(table, sample_path):
    # Default lattice behaviour must stay on the flat matrix.
    fresh = camelot.read_pdf(sample_path)[0]
    pd.testing.assert_frame_equal(table.df, fresh.df)
    assert table.shape == table.df.shape
    assert table.parsing_report["page"] == 1
    assert table.parsing_report["order"] == 1
    assert "accuracy" in table.parsing_report
    # The flat matrix still carries the text in the cells chosen by the
    # lattice text assignment (including the shifted ones).
    flat = table.data
    assert flat[0][0] == "Regional Enrollment Summary"
    assert flat[5][3] == ""


def _structure_snapshot(table):
    structure = table.structure
    return (
        [
            (
                cell.row,
                cell.col,
                cell.rowspan,
                cell.colspan,
                cell.text,
                tuple((s.row, s.col, s.text) for s in cell.sources),
                cell.bbox,
            )
            for cell in structure
        ],
        [(d.kind, tuple(d.positions)) for d in structure.diagnostics],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"shift_text": [""]},
        {"shift_text": ["r", "b"]},
        {"copy_text": "v"},
        {"copy_text": ["h", "v"]},
        {"shift_text": ["r", "b"], "copy_text": "h"},
    ],
)
def test_structure_is_independent_of_shift_and_copy(sample_path, table, kwargs):
    other = camelot.read_pdf(sample_path, **kwargs)[0]
    assert _structure_snapshot(other) == _structure_snapshot(table)


def test_shift_and_copy_still_change_the_flat_matrix(sample_path):
    default = camelot.read_pdf(sample_path)[0]
    shifted = camelot.read_pdf(sample_path, shift_text=["r", "b"])[0]
    assert not default.df.equals(shifted.df)
    copied = camelot.read_pdf(sample_path, copy_text="v")[0]
    assert not default.df.equals(copied.df)
    assert default.df.equals(
        camelot.read_pdf(sample_path, shift_text=["l", "t"])[0].df
    )


def _parse_html_grid(text):
    """Expand an HTML table into a grid of (text, rowspan, colspan)."""
    from html.parser import HTMLParser

    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.current = None
            self.cell = None

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "tr":
                self.current = []
                self.rows.append(self.current)
            elif tag == "td" and self.current is not None:
                self.cell = {
                    "text": "",
                    "rowspan": int(attrs.get("rowspan", 1)),
                    "colspan": int(attrs.get("colspan", 1)),
                }
                self.current.append(self.cell)

        def handle_endtag(self, tag):
            if tag == "td":
                self.cell = None

        def handle_data(self, data):
            if self.cell is not None:
                self.cell["text"] += data

    parser = TableParser()
    parser.feed(text)
    occupancy = {}
    for row_idx, row in enumerate(parser.rows):
        col_idx = 0
        for cell in row:
            while (row_idx, col_idx) in occupancy:
                col_idx += 1
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    occupancy[(row_idx + dr, col_idx + dc)] = cell
            col_idx += cell["colspan"]
    return parser.rows, occupancy


def test_html_export_with_merges(table, tmp_path):
    path = tmp_path / "structure.html"
    table.to_html(str(path), keep_merges=True)
    text = path.read_text(encoding="utf-8")
    merged = [c for c in table.structure if c.is_merged]
    for cell in merged:
        tag = "<td"
        if cell.rowspan > 1:
            tag += f' rowspan="{cell.rowspan}"'
        if cell.colspan > 1:
            tag += f' colspan="{cell.colspan}"'
        tag += ">"
        assert tag in text
    # One retained copy of each anchor text; covered positions are not
    # written again.
    assert text.count("Regional Enrollment Summary") == 1
    assert text.count("Combined Block") == 1
    assert text.count("DUP") == 1
    assert "<td>L-A</td>" in text
    assert "<td>L-B</td>" in text
    assert "<td></td>" in text


def test_html_read_back_reproduces_the_structure(table, tmp_path):
    path = tmp_path / "structure.html"
    table.to_html(str(path), keep_merges=True)
    _, grid = _parse_html_grid(path.read_text(encoding="utf-8"))
    assert set(grid) == {
        (r, c)
        for r in range(table.shape[0])
        for c in range(table.shape[1])
    }
    for position, parsed in grid.items():
        cell = table.structure.cell_at(*position)
        assert (parsed["rowspan"], parsed["colspan"]) == (
            cell.rowspan,
            cell.colspan,
        )
        assert _normalise(parsed["text"]) == _normalise(cell.text)


def test_default_exports_are_flat(table, tmp_path):
    path = tmp_path / "flat.html"
    table.to_html(str(path))
    text = path.read_text(encoding="utf-8")
    assert "rowspan" not in text
    assert "colspan" not in text
    assert text.count("<td>") == table.shape[0] * table.shape[1]


def test_excel_export_with_merges(table, tmp_path):
    from openpyxl import load_workbook

    path = tmp_path / "structure.xlsx"
    table.to_excel(str(path), keep_merges=True)
    workbook = load_workbook(str(path))
    sheet = workbook[f"page-{table.page}-table-{table.order}"]
    ranges = {str(rng) for rng in sheet.merged_cells.ranges}
    expected = set()
    for cell in table.structure:
        if not cell.is_merged:
            continue
        from openpyxl.utils import get_column_letter

        start = f"{get_column_letter(cell.col + 1)}{cell.row + 1}"
        end = (
            f"{get_column_letter(cell.col + cell.colspan)}"
            f"{cell.row + cell.rowspan}"
        )
        expected.add(f"{start}:{end}")
        value = sheet.cell(row=cell.row + 1, column=cell.col + 1).value
        assert (_normalise(value) if value is not None else "") == _normalise(
            cell.text
        )
    assert ranges == expected
    # Strings stay strings on the way back.
    assert sheet["A4"].value == "North"
    assert isinstance(sheet["A4"].value, str)
    assert sheet["B3"].value == "Lives"


def test_excel_export_anchor_text_and_empty_cells(table, tmp_path):
    from openpyxl import load_workbook

    path = tmp_path / "structure.xlsx"
    table.to_excel(str(path), keep_merges=True)
    sheet = load_workbook(str(path)).active
    assert sheet.cell(row=1, column=1).value == "Regional Enrollment Summary"
    assert sheet.cell(row=1, column=5).value is None  # covered by the title
    assert sheet.cell(row=1, column=6).value is None  # standalone empty cell
    assert sheet.cell(row=6, column=4).value is None  # standalone empty cell


def test_default_excel_export_is_flat(table, tmp_path):
    from openpyxl import load_workbook

    path = tmp_path / "flat.xlsx"
    table.to_excel(str(path))
    sheet = load_workbook(str(path)).active
    assert not sheet.merged_cells.ranges
    assert sheet.max_row == table.shape[0] + 1
    assert sheet.max_column == table.shape[1] + 1


def test_export_options_rejected_with_merges(table, tmp_path):
    with pytest.raises(TypeError, match="keep_merges=True"):
        table.to_html(str(tmp_path / "x.html"), keep_merges=True, index=False)
    with pytest.raises(TypeError, match="keep_merges=True"):
        table.to_excel(str(tmp_path / "x.xlsx"), keep_merges=True, index=False)


def test_structure_rejects_other_flavors(foo_pdf):
    for flavor in ("stream", "network", "hybrid"):
        tables = camelot.read_pdf(foo_pdf, flavor=flavor)
        with pytest.raises(StructureUnavailableError, match="flavor='lattice'"):
            tables[0].structure


def test_structure_rejects_table_without_lattice_evidence():
    table = Table([(0.0, 10.0), (10.0, 20.0)], [(20.0, 10.0), (10.0, 0.0)])
    table.set_all_edges()
    table.flavor = "lattice"
    table.page = 1
    table.order = 1
    table.df = pd.DataFrame([["a", "b"], ["c", "d"]])
    with pytest.raises(StructureUnavailableError, match="no lattice parsing"):
        table.structure


def test_structure_rejects_logical_tables(continuation_paths):
    tables = camelot.read_pdf(
        continuation_paths["lattice"]["report_pdf"],
        pages="all",
        flavor="lattice",
        continuation="auto",
    )
    assert tables.logical_tables
    logical = tables.logical_tables[0]
    with pytest.raises(StructureUnavailableError, match="Logical tables"):
        logical.structure
    # Physical fragments still expose the view.
    assert tables[0].structure is not None


def test_structure_cost_scales_with_the_grid():
    """Seams are enumerated once per grid edge, never per rectangle."""
    import time

    n_rows, n_cols = 120, 80
    width, height = 10.0, 10.0
    cols = [(c * width, (c + 1) * width) for c in range(n_cols)]
    rows = [((n_rows - r) * height, (n_rows - r - 1) * height) for r in range(n_rows)]
    table = Table(cols, rows)
    table.set_all_edges()
    table.flavor = "lattice"
    table.page = 1
    table.order = 1
    table._bbox = (0.0, 0.0, n_cols * width, n_rows * height)
    vertical = [
        (c * width, 0.0, c * width, n_rows * height) for c in range(n_cols + 1)
    ]
    horizontal = [
        (0.0, r * height, n_cols * width, r * height) for r in range(n_rows + 1)
    ]
    table._structure_evidence = {
        "joint_tol": 2.0,
        "vertical_segments": vertical,
        "horizontal_segments": horizontal,
    }
    table._raw_text_cells = []
    table.df = pd.DataFrame(
        [[""] * n_cols for _ in range(n_rows)]
    )
    table.shape = table.df.shape

    start = time.perf_counter()
    structure = table.structure
    elapsed = time.perf_counter() - start

    assert structure.shape == (n_rows, n_cols)
    assert structure.merged_cells == []
    assert len(structure) == n_rows * n_cols
    # One vertical seam per interior column boundary per row, plus one
    # horizontal seam per interior row boundary per column.
    assert len(structure.seams) == n_rows * (n_cols - 1) + (n_rows - 1) * n_cols
    # Linear work: a dense 9,600-cell grid finishes comfortably below the
    # time a rectangle enumeration would need.
    assert elapsed < 5.0


def test_structure_is_cached_and_read_only(table):
    first = table.structure
    second = table.structure
    assert first is second
    snapshot = table.df.copy()
    _ = first.to_dict()
    pd.testing.assert_frame_equal(snapshot, table.df)
    payload = first.to_dict()
    assert payload["shape"] == list(table.shape)
    assert set(payload) == {"page", "order", "shape", "cells", "diagnostics", "seams"}
    kinds = {d["kind"] for d in payload["diagnostics"]}
    assert kinds == {"partial_seam", "non_rectangular_component"}
    for entry in payload["diagnostics"]:
        assert entry["page"] == table.page
        assert entry["order"] == table.order
        assert entry["message"]
        assert entry["grid_positions"]
