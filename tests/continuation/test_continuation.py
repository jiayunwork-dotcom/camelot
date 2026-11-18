"""End-to-end tests for ``continuation='auto'``.

Every test reads an independently generated PDF through the real
:func:`camelot.read_pdf`, compares the logical matrices / fragment
membership / row provenance against generator-produced ground truth, and
also checks that the default physical extraction is never modified.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile

import pandas as pd
import pytest

import camelot

from .conftest import normalize_matrix


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_invalid_continuation_value(generated_paths, flavor):
    with pytest.raises(ValueError, match="continuation must be None or 'auto'"):
        camelot.read_pdf(
            generated_paths[flavor]["report_pdf"],
            flavor=flavor,
            continuation="yes",
        )


@pytest.mark.parametrize("bad_flavor", ["network", "hybrid"])
def test_continuation_rejects_other_flavors(generated_paths, bad_flavor):
    with pytest.raises(ValueError, match="only supported"):
        camelot.read_pdf(
            generated_paths["stream"]["report_pdf"],
            flavor=bad_flavor,
            continuation="auto",
        )


def test_default_mode_has_no_logical_tables(generated_paths, flavor):
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"], flavor=flavor, pages="1"
    )
    assert tables.logical_tables is None
    assert tables.continuation_diagnostics is None
    assert tables[0].fragment_context is None


# ---------------------------------------------------------------------------
# Logical grouping and matrix truth
# ---------------------------------------------------------------------------


def test_logical_matrices_match_truth(generated_paths, flavor, report_truth):
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="all",
        flavor=flavor,
        continuation="auto",
    )
    got = sorted(
        (
            normalize_matrix(lt.df.values.tolist())
            for lt in tables.logical_tables
        ),
        key=lambda rows: rows[0],
    )
    expected = sorted(
        (meta["full_matrix"] for meta in report_truth["logicals"].values()),
        key=lambda rows: rows[0],
    )
    assert got == expected


def test_three_page_two_row_header_table(generated_paths, flavor, report_truth):
    """Logical A spans three pages with a wrapped, shifted header."""
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="1-3",
        flavor=flavor,
        continuation="auto",
    )
    a = tables.logical_tables[0]
    assert len(a.fragments) == 3
    assert a.df.shape[1] == 3
    # the two logical header rows are present once at the top
    assert normalize_matrix(a.df.values.tolist()[:2]) == [
        ["Region A", "Metric A", "Unit A"],
        ["Name A", "Value A", "Scale A"],
    ]
    # repeated headers on pages 2 and 3 were removed and audited
    assert len(a.removed_headers) >= 2
    for record in a.removed_headers:
        assert {"logical_id", "page", "order", "physical_row", "values"} <= set(
            record
        )


def test_body_header_word_and_empty_leading_cell_kept(
    generated_paths, flavor, report_truth
):
    """A body value equal to a header word must never be removed."""
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="1",
        flavor=flavor,
        continuation="auto",
    )
    values = normalize_matrix(tables.logical_tables[0].df.values.tolist())
    # A-r2-c1 is exactly the header word "Value A" and is retained as data
    assert any("Value A" in row and row[0].startswith("A-r2") for row in values)
    # row A-r1 has an empty first cell, preserved
    empty_first = [row for row in values if row[0] == "" and "A-r1" in row[1]]
    assert empty_first


def test_two_side_by_side_tables_continue(generated_paths, flavor):
    """Two same-page independent tables each continue on the next page."""
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="4,5",
        flavor=flavor,
        continuation="auto",
    )
    logical = tables.logical_tables
    assert len(logical) == 2
    for lt in logical:
        assert len(lt.fragments) == 2
        assert lt.df.shape == (9, 3)
    # the two logical tables have distinct headers
    headers = {tuple(lt.df.values.tolist()[0]) for lt in logical}
    assert headers == {("Item B", "Qty B", "Price B"), ("Item C", "Qty C", "Price C")}


def test_same_header_distinct_table_numbers_not_merged(
    generated_paths, flavor
):
    """Tables D and E look identical but have different table numbers."""
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="6,7",
        flavor=flavor,
        continuation="auto",
    )
    assert len(tables.logical_tables) == 2
    blocker = [
        d
        for d in tables.continuation_diagnostics
        if d.get("type") == "blocked_edge"
        and any("table number" in r for r in d["edge"]["reasons"])
    ]
    assert blocker


def test_chapter_switch_blocks_merge(generated_paths, flavor):
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="8,9",
        flavor=flavor,
        continuation="auto",
    )
    assert len(tables.logical_tables) == 2
    section = [
        d
        for d in tables.continuation_diagnostics
        if d.get("type") == "blocked_edge"
        and any("section heading" in r for r in d["edge"]["reasons"])
    ]
    assert section


def test_missing_intermediate_page_blocks_merge(generated_paths, flavor):
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="10,11,13",
        flavor=flavor,
        continuation="auto",
    )
    fragments = [lt.fragments for lt in tables.logical_tables]
    assert fragments == [[(10, 1, 0), (11, 1, 0)], [(13, 1, 0)]]
    blocked = [
        d
        for d in tables.continuation_diagnostics
        if d.get("type") == "blocked_edge"
        and any("missing intermediate pages" in r for r in d["edge"]["reasons"])
    ]
    assert blocked


def test_empty_column_continuation_keeps_canonical_layout(
    generated_paths, flavor
):
    """The whole empty column / interrupted line scenario (logical H)."""
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="10,11",
        flavor=flavor,
        continuation="auto",
    )
    h = tables.logical_tables[0]
    assert h.df.shape == (9, 3)
    # continuation rows keep an empty middle column instead of shifting data
    page11_rows = [
        rs for rs in h.row_sources if rs.page == 11 and rs.cells
    ]
    for rs in page11_rows:
        columns = {cell["column"] for cell in rs.cells}
        assert 1 not in columns  # middle column empty on the continuation
    # a column-layout repair diagnostic must explain the restored layout
    repairs = [d for d in h.diagnostics if d["type"] == "column_layout_repaired"]
    assert repairs
    assert repairs[0]["canonical_columns"] == 3


# ---------------------------------------------------------------------------
# Ambiguity
# ---------------------------------------------------------------------------


def test_ambiguity_is_reported_not_arbitrarily_resolved(
    generated_paths, flavor, ambiguity_truth
):
    tables = camelot.read_pdf(
        generated_paths[flavor]["ambiguity_pdf"],
        pages="all",
        flavor=flavor,
        continuation="auto",
    )
    # no merge: every fragment stays its own single-fragment logical table
    assert len(tables.logical_tables) == 3
    assert all(len(lt.fragments) == 1 for lt in tables.logical_tables)
    ambiguity = [
        d for d in tables.continuation_diagnostics if d.get("type") == "ambiguity"
    ]
    assert len(ambiguity) == 1
    diag = ambiguity[0]
    assert diag["candidate"] == [2, 1, 0]
    assert len(diag["candidates"]) == 2
    # competing predecessors with equal strength and explicit evidence
    strengths = {c["strength"] for c in diag["candidates"]}
    assert strengths == {"medium"}
    # all data is preserved (no single-fragment collapse)
    got = sorted(
        (normalize_matrix(lt.df.values.tolist()) for lt in tables.logical_tables),
        key=lambda rows: rows[0],
    )
    expected = sorted(
        (meta["full_matrix"] for meta in ambiguity_truth["logicals"].values()),
        key=lambda rows: rows[0],
    )
    assert got == expected


# ---------------------------------------------------------------------------
# Provenance and physical table integrity
# ---------------------------------------------------------------------------


def test_every_physical_table_is_attributed_once(generated_paths, flavor):
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="all",
        flavor=flavor,
        continuation="auto",
    )
    physical = {(t.page, t.order) for t in tables}
    backed = {
        (f[0], f[1])
        for lt in tables.logical_tables
        for f in lt.fragments
    }
    assert physical == backed
    # fragments never appear in two logical tables
    all_fragments = [
        f for lt in tables.logical_tables for f in lt.fragments
    ]
    assert len(all_fragments) == len(set(all_fragments))


def test_physical_extraction_is_untainted(generated_paths, flavor):
    default_tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="all",
        flavor=flavor,
    )
    auto_tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="all",
        flavor=flavor,
        continuation="auto",
    )
    assert default_tables.n == auto_tables.n
    for default, auto in zip(default_tables, auto_tables):
        assert (
            default.df.fillna("").values.tolist()
            == auto.df.fillna("").values.tolist()
        )
        assert default.parsing_report == auto.parsing_report
        assert (default.page, default.order) == (auto.page, auto.order)
        assert default.cells[0][0].x1 == auto.cells[0][0].x1
        # evidence is attached without altering the physical object's data
        assert auto.fragment_context is not None
        assert auto.fragment_context.page == auto.page


def test_row_sources_point_to_original_page_coordinates(
    generated_paths, flavor
):
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="10,11",
        flavor=flavor,
        continuation="auto",
    )
    h = tables.logical_tables[0]
    assert len(h.row_sources) == h.df.shape[0]
    for rs in h.row_sources:
        assert rs.page in (10, 11)
        assert rs.order >= 1
        for cell in rs.cells:
            x0, y0, x1, y1 = cell["bbox"]
            # original page coordinates: positive, left-bottom origin
            assert 0 <= x0 < x1 and 0 <= y0 < y1
            assert y1 <= h.row_sources[0].cells[0]["bbox"][3] + 2000
            assert cell["text"]  # non-empty cells only


def test_fragment_context_exposes_page_evidence(generated_paths, flavor):
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="1",
        flavor=flavor,
        continuation="auto",
    )
    ctx = tables[0].fragment_context
    assert ctx.page == 1
    assert ctx.table_id == "1"
    assert ctx.n_cols == 3
    assert ctx.header_depth >= 1
    assert ctx.col_anchors == sorted(ctx.col_anchors)
    # JSON serializable evidence (needed for diagnostics/manifest)
    json.dumps(ctx.to_dict())


def test_repeated_calls_are_idempotent(generated_paths, flavor):
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="10,11",
        flavor=flavor,
        continuation="auto",
    )
    physical_before = [
        t.df.fillna("").values.tolist() for t in tables
    ]
    first = [
        normalize_matrix(lt.df.values.tolist()) for lt in tables.logical_tables
    ]
    second = [
        normalize_matrix(lt.df.values.tolist()) for lt in tables.logical_tables
    ]
    physical_after = [t.df.fillna("").values.tolist() for t in tables]
    assert first == second
    assert physical_before == physical_after


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_logical_csv_export_and_manifest_readback(generated_paths, flavor):
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="all",
        flavor=flavor,
        continuation="auto",
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.csv")
        tables.export(out, f="csv", logical=True)
        with open(
            os.path.join(tmp, "report-logical-manifest.json"), encoding="utf-8"
        ) as fh:
            manifest = json.load(fh)
        assert len(manifest["logical_tables"]) == len(tables.logical_tables)
        # unique data file names even for same start page side tables
        names = [e["data_file"] for e in manifest["logical_tables"]]
        assert len(names) == len(set(names))
        for entry in manifest["logical_tables"]:
            back = pd.read_csv(
                os.path.join(tmp, entry["data_file"]),
                header=None,
                dtype=str,
                keep_default_na=False,
            ).values.tolist()
            lt = next(
                x
                for x in tables.logical_tables
                if x.logical_id == entry["logical_id"]
            )
            assert back == lt.df.fillna("").astype(str).values.tolist()
            logical_rows = {rs["logical_row"] for rs in entry["row_sources"]}
            assert logical_rows == set(range(lt.df.shape[0]))


def test_logical_json_export_readback(generated_paths, flavor):
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="all",
        flavor=flavor,
        continuation="auto",
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.json")
        tables.export(out, f="json", logical=True)
        manifest = json.load(
            open(os.path.join(tmp, "report-logical-manifest.json"))
        )
        for entry in manifest["logical_tables"]:
            data = json.load(open(os.path.join(tmp, entry["data_file"])))
            lt = next(
                x
                for x in tables.logical_tables
                if x.logical_id == entry["logical_id"]
            )
            assert data == lt.df.fillna("").astype(str).values.tolist()


def test_logical_zip_manifest_references_resolve(generated_paths, flavor):
    tables = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="all",
        flavor=flavor,
        continuation="auto",
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.csv")
        tables.export(out, f="csv", logical=True, compress=True)
        zip_path = os.path.join(tmp, "report.zip")
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.namelist()
            manifest = json.loads(zf.read("report-logical-manifest.json"))
            assert all(
                entry["data_file"] in members
                for entry in manifest["logical_tables"]
            )


def test_logical_export_error_branches(generated_paths, flavor, tmp_path):
    auto = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"],
        pages="1",
        flavor=flavor,
        continuation="auto",
    )
    legacy = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"], pages="1", flavor=flavor
    )
    with pytest.raises(ValueError, match="continuation='auto'"):
        legacy.export(str(tmp_path / "x.csv"), logical=True)
    with pytest.raises(ValueError, match="only supports"):
        auto.export(str(tmp_path / "x.xlsx"), f="excel", logical=True)
    with pytest.raises(ValueError, match="only supports"):
        auto.export(str(tmp_path / "x.html"), f="html", logical=True)


def test_physical_export_naming_unchanged(generated_paths, flavor, tmp_path):
    legacy = camelot.read_pdf(
        generated_paths[flavor]["report_pdf"], pages="1", flavor=flavor
    )
    out = str(tmp_path / "legacy.csv")
    legacy.export(out, f="csv")
    produced = [p.name for p in tmp_path.iterdir()]
    assert "legacy-page-1-table-1.csv" in produced
