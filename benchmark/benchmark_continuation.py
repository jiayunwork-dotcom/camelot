"""Scalability benchmark for continuation collection.

Keeps the per-page table size fixed and grows the number of pages from 10 to
40 to 100.  Records wall-clock time and peak memory for both a plain read
and a ``continuation='auto'`` read, and checks that grouping stays linear
(no Cartesian product of pages/cells).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib.pagesizes import A4  # noqa: E402

from tests.continuation.generate_pdfs import _data_rows  # noqa: E402
from tests.continuation.generate_pdfs import ReportBuilder  # noqa: E402


def build_scaled_pdf(path: str, n_pages: int, flavor: str) -> None:
    """Build an n-page report with one fixed-size table per page."""
    builder = ReportBuilder(path, flavor)
    header = [["Region", "Metric", "Unit"]]
    widths = [150, 120, 110]
    for page in range(1, n_pages + 1):
        if page > 1:
            builder.new_page()
        caption = (
            f"Table 1 (continued)" if page > 1 else "Table 1: Long metrics"
        )
        builder.draw_fragment(
            "A",
            header + _data_rows("A", 4, 3, start_idx=(page - 1) * 4),
            widths,
            x=60,
            top_y=builder.page_height - 50,
            header_depth=1,
            caption=caption,
            continued_caption=page > 1,
        )
    builder.save()


def measure(call):
    tracemalloc.start()
    start = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / (1024 * 1024), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    parser.add_argument("--flavor", default="stream", choices=["stream", "lattice"])
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    import camelot

    records = []
    for n_pages in (10, 40, 100):
        pdf = os.path.join(args.outdir, f"scale_{n_pages}_{args.flavor}.pdf")
        build_scaled_pdf(pdf, n_pages, args.flavor)
        plain_t, plain_m, plain = measure(
            lambda: camelot.read_pdf(pdf, pages="all", flavor=args.flavor)
        )
        auto_t, auto_m, auto = measure(
            lambda: camelot.read_pdf(
                pdf, pages="all", flavor=args.flavor, continuation="auto"
            )
        )
        # one logical table spanning every page confirms linear chaining
        assert len(auto.logical_tables) == 1
        assert len(auto.logical_tables[0].fragments) == n_pages
        records.append(
            {
                "pages": n_pages,
                "physical_tables": plain.n,
                "plain_seconds": round(plain_t, 3),
                "plain_peak_mib": round(plain_m, 2),
                "continuation_seconds": round(auto_t, 3),
                "continuation_peak_mib": round(auto_m, 2),
                "logical_tables": len(auto.logical_tables),
            }
        )
    out = os.path.join(args.outdir, f"benchmark_{args.flavor}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
