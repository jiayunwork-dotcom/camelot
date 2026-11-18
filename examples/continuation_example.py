"""Collect a detail table that continues across many PDF pages.

Run with:

    python continuation_example.py path/to/report.pdf

The script reads the whole document with continuation collection enabled,
prints the logical grouping and the row provenance, and exports the logical
tables plus a sources/diagnostics manifest to CSV.
"""

import os
import sys

import camelot


def main(path: str) -> None:
    tables = camelot.read_pdf(
        path,
        pages="all",
        flavor="stream",
        continuation="auto",
    )

    print(f"physical tables: {tables.n}")
    print(f"logical tables: {len(tables.logical_tables)}")

    for logical in tables.logical_tables:
        print(
            f"\n{logical.logical_id}: {logical.df.shape[0]} rows x "
            f"{logical.df.shape[1]} cols from fragments {logical.fragments}"
        )
        # Where does the first data row of each fragment come from?
        for source in logical.row_sources[:1]:
            print(
                "  first row source -> page",
                source.page,
                "table",
                source.order,
                "physical row",
                source.physical_row,
            )
        if logical.removed_headers:
            print(f"  removed {len(logical.removed_headers)} repeated headers")

    out_dir = os.path.splitext(os.path.basename(path))[0] + "_logical"
    os.makedirs(out_dir, exist_ok=True)
    tables.export(
        os.path.join(out_dir, "logical.csv"),
        f="csv",
        logical=True,
    )
    print(f"\nwrote logical CSV files and manifest to {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python continuation_example.py <pdf-path>")
        raise SystemExit(1)
    main(sys.argv[1])
