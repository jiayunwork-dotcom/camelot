"""Serial vs parallel equivalence check for continuation collection."""

import json

import camelot


def normalize(rows):
    return [[str(c).replace("\n", "") for c in r] for r in rows]


def main():
    for flavor in ("stream", "lattice"):
        pdf = f"tests/generated/continuation_{flavor}.pdf"
        serial = camelot.read_pdf(
            pdf, pages="all", flavor=flavor, continuation="auto", parallel=False
        )
        parallel = camelot.read_pdf(
            pdf, pages="all", flavor=flavor, continuation="auto", parallel=True
        )
        g_s = [[list(f) for f in lt.fragments] for lt in serial.logical_tables]
        g_p = [[list(f) for f in lt.fragments] for lt in parallel.logical_tables]
        assert g_s == g_p, f"{flavor}: groups differ"
        for a, b in zip(serial.logical_tables, parallel.logical_tables):
            assert normalize(a.df.values.tolist()) == normalize(
                b.df.values.tolist()
            ), f"{flavor}: matrices differ"
            assert [rs.to_dict() for rs in a.row_sources] == [
                rs.to_dict() for rs in b.row_sources
            ], f"{flavor}: row sources differ"
        print(flavor, "serial==parallel OK")


if __name__ == "__main__":
    main()
