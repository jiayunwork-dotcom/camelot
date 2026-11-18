"""Fixtures for cross-page continuation tests.

The PDFs and their ground truth are generated independently by
``generate_pdfs.py`` (reportlab only; it never imports camelot).  Tests run
the real :func:`camelot.read_pdf` against both stream and lattice flavors.
"""

from __future__ import annotations

import json
import os

import pytest

from .generate_pdfs import build_ambiguity_report
from .generate_pdfs import build_report

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "..", "generated")


def _ensure_pdfs():
    os.makedirs(GENERATED_DIR, exist_ok=True)
    paths = {}
    for flavor in ("stream", "lattice"):
        pdf_path = os.path.join(GENERATED_DIR, f"continuation_{flavor}.pdf")
        truth_path = os.path.join(
            GENERATED_DIR, f"continuation_{flavor}_truth.json"
        )
        if not (os.path.exists(pdf_path) and os.path.exists(truth_path)):
            truth = build_report(pdf_path, flavor)
            with open(truth_path, "w", encoding="utf-8") as fh:
                json.dump(truth, fh, ensure_ascii=False, indent=2)
        amb_pdf = os.path.join(GENERATED_DIR, f"ambiguity_{flavor}.pdf")
        amb_truth = os.path.join(GENERATED_DIR, f"ambiguity_{flavor}_truth.json")
        if not (os.path.exists(amb_pdf) and os.path.exists(amb_truth)):
            truth = build_ambiguity_report(amb_pdf, flavor)
            with open(amb_truth, "w", encoding="utf-8") as fh:
                json.dump(truth, fh, ensure_ascii=False, indent=2)
        paths[flavor] = {
            "report_pdf": pdf_path,
            "report_truth": truth_path,
            "ambiguity_pdf": amb_pdf,
            "ambiguity_truth": amb_truth,
        }
    return paths


@pytest.fixture(scope="session")
def generated_paths():
    """Generate (once) and return paths of the independent test PDFs."""
    return _ensure_pdfs()


@pytest.fixture
def report_truth(generated_paths, flavor):
    with open(generated_paths[flavor]["report_truth"], encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def ambiguity_truth(generated_paths, flavor):
    with open(
        generated_paths[flavor]["ambiguity_truth"], encoding="utf-8"
    ) as fh:
        return json.load(fh)


@pytest.fixture(params=["stream", "lattice"])
def flavor(request):
    return request.param


def normalize_matrix(rows):
    """Normalize only for comparison: never mutate retained data values."""
    return [[str(c).replace("\n", "") for c in row] for row in rows]
