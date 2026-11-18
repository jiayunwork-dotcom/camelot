Continuation Scaling Notes
==========================

Setup
-----

Each benchmark PDF contains one fixed-size table (3 columns, 4 body rows per
page plus a header) repeated over N pages, so the per-page work is constant
and only the page count changes.  Measurements use
`benchmark/benchmark_continuation.py` (wall-clock time via `time.perf_counter`
and peak allocations via `tracemalloc`) on Python 3.11.

## Observed growth (10 / 40 / 100 pages)

### stream

.. list-table::
   :header-rows: 1

   * - pages
     - physical tables
     - plain time (s)
     - continuation time (s)
     - plain peak (MiB)
     - continuation peak (MiB)
   * - 10
     - 10
     - 0.30
     - 0.34
     - 1.46
     - 1.94
   * - 40
     - 40
     - 1.56
     - 1.59
     - 6.77
     - 6.13
   * - 100
     - 100
     - 6.66
     - 5.88
     - 16.08
     - 17.27

### lattice

.. list-table::
   :header-rows: 1

   * - pages
     - physical tables
     - plain time (s)
     - continuation time (s)
     - plain peak (MiB)
     - continuation peak (MiB)
   * - 10
     - 10
     - 2.92
     - 2.82
     - 291.9
     - 292.0
   * - 40
     - 40
     - 12.03
     - 12.00
     - 1044.5
     - 1044.1
   * - 100
     - 100
     - 33.43
     - 31.39
     - 2546.8
     - 2547.7

## Interpretation

* **Time is linear in the number of pages.** Going from 10 to 100 pages (a
  10x increase) multiplies stream time by roughly 10x and lattice time by
  roughly 11x.  The continuation grouping itself adds no visible overhead on
  top of the single-page parsing.
* **Memory is linear in the number of fragments.**  Each page contributes one
  fragment's evidence, original page text lines and data matrix; nothing is
  held as a page-by-page or cell-by-cell product.  Lattice's much larger
  footprint comes from the existing per-page image conversion, not from the
  continuation layer (the plain and continuation peaks are essentially
  identical).

## Why there is no Cartesian product

The grouping is a local-adjacency algorithm:

1. fragments are sorted once by `(page, order, sub)` -- O(F log F) for F
   fragments;
2. for each fragment, edges are evaluated only against fragments on the
   immediately preceding selected page, never against all earlier pages;
3. each fragment keeps at most one predecessor and one successor, so the
   chain construction is a single O(F) pass;
4. column reassembly compares each continuation fragment only to the
   canonical column layout of its group's first fragment, again O(F) in the
   total number of fragments.

The total decision work is therefore O(F) edge evaluations with small
constant-size geometry comparisons per edge, not O(F^2) page pairs and never
a product of cell counts.

## Worst case

The worst case for the *number of edges* is a report where every selected
page holds K side-by-side tables: each of the F fragments still only
compares against the K fragments of the previous page, giving O(F * K)
geometry comparisons.  Because K is the number of independent tables on a
single page, it stays small and bounded by the page's horizontal capacity;
it does not grow with the document length.  Memory remains O(F) because the
evidence of every fragment is stored exactly once.

Ambiguity handling is also linear: when several predecessor edges tie in
strength, the competing edges are recorded as one diagnostic and the
fragment is left unmerged -- no backtracking or global re-optimization over
all fragment pairs is performed.
