.. _continuation:

Collecting a Detail Table Across Pages
======================================

Public reports often split one logical table across many pages: every
continuation page repeats the header, may shift horizontally, use a
different page width, or lose a column when it is entirely empty.  By
default Camelot returns one *physical* table per detected page fragment, so
a single logical table shows up as several independent results.  Pass
``continuation="auto"`` to :func:`camelot.read_pdf` to additionally collect
these fragments into logical tables.

.. code-block:: python

    import camelot

    tables = camelot.read_pdf(
        "annual_report.pdf",
        pages="1-end",
        flavor="stream",
        continuation="auto",
    )

    # Physical tables are still all there, unchanged.
    print(tables.n)

    # Logical tables group continuation fragments across pages.
    for logical in tables.logical_tables:
        print(logical.logical_id, logical.df.shape, logical.fragments)

``continuation`` accepts only ``None`` (the default, legacy behavior) or
``"auto"``; it is supported by the ``stream`` and ``lattice`` flavors.
Network and hybrid raise ``ValueError``.

How grouping is decided
-----------------------

Grouping uses evidence collected while each page is parsed:

* the original PDF page number and left-bottom PDF coordinates,
* column and row geometry of every fragment,
* the multi-row header detected inside the fragment,
* table-number captions (for example ``Table 3``), continuation wording
  (``Table 3 (continued)`` / ``续表``) and chapter headings found near but
  outside the fragment.

A fragment only joins a predecessor on the immediately preceding selected
page.  Distinct table numbers, a chapter/section switch, missing
intermediate pages (a gap in the selected page list), same-page side-by-side
tables and incompatible column geometry all block a merge.  When two
predecessors are equally strong candidates for the same continuation
fragment, Camelot does not guess: the fragments stay separate and the
competition is listed in ``tables.continuation_diagnostics`` with the
adopted/rejected evidence for each edge.

Recovering a column lost on a continuation page
-----------------------------------------------

When a whole column is empty on a continuation page, stream cannot infer
the column and lattice may only see an interrupted vertical ruling line.
After the fragments of a logical table are known, Camelot rebuilds every
continuation fragment against the column layout established by the *first*
fragment, reassigning text from the original page text lines and their PDF
geometry.  The canonical number of columns is therefore preserved; no data
is shifted into the wrong column.  Each repair is recorded in the logical
table's ``diagnostics``.

The default physical extraction is never modified: physical ``df``,
``cells``, ``page``, ``order`` and ``parsing_report`` stay identical with or
without ``continuation="auto"``.  A logical table may use a column layout
different from the single-page physical layout; the physical tables remain
available as evidence.

Tracing every row back to the page
-----------------------------------

Each :class:`~camelot.continuation.LogicalTable` exposes ``row_sources``:
for every logical row it stores the original PDF page number, the physical
table order, the physical row index, and -- for each non-empty cell -- the
logical column, the original retained text and the cell bounding box in the
page's PDF coordinate space.  Repeated headers removed from continuation
fragments are listed separately in ``removed_headers`` so the removal stays
auditable; body text that merely equals a header word is never removed.

.. code-block:: python

    logical = tables.logical_tables[0]
    for source in logical.row_sources:
        print(source.page, source.order, source.physical_row)

Exporting logical tables
------------------------

:meth:`camelot.core.TableList.export` accepts ``logical=True`` for CSV and
JSON output.  Every logical table gets its own data file (named so that two
tables starting on the same page cannot overwrite each other) plus a JSON
manifest referencing the data file names and the logical row numbers, row
sources, removed headers and diagnostics.  With ``compress=True`` the data
files and manifest are bundled into one ZIP whose internal references still
resolve.  Excel, HTML, Markdown and SQLite logical export raise
``ValueError``; use the physical export for those formats.

.. code-block:: python

    tables.export("logical.csv", f="csv", logical=True)
    tables.export("logical.zip", f="csv", logical=True, compress=True)

Design trade-offs
-----------------

The grouping is a *local adjacency* strategy: each boundary considers only
the previous selected page instead of optimizing over every fragment pair.
This keeps the cost linear in the number of fragments and avoids any
Cartesian product of pages or cells, and makes every accept/reject decision
explainable from the page evidence.  The conservative trade-off is that a
fragment separated from its predecessor by an unselected page cannot be
re-attached -- the missing-page boundary is reported rather than guessed,
which prevents false merges across unrelated tables.
