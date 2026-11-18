.. _structure:

Recovering Merged Cells from a Lattice Table
=============================================

Lattice tables are built in two independent stages: line and joint detection
decides where the grid boundaries are, and text assignment decides which grid
cell owns each extracted text line.  Neither stage exposes merged cells, so a
spanning title and a genuinely empty cell both look like ``""`` in the flat
``Table.df``.

For physical lattice tables Camelot derives a structural view on demand:

.. code-block:: python

    import camelot

    tables = camelot.read_pdf("merged_header.pdf", flavor="lattice")
    table = tables[0]
    structure = table.structure

    for cell in structure.cells:
        print(
            cell.row, cell.col,          # top-left anchor
            cell.rowspan, cell.colspan,  # grid rows / columns covered
            cell.page, cell.order,       # original page and table order
            cell.bbox,                   # (x0, y0, x1, y1) in PDF coordinates
            repr(cell.text),
        )

How it is exposed
-----------------

``table.structure`` is a :class:`camelot.structure.TableStructure`:

* ``cells`` covers every grid position exactly once.  A merged region appears
  once at its top-left anchor with ``rowspan``/``colspan``; a standalone cell,
  including a standalone empty cell next to a merged region, appears as an
  ordinary ``rowspan == colspan == 1`` cell with empty ``text`` and no
  ``sources``.
* ``cell.sources`` keeps one
  :class:`~camelot.structure.TextSource` per original text fragment, with the
  grid position and bounding box the text stage assigned it *before*
  ``shift_text``/``copy_text``.  Two source positions that carry the same
  string stay two sources while ``cell.text`` keeps one copy of the string.
* ``diagnostics`` lists every question the evidence could not settle; see
  below.
* ``to_dict()`` returns the same information as plain JSON-friendly data.

The view is computed lazily on first access and cached.  It is read-only: it
never reads or rewrites ``df``, ``cells``, ``parsing_report`` or the default
exports.  ``shift_text`` and ``copy_text`` still move or copy text in the flat
matrix exactly as before; the structural regions and their texts do not
change.

When the view is available
--------------------------

The structural view requires the evidence produced while parsing a single
page with ``flavor="lattice"``: the detected line segments inside the table
and the pre-shift text assignment.  Accessing ``table.structure`` on a table
produced by another flavor raises
:class:`camelot.structure.StructureUnavailableError` (a subclass of
``NotImplementedError``) naming the flavor.  The same error is raised for
tables with no lattice evidence -- hand-built tables, tables rebuilt by
``continuation="auto"`` and logical tables -- and for tables whose line
evidence is missing.  Logical tables and other flavors are never silently
given a guessed structure.

Exports with merges
-------------------

Both exports accept ``keep_merges=True``.  The default remains the flat
matrix.

.. code-block:: python

    table.to_html("merged.html", keep_merges=True)
    table.to_excel("merged.xlsx", keep_merges=True)

* HTML contains real ``rowspan``/``colspan`` attributes; reading the file back
  (for example with a browser or any HTML parser) shows one cell per merged
  region.  Anchor text is written once, covered positions are omitted.
* Excel contains real merged ranges (``ws.merged_cells.ranges``).  The anchor
  text is stored as an Excel string, covered positions are left empty, and
  standalone empty cells have no value.  Reading the workbook back with
  ``openpyxl`` yields plain ``str`` values for the strings.
* Both exports use the structural view as their source of truth, so they agree
  with each other and with the documented diagnostics treatment: a region
  involved in a diagnostic stays as separate cells and keeps its text.
* ``keep_merges=True`` only accepts ``classes``/``border``/``escape`` for HTML
  and no options for Excel; other DataFrame options raise ``TypeError``
  instead of being silently ignored.

Reading diagnostics
-------------------

``structure.diagnostics`` is a list of
:class:`~camelot.structure.StructureDiagnostic`.  Each entry carries the
original ``page`` and table ``order``, the affected grid positions
(``positions``, plus ``details["grid_positions"]``) and a ``message``.  The
``kind`` is one of:

``partial_seam``
    A seam between two adjacent grid cells is only partly covered by a line
    (a short break): the line coverage fraction is in ``details["coverage"]``.
    The cells are *not* merged; both keep their own text, which is also
    repeated in ``details["texts"]``.
``seam_conflict``
    The measured line coverage and the lattice per-cell edge flags disagree
    (a boundary recorded where no line exists, or a line along a seam the
    lattice did not record).  The cells are kept apart and the texts retained.
``non_rectangular_component``
    Missing separators connect cells into an L-shaped or otherwise ragged
    region that does not fill its bounding rectangle.  ``details`` includes
    the ``bounding_box`` and the ``missing_positions``.  More than one
    rectangular interpretation is possible, so no merge is created and each
    cell keeps its text.
``text_assignment_mismatch``
    The text stage assigned a fragment to a cell whose geometry does not
    contain the fragment's centre.  The text is retained; the mismatch is
    reported rather than resolved.

How seams are classified
------------------------

For every interior seam the fraction of its length covered by the union of
detected line segments aligned with the seam is measured (segment coordinate
matched within ``joint_tol``):

* coverage ``<= 0.15`` with both adjacent lattice edge flags open -> the seam
  is **open** and joins the two cells;
* coverage ``>= 0.5`` with at least one adjacent cell recording the boundary
  -> the seam is **solid**, and a drawn separator is never crossed by a
  region;
* everything else -- a short break, no pixels where the lattice recorded a
  boundary, or a mostly covered seam the lattice did not record -- is
  reported and never joins cells.

Regions are connected components of grid cells under open seams, and a
component is only accepted when it exactly fills its bounding rectangle.

Handling line noise, and limits
-------------------------------

Line segments are detected from a binarised, morphologically cleaned image;
short gaps can survive as short missing pieces, while stray text strokes can
in rare cases be detected as short lines.  The thresholds keep those cases
apart: a seam with more than 15% line coverage is never opened, so a short
break cannot silently merge two cells, and a clearly drawn separator (50% or
more) with a recorded lattice boundary is always solid, so it is never
swallowed.  The conservative middle band is reported instead of guessed.

Known failure modes:

* **missed merge** -- a real merged cell whose open seam still has more than
  15% line coverage (for example a text stroke or a border stub crossing it)
  is classified ``partial_seam`` and kept as separate cells.  The loss is
  visible in ``diagnostics`` with the measured coverage.
* **false merge** -- a real separator that was eroded or never detected
  leaves a seam with both edge flags open; if its coverage also stays below
  15% the two cells are joined.  A mostly-covered seam in the same situation
  is instead reported as ``partial_seam``.
* **lost merge near a break** -- a seam drawn 50-85% of its length (a short
  break) is reported and never merged, even if a single merged region was
  intended.
* **ragged regions** -- when the open seams form an L shape, no rectangle is
  chosen; every member cell is listed in the diagnostic and keeps its text.

The conservative behaviour is deliberate: a questionable merge is never
made, and every question is recorded with its page, table order, grid
positions and coverage.

Cost
----

The computation touches every interior seam once, ``O(R * C)`` for a table
with ``R`` rows and ``C`` columns, and performs union-find over the
``R * C`` grid cells.  Segments are indexed by their fixed coordinate and
sorted once, so only segments near a seam are inspected.  Rectangles are
never enumerated, so the cost grows linearly with the grid size rather than
with combinations of rows and columns.  Computing the view for a 40x4 table
takes about a millisecond; a 50x8 table takes a few milliseconds.
