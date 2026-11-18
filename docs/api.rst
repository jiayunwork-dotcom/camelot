.. _api:

API Reference
=============

.. module:: camelot

Main Interface
--------------
.. autofunction:: camelot.read_pdf

Lower-Level Classes
-------------------

.. autoclass:: camelot.handlers.PDFHandler
   :inherited-members:

.. autoclass:: camelot.parsers.Stream
   :inherited-members:

.. autoclass:: camelot.parsers.Lattice
   :inherited-members:

.. autoclass:: camelot.parsers.Network
   :inherited-members:

.. autoclass:: camelot.parsers.Hybrid
   :inherited-members:

Lower-Lower-Level Classes
-------------------------

.. autoclass:: camelot.core.TableList
   :inherited-members:

.. autoclass:: camelot.core.Table
   :inherited-members:

.. autoclass:: camelot.core.Cell

Cross-Page Continuation
-----------------------

When a PDF is read with ``read_pdf(..., continuation="auto")`` the returned
:class:`~camelot.core.TableList` keeps every physical table and additionally
exposes the assembled logical tables and grouping diagnostics.

.. autoclass:: camelot.continuation.LogicalTable
   :inherited-members:

.. autoclass:: camelot.continuation.LogicalFragment

.. autoclass:: camelot.continuation.FragmentContext
   :inherited-members:

.. autoclass:: camelot.continuation.RowSource
   :inherited-members:

.. autoclass:: camelot.continuation.CaptionInfo
   :inherited-members:

Merged Cell Structure (Lattice)
-------------------------------

Physical lattice tables expose a lazy structural view of merged regions via
``Table.structure``; :meth:`~camelot.core.Table.to_html` and
:meth:`~camelot.core.Table.to_excel` accept ``keep_merges=True`` to export it.

.. autoclass:: camelot.structure.TableStructure
   :inherited-members:

.. autoclass:: camelot.structure.StructuralCell
   :inherited-members:

.. autoclass:: camelot.structure.TextSource
   :inherited-members:

.. autoclass:: camelot.structure.StructureDiagnostic
   :inherited-members:

.. autoexception:: camelot.structure.StructureUnavailableError

Plotting
--------

.. autofunction:: camelot.plot

.. autoclass:: camelot.plotting.PlotMethods
   :inherited-members:
