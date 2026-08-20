Migrating to PyNST 0.4
======================

PyNST 0.4 reorganizes the source tree without changing the version-3 storage
format. Existing merged datasets and resumable v1/v2/v3 runs remain readable.

Existing imports remain valid:

.. code-block:: python

   from pynst import SweepManager, GenericSweepDataset
   from pynst.data_model import SweepMetadata
   from pynst.dataset import BaseSweepDataset
   from pynst.utilities import insert_row_into_df

New code may use the responsibility-specific paths:

.. code-block:: python

   from pynst.execution import SweepManager
   from pynst.data import GenericSweepDataset, SweepMetadata
   from pynst.data.frame import insert_row_into_df

Point-generation code previously importing from ``MultiADSweep.simpoints`` can
move to ``pynst.planning``. The established names ``PointPattern``,
``define_range``, ``create_range``, ``create_multiindex`` and
``combine_multiindexes`` are retained.

The interactive plotter and ``plot_mi`` now live under
``pynst.visualization``. Install the ``visualization`` extra before importing
widget or Matplotlib functionality.
