Getting started
===============

Installation
------------

PyNST requires Python 3.10 or newer. Install it with:

.. code-block:: console

   python -m pip install pynst

Install the optional Matplotlib and Jupyter integration with:

.. code-block:: console

   python -m pip install "pynst[visualization]"

To work from a source checkout instead:

.. code-block:: console

   python -m pip install -e .

A minimal sweep
---------------

A sweep consists of an ordered :class:`pandas.MultiIndex`, a measurement
callback and a unique measurement name. The callback receives the current
parameter values as a mapping and returns one or more named DataFrames.

.. code-block:: python

   from pathlib import Path

   import pandas as pd

   from pynst import GenericSweepDataset, SweepManager
   from pynst.planning import create_multiindex, create_range


   def measure(params):
       bias = params["bias"]
       return {
           "measurement": pd.DataFrame(
               {"response": [bias**2]},
               index=pd.Index([0], name="sample"),
           )
       }


   bias = create_range(min=0.0, max=1.0, step=0.5)
   sweep_grid = create_multiindex([bias], names=["bias"])

   with SweepManager(
       measurement_func=measure,
       ivars=sweep_grid,
       meas_name="example_run",
       output_root=Path("runs"),
       chunk_size=2,
       resume=False,
   ) as manager:
       if manager.run():
           manager.merge("example_run.h5")

   dataset = GenericSweepDataset("example_run.h5")
   print(dataset.block_names)
   print(dataset["measurement"])

The DataFrame's local ``sample`` index is retained. PyNST prepends the global
``bias`` sweep level, so the merged block has a named MultiIndex that identifies
every result row.

Named and positional result blocks
----------------------------------

Named mappings are recommended for new code:

.. code-block:: python

   return {
       "measurement": measurement_df,
       "status": status_df,
   }

The mapping keys become HDF5 block names. Their insertion order is irrelevant,
but their names and schemas must remain stable throughout a run.

Legacy positional results remain supported:

.. code-block:: python

   return [measurement_df, status_df]

They are stored as ``block_0`` and ``block_1``. Positions are structural, so a
block must not move to another list position during a run or resume.

Resuming a run
--------------

``resume=True`` is the default. Construct a manager with the same sweep grid,
measurement name and result schema to continue an interrupted run. PyNST
validates the stored contract and all manifest-owned chunks before invoking the
measurement callback again. See :doc:`persistence` for the exact guarantees.
