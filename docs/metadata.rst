Metadata
========

Semantic metadata is optional and does not replace PyNST's structural schema.
Pass it to :class:`pynst.SweepManager` when constructing a run:

.. code-block:: python

   manager = SweepManager(
       measurement_func=measure,
       ivars=sweep_grid,
       meas_name="example_run",
       metadata={
           "operator": "Example User",
           "blocks": {
               "measurement": {
                   "description": "Primary measurement block",
                   "variables": {
                       "frequency": {"unit": "Hz"},
                       "gain": {"unit": "dB", "role": "measurement"},
                   },
               }
           },
       },
   )

PyNST generates the structural fields for every materialized block:

* independent variables (``ivars``),
* global sweep variables (``sweep_ivars``),
* local independent variables (``local_ivars``),
* dependent variables (``dvars``), and
* stored dtypes.

Units, roles, descriptions and arbitrary additional fields remain optional.
Merged files store the global configuration under ``/__metadata__/config`` and
the diagnostic log under ``/__metadata__/log``. A traceback block is present
only when errors were recorded.

Use :meth:`pynst.BaseSweepDataset.get_block_metadata` to read one block's
metadata or the convenience methods on :class:`pynst.SweepManager` to inspect a
merged file without constructing a dataset.
