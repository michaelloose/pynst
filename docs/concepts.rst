Core concepts
=============

Sweep grid
----------

The ordered :class:`pandas.MultiIndex` passed as ``ivars`` defines the complete
sweep. Each entry receives a canonical integer ``sequence_index``. This index,
not a formatted parameter string, is the persistent identity of a sweep point.

Measurement callback
--------------------

The callback is executed serially for every missing sequence. It returns a
mapping of block names to DataFrames or a legacy list of DataFrames. A block may
be ``None``, but the return container and block structure must remain stable for
the complete run.

When ``provide_previous_result=True``, PyNST calls the callback with a second
argument containing a defensive copy of the immediate valid predecessor. On
resume, that predecessor is loaded from the exact committed chunk recorded in
the run manifest.

Chunks and merged datasets
--------------------------

Accepted results are tagged with the global sweep levels and written to
compressed HDF5 chunks. Chunks are the durable unit of an active run. A merged
file is a separate consolidated artifact intended for analysis and long-term
use.

:class:`pynst.GenericSweepDataset` exposes the stored blocks and metadata
without applying domain-specific reshaping. Domain libraries can subclass
:class:`pynst.BaseSweepDataset` and implement their own validation and analysis
helpers.

Responsibility boundary
-----------------------

PyNST validates generic persistence structure, schemas and run identity. It
does not know whether an instrument state is safe, whether a VNA frequency axis
is correct or whether a calibration belongs to a specific setup. Those checks
must live in the domain measurement wrapper.

Package responsibilities
------------------------

The public package is divided by lifecycle rather than measurement domain:

* :mod:`pynst.planning` constructs and validates sweep plans.
* :mod:`pynst.execution` owns execution and public execution errors.
* :mod:`pynst.storage` contains advanced chunk and persistence interfaces.
* :mod:`pynst.data` exposes metadata models, datasets and frame helpers.
* :mod:`pynst.visualization` contains optional MultiIndex exploration tools.

The established top-level imports remain supported. This organization allows
domain libraries to depend only on the layer they need.
