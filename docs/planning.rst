Planning sweeps
===============

``pynst.planning`` contains generic tools for constructing the ordered
:class:`pandas.MultiIndex` consumed by :class:`pynst.SweepManager`.

Ranges and areas
----------------

:func:`pynst.planning.define_range` accepts the familiar combinations of
``min``, ``max``, ``center``, ``span``, ``step`` and ``num``. It validates
ambiguous or underdetermined combinations and returns the resolved definition.
:func:`pynst.planning.create_range` produces the corresponding NumPy samples.

:func:`pynst.planning.define_area` resolves paired two-dimensional range
definitions from the same vocabulary. Point-pattern generators use those
definitions to create the actual coordinates.

MultiIndex construction
-----------------------

:func:`pynst.planning.create_multiindex` creates a Cartesian sweep grid from
named arrays. :func:`pynst.planning.combine_multiindexes` supports two distinct
semantics:

``mode="existing"``
   Preserve the tuples observed inside every input index, then take the
   Cartesian product between indexes. This retains correlated levels such as a
   magnitude/angle point pattern.

``mode="levels"``
   Recombine every individual level value. Use this only when all combinations
   are physically meaningful.

Point patterns
--------------

:class:`pynst.planning.PointPattern` builds complex-valued grids and converts
them into named real/imaginary or magnitude/angle MultiIndexes. Supporting
rounding and duplicate-separation helpers are part of the same package and do
not depend on ADS or a plotting backend.

Basic point plotting is included in the ``visualization`` extra. A Smith-chart
background additionally requires the ``smith`` extra.

Validation and timing
---------------------

:func:`pynst.planning.validate_grid` returns a structured report containing
regularity, expected and observed sizes, duplicate points and missing
combinations. :func:`pynst.planning.is_regular_grid` provides the simple
boolean form.

:func:`pynst.planning.estimate_sweep_time` estimates total duration and finish
time from a point count and observed or assumed point duration.

See :doc:`tutorials/01_planning` for a complete executable example.
