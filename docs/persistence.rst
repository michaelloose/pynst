Persistence, resume and merge
=============================

Run contract
------------

Each new run creates ``sweep_contract.json``. The contract binds the ordered
sweep grid, parameter names and dtypes, result container type, block names or
list positions, and the observed DataFrame schema. Resume rejects changes that
would reinterpret already committed data.

Run manifest
------------

``run_manifest.json`` is the authority for committed chunks and completed or
failed sequence indices. Every chunk embeds its run UUID, contract fingerprint,
chunk index, block structure and exact sequence list. The TSV log is diagnostic
and can be reconstructed from the manifest.

Before a resumed callback is invoked, PyNST validates manifest-owned chunks and
reconciles only fully valid crash-orphan chunks. Missing, corrupt, overlapping
or foreign chunks abort the run.

Concurrency
-----------

A non-blocking operating-system file lock protects a run from concurrent
managers. Metadata updates and merge targets are protected as well. A
measurement name must therefore be a single safe relative path component.

Merge strategies
----------------

The default fixed strategy builds one complete block in memory and normally
uses fixed-format HDF storage:

.. code-block:: python

   manager.merge("merged.h5", strategy="fixed")

Before writing, PyNST estimates whether sufficient physical memory is available.
For large runs, select the bounded-memory streaming strategy explicitly:

.. code-block:: python

   manager.merge("merged.h5", strategy="streaming")

Both strategies preserve committed chunk order and publish the target only
after a successful write and deep validation. By default, merge requires a
complete run. ``require_complete=False`` is an explicit diagnostic escape hatch.

Durability boundary
-------------------

PyNST provides crash-consistent local persistence and at-least-once execution
of the external measurement callback. No generic library can prove that a
hardware action occurred exactly once when the process dies after the device
acted but before the result was durably committed. Measurement callbacks must
therefore tolerate a repeated point after such an ambiguous crash.
