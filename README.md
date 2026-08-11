# pynst 0.3.0

Generic nested-sweep execution, chunk storage and HDF5 data model.

## Measurement return values

Legacy list mode remains supported:

```python
return [measurement_df, status_df]
```

This creates:

```text
/block_0
/block_1
```

Named mapping mode is also supported:

```python
return {
    "measurement": measurement_df,
    "status": status_df,
}
```

This creates:

```text
/measurement
/status
```

The return mode and structure must remain constant during a sweep. Mapping
block insertion order is irrelevant; list positions are structural.

## Safe resume contract

Each new run writes `sweep_contract.json`. It binds the exact ordered sweep
grid, parameter names and dtypes, return container type, block names/list
positions, and the observed DataFrame schema. Local-index and dependent-variable
dtypes are structural; categorical dtypes additionally bind the complete
category vocabulary and its ordering. A declared schema may omit dtypes; the
observed dtypes are then added without invalidating the original declaration on
resume. Sweep values must be finite and serializable by PyNST's generic JSON
data model. Because pandas HDF cannot persist nullable/string extension arrays
or complex-valued index levels reliably, result/index extension dtypes other
than `CategoricalDtype` are rejected before schema binding; represent complex
indices as separate real/imaginary levels. Complex dependent-variable columns
remain supported. Object-typed index levels must contain homogeneous strings;
object dvar columns may additionally contain only missing values. Mixed Python
object sweep levels are rejected during manager construction; mixed result
objects are rejected immediately after the callback and before schema binding.
Unsigned 64-bit index levels are likewise rejected because PyTables cannot
index them; use a range-checked `int64` or string representation instead.

PyNST deliberately keeps this contract generic. Domain identities such as a
load-pull plan/model UUID and instrument-specific checks such as a PNA
frequency axis belong in the domain measurement wrapper.

Every run also owns an atomically updated `run_manifest.json`. The manifest,
not `simlog.tsv`, is the authority for completed and failed sequence indices.
Each HDF chunk embeds its run UUID, contract fingerprint, chunk index, block
structure and exact `sequence_index` list. Resume validates all manifest-owned
chunks before calling the measurement function, adopts only fully validated
crash-orphan chunks, and rejects missing, corrupt, overlapping or foreign
chunks. The TSV log is rebuilt from this state and remains diagnostic only.
Legacy v1/v2 runs are upgraded through a recoverable two-phase migration, so a
process stop between contract and manifest replacement does not strand the run.

`sequence_index` is the canonical identity of a grid entry. Parameter string
representations are never used as storage keys; strings containing tabs or
otherwise awkward display text therefore remain unambiguous.

The measurement name must be one safe relative path component. A non-blocking
OS file lock protects each run from concurrent managers, including cleanup,
resume reconciliation, execution, metadata updates and merge preparation.

`run()` returns `True` only if every contract sequence is committed exactly
once. Recoverable measurement exceptions leave explicit failed sequences and
return `False`; schema, persistence and critical measurement errors abort the
run. `KeyboardInterrupt` and `SystemExit` first attempt to commit already
accepted buffered data and are then re-raised.

## Previous valid result

Set `provide_previous_result=True` when the next measurement needs information
from its immediate valid predecessor:

```python
def measure(params, previous_result):
    if previous_result is not None:
        previous_status = previous_result["status"]
    # perform measurement
    return {"measurement": measurement_df, "status": status_df}

manager = SweepManager(
    measurement_func=measure,
    ivars=ivars,
    meas_name="run_001",
    provide_previous_result=True,
)
```

During a normal run the predecessor is the last successfully normalised and
accepted result. During resume it is restored from the exact complete chunk
referenced by the run manifest. Failed or partial points never replace it. If the
referenced result cannot be loaded unambiguously, the sweep aborts before the
next measurement callback. The callback receives a defensive copy with the
global sweep levels already attached to each DataFrame index.

## Metadata

Global metadata is passed to `SweepManager`:

```python
manager = SweepManager(
    measurement_func=measure,
    ivars=ivars,
    meas_name="run_001",
    metadata={
        "operator": "Michael Loose",
        "blocks": {
            "measurement": {
                "description": "Primary measurement block",
                "variables": {
                    "frequency": {"unit": "Hz"},
                    "gain": {"unit": "dB", "role": "measurement"},
                },
            },
            "waveforms": {
                "default_load": False,
            },
        },
    },
)
```

The following structural fields are generated automatically per block:

- `ivars`
- `sweep_ivars`
- `local_ivars`
- `dvars`
- variable `dtype`

The following semantic fields are optional:

- `unit`
- `role`
- `description`
- arbitrary additional metadata

Merged files contain:

```text
/<measurement blocks>
/__metadata__/config
/__metadata__/log
/__metadata__/traceback   # only when errors were recorded
```

The measurement blocks themselves retain their historical DataFrame format.
Declared mapping keys or list positions that are always `None` are retained in
`pynst_result_block_names` metadata and the root `pynst_block_names_json`
attribute even though they have no materialized HDF block. Embedded log columns
are namespaced as `parameter:<name>` and `pynst:<field>`.

## Generic dataset

```python
from pynst import GenericSweepDataset

dataset = GenericSweepDataset("merged.h5")

print(dataset.block_names)
frame = dataset["measurement"]
block_metadata = dataset.get_block_metadata("measurement")
log = dataset.read_log()
```

Domain libraries should subclass `BaseSweepDataset` and implement
`validate_domain()` plus any domain-specific reshaping or processing.

## Merge behaviour

`partial_merge()` creates a table-format merged file without loading complete
blocks into RAM.

`merge()` creates a block-wise optimized merged file and preserves the full
MultiIndex. It uses fixed-format storage when possible and transparently falls
back to table format for pandas extension dtypes such as categorical levels.

Both methods write to a temporary file first and replace the target only after
a successful, durable write. By default both require a complete run;
`require_complete=False` is the explicit diagnostic escape hatch for a partial
artifact. Merge targets are locked across runs and may not overwrite chunks or
run-control files. Random private temporary names prevent collisions with user
files. `drop_columns` may not remove every dependent variable from a block.

`remove_chunks=True` is permitted only for a complete merge. The merged file is
deeply validated, then its path, hash and size are committed to the manifest as
an archived artifact before source chunks are retired. Archived runs cannot be
resumed accidentally. The archive target must be outside the run directory so
that a later intentional `resume=False` replacement cannot delete the sole
retained artifact.

## Durability boundary

PyNST guarantees crash-consistent local persistence and at-least-once execution
of an external measurement callback. No general sweep library can prove that a
hardware action happened exactly once if the process dies after the instrument
acted but before the result was durably committed. Measurement functions must
therefore tolerate a repeated point after such an ambiguous crash. Domain and
instrument safety checks remain the responsibility of the measurement wrapper.
