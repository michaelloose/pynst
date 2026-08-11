# pynst 0.2.0

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
positions, and the observed DataFrame schema. A declared schema may omit
dtypes; the observed dtypes are then added without invalidating the original
declaration on resume.

PyNST deliberately keeps this contract generic. Domain identities such as a
load-pull plan/model UUID and instrument-specific checks such as a PNA
frequency axis belong in the domain measurement wrapper.

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
referenced by the log. Failed or partial points never replace it. If the
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

`merge()` creates a fixed-format merged file and preserves the full MultiIndex.

Both methods write to a temporary file first and replace the target only after
a successful merge.
