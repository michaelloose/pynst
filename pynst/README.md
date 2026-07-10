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

The return mode and block names must remain constant during a sweep.

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
