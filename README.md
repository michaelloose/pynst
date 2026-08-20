# PyNST 0.4.0

[![Tests](https://github.com/michaelloose/pynst/actions/workflows/tests.yml/badge.svg)](https://github.com/michaelloose/pynst/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/pynst/badge/?version=latest)](https://pynst.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PyNST is a domain-independent Python toolkit for planning, executing,
persisting, inspecting, and visualizing nested parameter sweeps. It combines a
pandas `MultiIndex` sweep model with crash-consistent chunk storage, strict
resume validation, and convenient tools for constructing and exploring
multidimensional measurement plans.

## Installation

PyNST requires Python 3.10 or newer.

```console
python -m pip install pynst
```

Interactive MultiIndex widgets and Matplotlib helpers are optional:

```console
python -m pip install "pynst[visualization]"
```

The optional Smith-chart background of ``PointPattern.plot(smith=True)`` is
available through ``pynst[smith]``.

For development from a source checkout:

```console
python -m pip install -e ".[dev]"
```

## A minimal sweep

```python
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


bias = create_range(min=0.0, max=1.0, step=0.25)
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
```

The callback may return a named mapping of DataFrames, which is recommended,
or the historical positional list. PyNST prepends the global sweep levels to
each local DataFrame index and stores the results in bounded HDF5 chunks.

## Planning sweep points

The `pynst.planning` package contains the generic point-generation tools that
previously lived in MultiADSweep:

```python
from pynst.planning import (
    PointPattern,
    combine_multiindexes,
    create_multiindex,
    create_range,
    define_area,
    define_range,
    estimate_sweep_time,
    validate_grid,
)
```

``define_range`` resolves and validates a range description, while
``create_range`` returns its samples. The package also supports Cartesian and
observed MultiIndex combinations, complex point patterns, grid regularity
reports, and sweep-duration estimates.

## Exploring MultiIndex data

Selection is available without a graphical backend:

```python
from pynst.visualization import MultiIndexSelector

selector = MultiIndexSelector({"response": frame}, axis="columns")
selected = selector.select(frequency=2.4e9)
```

In Jupyter, the same selection model can be controlled with widgets:

```python
from pynst.visualization import InteractiveMultiIndexPlotter

plotter = InteractiveMultiIndexPlotter(
    frame,
    active_levels=["frequency"],
)
plotter.show()
```

`plot_mi` plots every MultiIndex-labelled column on a normal Matplotlib axis;
PyNST does not register a custom projection or modify Matplotlib globally:

```python
import matplotlib.pyplot as plt

from pynst.visualization.matplotlib import plot_mi

fig, ax = plt.subplots()
plot_mi(x, y, ax=ax, label_levels=["frequency"])
ax.legend()
```

## Persistence and resume guarantees

Each run records an ordered sweep contract and an atomically updated manifest.
Before resuming, PyNST checks the grid, parameter dtypes, result structure,
DataFrame schemas, run identity, and every committed chunk. Missing, corrupt,
overlapping, or foreign chunks are rejected before the measurement callback is
called.

`merge()` offers two explicit strategies:

- `fixed` concatenates a complete block after a memory preflight and usually
  writes fixed-format HDF5.
- `streaming` appends bounded batches and avoids materializing a complete block.

Both strategies write a private temporary file and replace the target only
after successful validation. Existing version-3 files and legacy v1/v2 runs
remain supported by PyNST 0.4.0.

## Package organization

```text
pynst.planning       Point and sweep-plan construction
pynst.execution      Sweep execution and public errors
pynst.storage        Chunk and persistence infrastructure
pynst.data           Metadata, datasets, and DataFrame helpers
pynst.visualization  Optional MultiIndex selection and plotting
```

Established imports such as `from pynst import SweepManager` and
`from pynst.dataset import GenericSweepDataset` remain valid.

## Documentation

The complete user guide, API reference, and tutorial notebooks are published at
[pynst.readthedocs.io](https://pynst.readthedocs.io/). The documentation can be
built locally with:

```console
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

## Project information

PyNST is developed at the Chair of Smart Electronics and Systems (LITES),
Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
Michael Loose is the author and copyright holder; Alexander Deublein is a
contributor. See [AUTHORS.md](AUTHORS.md), [CONTRIBUTING.md](CONTRIBUTING.md),
and [CITATION.cff](CITATION.cff).

PyNST is distributed under the [MIT License](LICENSE).
