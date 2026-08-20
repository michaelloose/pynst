# Changelog

All notable changes to PyNST are documented in this file. The project follows
[Semantic Versioning](https://semver.org/) while it is in the `0.x` development
series.

## [0.4.2] - 2026-08-20

### Fixed

- Updated the displayed project and citation version to match the package
  release.

## [0.4.1] - 2026-08-20

### Fixed

- Corrected the LITES affiliation in the project and citation metadata.

## [0.4.0] - 2026-08-19

### Added

- A structured public API for sweep planning, execution, storage, data access,
  and visualization.
- Range, area, MultiIndex-combination, point-pattern, grid-validation, and
  sweep-time planning helpers migrated from MultiADSweep.
- Headless MultiIndex selection and an optional interactive Jupyter plotter.
- `plot_mi` for plotting MultiIndex-labelled pandas objects on ordinary
  Matplotlib axes.
- Sphinx API pages and executable tutorial notebooks.
- Modern package metadata, citation information, and automated release
  workflows.

### Changed

- The package uses a `src` layout and `pyproject.toml` as its metadata source.
- Internal modules are grouped by responsibility while the established import
  paths remain compatible.
- Visualization dependencies are optional through `pynst[visualization]`.

### Fixed

- Pandas 3 string inference is normalised at HDF boundaries so sweep grids,
  result blocks, resume, both merge strategies, and dataset validation retain
  the Pandas-2-compatible `object` storage schema.

### Removed

- The unused mandatory `pyserial` dependency.

## [0.3.0] - 2026-08-11

- Crash-consistent chunk persistence, strict resume contracts, deep dataset
  validation, and fixed/streaming merge strategies.

[0.4.2]: https://github.com/michaelloose/pynst/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/michaelloose/pynst/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/michaelloose/pynst/releases/tag/v0.4.0
[0.3.0]: https://github.com/michaelloose/pynst/commit/1561637
