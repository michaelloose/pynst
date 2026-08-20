# Contributing to PyNST

Bug reports, documentation improvements, and focused pull requests are
welcome. Please open an issue before starting a change that alters a public API
or the persisted HDF5 format.

## Development setup

Create and activate a Python 3.10 or newer virtual environment, then install an
editable checkout:

```console
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the complete test suite:

```console
python -m pytest
```

Build the documentation with warnings treated as errors:

```console
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Validate distributions before submitting a release-related change:

```console
python -m build
python -m twine check dist/*
```

## Compatibility expectations

Changes must preserve the documented public imports and existing on-disk sweep
format unless the pull request explicitly proposes a versioned migration.
Tests should cover both the new behavior and compatibility paths. Hardware- or
domain-specific behavior belongs in the corresponding measurement library, not
in PyNST's generic core.
