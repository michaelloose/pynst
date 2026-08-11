"""Generic read-only dataset access for merged pynst sweep files."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from .data_model import BlockMetadata, SweepMetadata, json_default


_METADATA_PREFIX = "/__metadata__/"
_MISSING_INDEX_VALUE = object()


def _normalise_key(name: str) -> str:
    key = str(name).strip("/")
    if not key:
        raise ValueError("HDF block name must not be empty.")
    if "/" in key:
        raise ValueError(
            "HDF block names must not contain '/'. "
            f"Received {name!r}."
        )
    return key


def _is_data_block_key(key: str) -> bool:
    """Exclude pandas-internal nested table metadata from user blocks."""
    normalised = str(key).strip("/")
    return (
        bool(normalised)
        and "/" not in normalised
        and not normalised.startswith("__metadata__")
    )


def _data_block_keys(store: pd.HDFStore) -> list[str]:
    keys = list(store.keys())
    data_keys = [key for key in keys if _is_data_block_key(key)]
    top_level = {key.strip("/") for key in data_keys}
    for key in keys:
        normalised = key.strip("/")
        if not normalised or normalised.startswith("__metadata__"):
            continue
        if "/" not in normalised:
            continue
        parts = normalised.split("/")
        is_pandas_categorical_metadata = (
            len(parts) >= 4
            and parts[0] in top_level
            and parts[1] == "meta"
            and parts[-1] == "meta"
        )
        if not is_pandas_categorical_metadata:
            raise ValueError(
                "HDF block names must not contain nested '/' paths; "
                f"{key!r} is not pandas categorical metadata."
            )
    return data_keys


def _dtype_spec(dtype: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {"dtype": str(dtype)}
    if isinstance(dtype, pd.CategoricalDtype):
        spec["categories"] = [
            json.loads(
                json.dumps(
                    value,
                    default=json_default,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            for value in dtype.categories.tolist()
        ]
        spec["ordered"] = bool(dtype.ordered)
    return spec


def _canonical_index_value(value: Any) -> Any:
    """Make scalar missing values compare equal across validation chunks."""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return _MISSING_INDEX_VALUE
    return value


class BaseSweepDataset(ABC):
    """Abstract, domain-independent view of a merged sweep file.

    This base class intentionally does not reshape blocks into a specialised
    analysis representation. It exposes the stored long-form DataFrames and
    their metadata. Domain libraries may subclass it and implement
    ``validate_domain`` plus domain-specific transformation methods.
    """

    def __init__(
        self,
        file_path: str | Path,
        *,
        validate: bool = True,
    ) -> None:
        self.file_path = Path(file_path).expanduser().resolve()
        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)
        if not self.file_path.is_file():
            raise ValueError(
                f"Sweep dataset path must reference a file: "
                f"{self.file_path}"
            )

        self._block_names: list[str] = []
        self.metadata = self._read_metadata_and_keys()

        if validate:
            self.validate_storage(deep=False)
            self.validate_domain()

    def _read_metadata_and_keys(self) -> SweepMetadata:
        with pd.HDFStore(self.file_path, mode="r") as store:
            self._block_names = [
                _normalise_key(key)
                for key in _data_block_keys(store)
            ]

            config_key = "/__metadata__/config"
            if config_key in store.keys():
                config_df = store[config_key]
                metadata = SweepMetadata.from_dict(
                    json.loads(config_df.iloc[0]["json"])
                )
            else:
                # Legacy files remain readable. They simply lack an explicit
                # distinction between global and local independent variables.
                blocks: dict[str, BlockMetadata] = {}
                for block_name in self._block_names:
                    frame = store[f"/{block_name}"]
                    ivars = [
                        str(name) if name is not None else f"index_{index}"
                        for index, name in enumerate(frame.index.names)
                    ]
                    block = BlockMetadata(
                        hdf_key=block_name,
                        ivars=ivars,
                        sweep_ivars=[],
                        local_ivars=ivars,
                        dvars=[str(name) for name in frame.columns],
                    )
                    blocks[block_name] = block

                metadata = SweepMetadata(
                    measurement_name=self.file_path.stem,
                    nested_sweep_levels=[],
                    blocks=blocks,
                    schema_version=0,
                    extra={"legacy_file": True},
                )

        return metadata

    @property
    def block_names(self) -> tuple[str, ...]:
        return tuple(self._block_names)

    def __contains__(self, block_name: str) -> bool:
        return _normalise_key(block_name) in self._block_names

    def __getitem__(self, block_name: str) -> pd.DataFrame:
        return self.get_block(block_name)

    def get_block(
        self,
        block_name: str,
        *,
        where: str | list[str] | None = None,
        start: int | None = None,
        stop: int | None = None,
    ) -> pd.DataFrame:
        """Read one stored block.

        ``where``, ``start`` and ``stop`` are forwarded to HDF table selection
        when possible. Fixed-format blocks fall back to full loading followed
        by positional slicing.
        """
        key = _normalise_key(block_name)
        if key not in self._block_names:
            raise KeyError(
                f"Unknown block {key!r}. Available: {self._block_names!r}"
            )

        with pd.HDFStore(self.file_path, mode="r") as store:
            hdf_key = f"/{key}"
            try:
                return store.select(
                    hdf_key,
                    where=where,
                    start=start,
                    stop=stop,
                )
            except (TypeError, ValueError, NotImplementedError):
                frame = store[hdf_key]

        if where is not None:
            raise ValueError(
                f"Block {key!r} does not support HDF query selection."
            )
        return frame.iloc[slice(start, stop)]

    def iter_blocks(
        self,
        names: Iterable[str] | None = None,
        *,
        default_load_only: bool = False,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """Yield blocks one at a time to avoid loading the entire file."""
        selected = (
            list(names)
            if names is not None
            else list(self._block_names)
        )

        for name in selected:
            key = _normalise_key(name)
            block_meta = self.metadata.blocks.get(key)
            if (
                default_load_only
                and block_meta is not None
                and not block_meta.default_load
            ):
                continue
            yield key, self.get_block(key)

    def get_block_metadata(self, block_name: str) -> BlockMetadata:
        key = _normalise_key(block_name)
        try:
            return self.metadata.blocks[key]
        except KeyError as exc:
            raise KeyError(
                f"No metadata available for block {key!r}."
            ) from exc

    def read_log(self) -> pd.DataFrame:
        key = "/__metadata__/log"
        with pd.HDFStore(self.file_path, mode="r") as store:
            if key not in store.keys():
                return pd.DataFrame()
            return store[key]

    def read_traceback(self) -> str:
        key = "/__metadata__/traceback"
        with pd.HDFStore(self.file_path, mode="r") as store:
            if key not in store.keys():
                return ""
            frame = store[key]
        return "" if frame.empty else str(frame.iloc[0]["text"])

    @staticmethod
    def _validate_frame_schema(
        key: str,
        frame: pd.DataFrame,
        block: BlockMetadata,
    ) -> None:
        actual_index_names = [
            str(name) if name is not None else f"index_{index}"
            for index, name in enumerate(frame.index.names)
        ]
        if actual_index_names != block.ivars:
            raise ValueError(
                f"{key!r}: stored index names differ from metadata."
            )
        if [str(name) for name in frame.columns] != block.dvars:
            raise ValueError(
                f"{key!r}: stored columns differ from metadata."
            )
        for level, variable_name in enumerate(block.ivars):
            variable = block.variables.get(variable_name)
            expected_dtype = None if variable is None else variable.dtype
            dtype = frame.index.get_level_values(level).dtype
            actual_dtype = str(dtype)
            if expected_dtype is not None and actual_dtype != expected_dtype:
                raise ValueError(
                    f"{key!r}: index dtype for {variable_name!r} differs "
                    "from metadata."
                )
            actual_spec = _dtype_spec(dtype)
            for field in ("categories", "ordered"):
                if (
                    variable is not None
                    and field in variable.extra
                    and variable.extra[field] != actual_spec.get(field)
                ):
                    raise ValueError(
                        f"{key!r}: index categorical {field} for "
                        f"{variable_name!r} differs from metadata."
                    )
        for variable_name in block.dvars:
            variable = block.variables.get(variable_name)
            expected_dtype = None if variable is None else variable.dtype
            dtype = frame[variable_name].dtype
            actual_dtype = str(dtype)
            if expected_dtype is not None and actual_dtype != expected_dtype:
                raise ValueError(
                    f"{key!r}: column dtype for {variable_name!r} differs "
                    "from metadata."
                )
            actual_spec = _dtype_spec(dtype)
            for field in ("categories", "ordered"):
                if (
                    variable is not None
                    and field in variable.extra
                    and variable.extra[field] != actual_spec.get(field)
                ):
                    raise ValueError(
                        f"{key!r}: column categorical {field} for "
                        f"{variable_name!r} differs from metadata."
                    )

    def validate_storage(self, *, deep: bool = False) -> None:
        """Validate metadata and optionally compare it to stored DataFrames."""
        self.metadata.validate()

        for key, block in self.metadata.blocks.items():
            normalised_key = _normalise_key(key)
            normalised_hdf_key = _normalise_key(block.hdf_key)
            if key != normalised_key or block.hdf_key != normalised_hdf_key:
                raise ValueError(
                    f"Invalid metadata block name {key!r} / "
                    f"{block.hdf_key!r}."
                )

        missing_required = [
            key
            for key, block in self.metadata.blocks.items()
            if block.required and key not in self._block_names
        ]
        if missing_required:
            raise ValueError(
                f"Required HDF blocks are missing: {missing_required!r}"
            )

        if not deep:
            return

        if self.metadata.schema_version > 0:
            blocks_without_metadata = sorted(
                set(self._block_names) - set(self.metadata.blocks)
            )
            if blocks_without_metadata:
                raise ValueError(
                    "Stored HDF blocks have no metadata entries: "
                    f"{blocks_without_metadata!r}"
                )

        with pd.HDFStore(self.file_path, mode="r") as store:
            for key in self._block_names:
                block = self.metadata.blocks.get(key)
                if block is None:
                    continue
                hdf_key = f"/{key}"
                storer = store.get_storer(hdf_key)
                if storer.is_table:
                    frames: Iterable[pd.DataFrame] = store.select(
                        hdf_key,
                        chunksize=100_000,
                    )
                else:
                    frames = (store[hdf_key],)

                seen_index_values: set[Any] = set()
                observed_rows = 0
                for frame in frames:
                    observed_rows += len(frame)
                    self._validate_frame_schema(key, frame, block)
                    if not frame.index.is_unique:
                        raise ValueError(
                            f"{key!r}: stored index contains duplicate "
                            "entries."
                        )
                    try:
                        index_values = {
                            tuple(
                                _canonical_index_value(value)
                                for value in row
                            )
                            for row in frame.index.to_frame(
                                index=False
                            ).itertuples(index=False, name=None)
                        }
                    except TypeError as error:
                        raise ValueError(
                            f"{key!r}: stored index values must be hashable "
                            "for exact uniqueness validation."
                        ) from error
                    if seen_index_values & index_values:
                        raise ValueError(
                            f"{key!r}: stored index contains duplicate "
                            "entries across table chunks."
                        )
                    seen_index_values.update(index_values)
                if observed_rows == 0:
                    raise ValueError(
                        f"{key!r}: stored measurement block is empty."
                    )

    @abstractmethod
    def validate_domain(self) -> None:
        """Validate domain-specific assumptions in a subclass."""

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"file_path={str(self.file_path)!r}, "
            f"blocks={self._block_names!r}, "
            f"schema_version={self.metadata.schema_version})"
        )


class GenericSweepDataset(BaseSweepDataset):
    """Concrete generic reader without domain-specific validation."""

    def validate_domain(self) -> None:
        return None
