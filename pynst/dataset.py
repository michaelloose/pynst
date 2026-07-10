"""Generic read-only dataset access for merged pynst sweep files."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from .data_model import BlockMetadata, SweepMetadata


_METADATA_PREFIX = "/__metadata__/"


def _normalise_key(name: str) -> str:
    key = str(name).strip("/")
    if not key:
        raise ValueError("HDF block name must not be empty.")
    return key


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

        self._block_names: list[str] = []
        self.metadata = self._read_metadata_and_keys()

        if validate:
            self.validate_storage(deep=False)
            self.validate_domain()

    def _read_metadata_and_keys(self) -> SweepMetadata:
        with pd.HDFStore(self.file_path, mode="r") as store:
            self._block_names = [
                key.strip("/")
                for key in store.keys()
                if not key.startswith(_METADATA_PREFIX)
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
                        str(name)
                        for name in frame.index.names
                        if name is not None
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

    def validate_storage(self, *, deep: bool = False) -> None:
        """Validate metadata and optionally compare it to stored DataFrames."""
        self.metadata.validate()

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

        for key in self._block_names:
            frame = self.get_block(key, start=0, stop=1)
            block = self.metadata.blocks.get(key)
            if block is None:
                continue

            if list(frame.index.names) != block.ivars:
                raise ValueError(
                    f"{key!r}: stored index names differ from metadata."
                )
            if [str(name) for name in frame.columns] != block.dvars:
                raise ValueError(
                    f"{key!r}: stored columns differ from metadata."
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
