"""Chunked on-disk storage backend used by :class:`SweepManager`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ..data.model import json_default
from .hdf import (
    _data_block_keys,
    _is_legacy_block_name,
    _normalise_block_name,
    _normalise_hdf_string_frame,
)
from .persistence import _fsync_file, _replace_file


BlockMode = Literal["list", "mapping"]


class DataManager(ABC):
    """Abstract storage backend used by ``SweepManager``."""

    @abstractmethod
    def add_worker_data(
        self,
        blocks: Mapping[str, pd.DataFrame],
        metadata: Mapping[str, Any] | None,
        sequence_indices: list[int],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_results(
        self,
        include_nan: bool = True,
    ) -> list[Any] | dict[str, pd.DataFrame | None]:
        raise NotImplementedError

    @abstractmethod
    def remove_incomplete_params_from_chunks(
        self,
        incomplete_keys: list[tuple[str, ...]],
        param_col_names: list[str],
    ) -> None:
        raise NotImplementedError


class OnDiskChunkManager(DataManager):
    """Store tagged measurement blocks in compressed HDF5 chunks."""

    def __init__(
        self,
        chunk_size: int = 10,
        output_dir: str | Path = "chunks",
        resume: bool = False,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")

        self.chunk_size = int(chunk_size)
        self.output_dir = Path(output_dir)
        self.resume = bool(resume)

        if (
            not self.resume
            and self.output_dir.exists()
            and any(self.output_dir.iterdir())
        ):
            raise FileExistsError(
                "OnDiskChunkManager refuses to remove a non-empty directory. "
                "Use SweepManager for validated run-directory replacement."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.worker_count = 0
        self.df_blocks: dict[str, list[pd.DataFrame]] = defaultdict(list)
        self.pending_sequence_indices: list[int] = []
        self.block_metadata: dict[str, dict[str, Any]] = {}
        self.block_names: tuple[str, ...] | None = None
        self.run_uuid: str | None = None
        self.contract_sha256: str | None = None

        self.block_mode: BlockMode | None = self._read_existing_block_mode()
        if resume:
            self.block_names = self._read_existing_block_names()
        self.chunks_written = (
            self._get_max_existing_chunk_index() + 1
            if resume
            else 0
        )

    def _chunk_files(self) -> list[Path]:
        """Return completed chunk files and ignore stale *.tmp.h5 files."""
        indexed: list[tuple[int, Path]] = []
        for path in self.output_dir.glob("chunk_*.h5"):
            if path.name.endswith(".tmp.h5"):
                continue
            try:
                index = int(path.stem.removeprefix("chunk_"))
            except ValueError as error:
                raise RuntimeError(
                    f"Invalid final chunk filename {path.name!r}."
                ) from error
            indexed.append((index, path))
        return [path for _, path in sorted(indexed)]

    def _get_max_existing_chunk_index(self) -> int:
        indices = [
            int(file_path.stem.removeprefix("chunk_"))
            for file_path in self._chunk_files()
        ]
        return max(indices) if indices else -1

    def _read_existing_block_mode(self) -> BlockMode | None:
        chunk_files = self._chunk_files()
        if not chunk_files:
            return None

        with pd.HDFStore(chunk_files[0], mode="r") as store:
            try:
                mode = str(store.root._v_attrs.pynst_block_mode)
            except AttributeError:
                keys = [key.strip("/") for key in _data_block_keys(store)]
                mode = (
                    "list"
                    if keys and all(_is_legacy_block_name(k) for k in keys)
                    else "mapping"
                )

        if mode not in {"list", "mapping"}:
            raise ValueError(
                f"Invalid block mode {mode!r} in existing chunk."
            )
        return mode  # type: ignore[return-value]

    def _read_existing_block_names(self) -> tuple[str, ...] | None:
        chunk_files = self._chunk_files()
        if not chunk_files:
            return None
        with pd.HDFStore(chunk_files[0], mode="r") as store:
            try:
                raw = store.root._v_attrs.pynst_block_names_json
            except AttributeError:
                return None
        return tuple(str(name) for name in json.loads(str(raw)))

    def configure_run_identity(
        self,
        *,
        run_uuid: str,
        contract_sha256: str,
        block_names: Sequence[str] | None = None,
    ) -> None:
        self.run_uuid = str(run_uuid)
        self.contract_sha256 = str(contract_sha256)
        if block_names is not None:
            self.block_names = tuple(str(name) for name in block_names)

    def add_worker_data(
        self,
        blocks: Mapping[str, pd.DataFrame],
        metadata: Mapping[str, Any] | None,
        sequence_indices: list[int],
    ) -> None:
        metadata = dict(metadata or {})
        incoming_mode = metadata.get("block_mode")
        incoming_block_names = metadata.get("block_names")

        if incoming_mode is not None:
            if incoming_mode not in {"list", "mapping"}:
                raise ValueError(
                    f"Invalid incoming block mode {incoming_mode!r}."
                )
            if self.block_mode is None:
                self.block_mode = incoming_mode
            elif self.block_mode != incoming_mode:
                raise ValueError(
                    "Measurement return type changed during the sweep: "
                    f"{self.block_mode!r} -> {incoming_mode!r}."
                )

        if incoming_block_names is not None:
            names = tuple(str(name) for name in incoming_block_names)
            if self.block_names is None:
                self.block_names = names
            elif self.block_names != names:
                raise ValueError(
                    "Measurement block names/list positions changed during "
                    f"the sweep: {self.block_names!r} -> {names!r}."
                )

        incoming_block_metadata = dict(
            metadata.get("blocks", {}) or {}
        )
        for key, values in incoming_block_metadata.items():
            self.block_metadata[_normalise_block_name(key)] = dict(values)

        for block_name, frame in blocks.items():
            key = _normalise_block_name(block_name)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"Tagged block {key!r} is not a DataFrame."
                )
            self.df_blocks[key].append(frame)

        self.pending_sequence_indices.extend(
            int(index) for index in sequence_indices
        )
        self.worker_count += 1

        if self.worker_count >= self.chunk_size:
            self._flush_chunk()

    def _flush_chunk(self) -> None:
        if not self.df_blocks:
            return

        tmp_path = self.output_dir / (
            f"chunk_{self.chunks_written}.tmp.h5"
        )
        final_path = self.output_dir / (
            f"chunk_{self.chunks_written}.h5"
        )

        if final_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing chunk {final_path}"
            )

        if tmp_path.exists():
            tmp_path.unlink()

        with pd.HDFStore(
            tmp_path,
            mode="w",
            complevel=9,
            complib="blosc",
        ) as store:
            store.root._v_attrs.pynst_schema_version = 2
            store.root._v_attrs.pynst_block_mode = (
                self.block_mode or "mapping"
            )
            store.root._v_attrs.pynst_chunk_index = self.chunks_written
            store.root._v_attrs.pynst_sequence_indices_json = json.dumps(
                self.pending_sequence_indices,
                separators=(",", ":"),
            )
            store.root._v_attrs.pynst_block_names_json = json.dumps(
                list(self.block_names or tuple(self.df_blocks)),
                separators=(",", ":"),
            )
            if self.run_uuid is not None:
                store.root._v_attrs.pynst_run_uuid = self.run_uuid
            if self.contract_sha256 is not None:
                store.root._v_attrs.pynst_contract_sha256 = (
                    self.contract_sha256
                )

            for block_name, frames in self.df_blocks.items():
                combined = pd.concat(
                    frames,
                    axis=0,
                    join="outer",
                )
                combined = _normalise_hdf_string_frame(combined)
                index_columns = list(combined.index.names)

                store.put(
                    block_name,
                    combined,
                    format="table",
                    data_columns=index_columns,
                    complevel=9,
                    complib="blosc",
                    min_itemsize=128,
                )

                block_meta = self.block_metadata.get(block_name)
                if block_meta is not None:
                    store.get_storer(block_name).attrs.block_metadata_json = (
                        json.dumps(
                            block_meta,
                            default=json_default,
                            ensure_ascii=False,
                        )
                    )

        _fsync_file(tmp_path)
        _replace_file(tmp_path, final_path)

        done_indices = list(self.pending_sequence_indices)
        chunk_index = self.chunks_written
        self._on_chunk_written(
            chunk_index,
            final_path.name,
            done_indices,
        )
        self.pending_sequence_indices.clear()
        self.df_blocks.clear()
        self.worker_count = 0
        self.chunks_written += 1

    def _on_chunk_written(
        self,
        chunk_idx: int,
        chunk_filename: str,
        sequence_indices: list[int],
    ) -> None:
        """Hook replaced by ``SweepManager`` for log-file updates."""

    def finalize(self) -> None:
        if self.worker_count > 0:
            self._flush_chunk()

    def get_results(
        self,
        include_nan: bool = True,
    ) -> list[Any] | dict[str, pd.DataFrame | None]:
        """Load and concatenate all chunks block by block."""
        block_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
        block_order: list[str] = []

        for chunk_file in self._chunk_files():
            with pd.HDFStore(chunk_file, mode="r") as store:
                if self.block_mode is None:
                    try:
                        self.block_mode = str(
                            store.root._v_attrs.pynst_block_mode
                        )  # type: ignore[assignment]
                    except AttributeError:
                        pass

                for key in _data_block_keys(store):
                    block_name = key.strip("/")
                    if block_name not in block_order:
                        block_order.append(block_name)
                    block_frames[block_name].append(
                        _normalise_hdf_string_frame(store[key])
                    )

        combined: dict[str, pd.DataFrame] = {}
        for block_name in block_order:
            frame = pd.concat(
                block_frames[block_name],
                axis=0,
                join="outer",
            )
            frame = _normalise_hdf_string_frame(frame)
            if not include_nan and frame.isna().all().all():
                continue
            combined[block_name] = frame

        mode = self.block_mode
        if mode is None:
            mode = (
                "list"
                if combined
                and all(_is_legacy_block_name(k) for k in combined)
                else "mapping"
            )

        if mode == "mapping":
            names = self.block_names or tuple(combined)
            return {name: combined.get(name) for name in names}

        names_for_positions = self.block_names or tuple(combined)
        indices = [
            int(name.removeprefix("block_"))
            for name in names_for_positions
            if _is_legacy_block_name(name)
        ]
        if not indices:
            return []

        out: list[Any] = []
        for index in range(max(indices) + 1):
            key = f"block_{index}"
            if key in combined:
                out.append(combined[key])
            else:
                out.append(None)
        return out

    def remove_incomplete_params_from_chunks(
        self,
        incomplete_keys: list[tuple[str, ...]],
        param_col_names: list[str],
    ) -> None:
        """Refuse in-place mutation of committed, manifest-owned chunks."""
        raise RuntimeError(
            "In-place removal from committed chunks is no longer supported. "
            "Resume failed points or create an explicit partial merge with "
            "require_complete=False instead."
        )

__all__ = ["BlockMode", "DataManager", "OnDiskChunkManager"]
