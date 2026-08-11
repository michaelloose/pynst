"""Chunked nested-sweep execution and HDF5 storage.

The module supports two measurement-function return styles:

Legacy list mode
----------------
``return [meas_df, status_df]`` stores ``/block_0`` and ``/block_1``.

Named mapping mode
------------------
``return {"meas": meas_df, "status": status_df}`` stores ``/meas`` and
``/status``.

Both forms are normalised to an ordered mapping immediately. All internal
storage logic therefore uses one representation while legacy files and calls
remain supported.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import traceback
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .data_model import (
    BlockMetadata,
    SweepMetadata,
    VariableMetadata,
    dumps_metadata,
    json_default,
)


BlockMode = Literal["list", "mapping"]
MeasurementResult = Sequence[Any] | Mapping[str, Any]
BlockMap = dict[str, pd.DataFrame]
SWEEP_CONTRACT_VERSION = 2
SUPPORTED_SWEEP_CONTRACT_VERSIONS = {1, SWEEP_CONTRACT_VERSION}


class CriticalMeasurementError(Exception):
    """Abort the complete measurement sweep after safely flushing data."""


def _normalise_block_name(name: str) -> str:
    """Validate and normalise a user-defined HDF block name."""
    if not isinstance(name, str):
        raise TypeError("Measurement block names must be strings.")

    key = name.strip("/")
    if not key:
        raise ValueError("Measurement block names must not be empty.")
    if "/" in key:
        raise ValueError(
            f"Measurement block name {name!r} must not contain '/'."
        )
    if key.startswith("__metadata__"):
        raise ValueError(
            f"Measurement block name {name!r} uses the reserved "
            "'__metadata__' prefix."
        )
    return key


def _is_legacy_block_name(name: str) -> bool:
    key = str(name).strip("/")
    return (
        key.startswith("block_")
        and key.removeprefix("block_").isdigit()
    )


def _block_sort_key(name: str) -> tuple[int, int | str]:
    key = str(name).strip("/")
    if _is_legacy_block_name(key):
        return (0, int(key.removeprefix("block_")))
    return (1, key)


class DataManager(ABC):
    """Abstract storage backend used by ``SweepManager``."""

    @abstractmethod
    def add_worker_data(
        self,
        blocks: Mapping[str, pd.DataFrame],
        metadata: Mapping[str, Any] | None,
        param_keys: list[tuple[str, ...]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_results(
        self,
        include_nan: bool = True,
    ) -> list[Any] | dict[str, pd.DataFrame]:
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

        if not self.resume and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.worker_count = 0
        self.df_blocks: dict[str, list[pd.DataFrame]] = defaultdict(list)
        self.pending_param_keys: list[tuple[str, ...]] = []
        self.block_metadata: dict[str, dict[str, Any]] = {}

        self.block_mode: BlockMode | None = self._read_existing_block_mode()
        self.chunks_written = (
            self._get_max_existing_chunk_index() + 1
            if resume
            else 0
        )

    def _chunk_files(self) -> list[Path]:
        """Return completed chunk files and ignore stale *.tmp.h5 files."""
        indexed: list[tuple[int, Path]] = []
        for path in self.output_dir.glob("chunk_*.h5"):
            try:
                index = int(path.stem.removeprefix("chunk_"))
            except ValueError:
                continue
            indexed.append((index, path))
        return [path for _, path in sorted(indexed)]

    def _get_max_existing_chunk_index(self) -> int:
        indices: list[int] = []
        for file_path in self.output_dir.glob("chunk_*.h5"):
            try:
                indices.append(int(file_path.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return max(indices) if indices else -1

    def _read_existing_block_mode(self) -> BlockMode | None:
        chunk_files = self._chunk_files()
        if not chunk_files:
            return None

        with pd.HDFStore(chunk_files[0], mode="r") as store:
            try:
                mode = str(store.root._v_attrs.pynst_block_mode)
            except AttributeError:
                keys = [
                    key.strip("/")
                    for key in store.keys()
                    if not key.startswith("/__metadata__/")
                ]
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

    def add_worker_data(
        self,
        blocks: Mapping[str, pd.DataFrame],
        metadata: Mapping[str, Any] | None,
        param_keys: list[tuple[str, ...]],
    ) -> None:
        metadata = dict(metadata or {})
        incoming_mode = metadata.get("block_mode")

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

        self.pending_param_keys.extend(param_keys)
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

            for block_name, frames in self.df_blocks.items():
                combined = pd.concat(
                    frames,
                    axis=0,
                    join="outer",
                )
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

        os.replace(tmp_path, final_path)

        done_keys = list(self.pending_param_keys)
        self.pending_param_keys.clear()
        self.df_blocks.clear()
        self.worker_count = 0

        chunk_index = self.chunks_written
        self.chunks_written += 1
        self._on_chunk_written(
            chunk_index,
            final_path.name,
            done_keys,
        )

    def _on_chunk_written(
        self,
        chunk_idx: int,
        chunk_filename: str,
        param_keys: list[tuple[str, ...]],
    ) -> None:
        """Hook replaced by ``SweepManager`` for log-file updates."""

    def finalize(self) -> None:
        if self.worker_count > 0:
            self._flush_chunk()

    def get_results(
        self,
        include_nan: bool = True,
    ) -> list[Any] | dict[str, pd.DataFrame]:
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

                for key in store.keys():
                    if key.startswith("/__metadata__/"):
                        continue
                    block_name = key.strip("/")
                    if block_name not in block_order:
                        block_order.append(block_name)
                    block_frames[block_name].append(store[key])

        combined: dict[str, pd.DataFrame] = {}
        for block_name in block_order:
            frame = pd.concat(
                block_frames[block_name],
                axis=0,
                join="outer",
            )
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
            return combined

        indices = [
            int(name.removeprefix("block_"))
            for name in combined
            if _is_legacy_block_name(name)
        ]
        if not indices:
            return []

        out: list[Any] = []
        for index in range(max(indices) + 1):
            key = f"block_{index}"
            if key in combined:
                out.append(combined[key])
            elif include_nan:
                out.append(None)
        return out

    def remove_incomplete_params_from_chunks(
        self,
        incomplete_keys: list[tuple[str, ...]],
        param_col_names: list[str],
    ) -> None:
        """Remove rows belonging to failed parameter combinations."""
        bad_keys = {
            tuple(str(value) for value in key)
            for key in incomplete_keys
        }
        chunk_files = self._chunk_files()

        progress = tqdm(
            total=len(chunk_files),
            desc="Cleaning incomplete combinations",
            leave=True,
        )
        for chunk_file in chunk_files:
            self._filter_incomplete_in_file(
                chunk_file,
                bad_keys,
                param_col_names,
            )
            progress.update(1)
        progress.close()

    def _filter_incomplete_in_file(
        self,
        chunk_file: Path,
        incomplete_keys: set[tuple[str, ...]],
        param_col_names: list[str],
    ) -> None:
        tmp_path = chunk_file.with_suffix(".tmp.h5")
        if tmp_path.exists():
            tmp_path.unlink()

        removed_something = False

        with (
            pd.HDFStore(chunk_file, mode="r") as source,
            pd.HDFStore(tmp_path, mode="w") as target,
        ):
            try:
                target.root._v_attrs.pynst_schema_version = (
                    source.root._v_attrs.pynst_schema_version
                )
            except AttributeError:
                target.root._v_attrs.pynst_schema_version = 2

            try:
                target.root._v_attrs.pynst_block_mode = (
                    source.root._v_attrs.pynst_block_mode
                )
            except AttributeError:
                target.root._v_attrs.pynst_block_mode = (
                    self.block_mode or "mapping"
                )

            for key in source.keys():
                if key.startswith("/__metadata__/"):
                    continue

                frame = source[key]
                missing = [
                    name
                    for name in param_col_names
                    if name not in frame.index.names
                ]
                if missing:
                    raise ValueError(
                        f"{key!r} is missing sweep index levels "
                        f"{missing!r}."
                    )

                index_frame = frame.index.to_frame(index=False)
                row_keys = zip(
                    *[
                        index_frame[name].map(str)
                        for name in param_col_names
                    ]
                )
                keep_mask = np.fromiter(
                    (
                        tuple(values) not in incomplete_keys
                        for values in row_keys
                    ),
                    dtype=bool,
                    count=len(frame),
                )

                if not keep_mask.all():
                    removed_something = True

                filtered = frame.loc[keep_mask]
                if filtered.empty:
                    continue

                target.put(
                    key,
                    filtered,
                    format="table",
                    data_columns=list(filtered.index.names),
                    complevel=9,
                    complib="blosc",
                    min_itemsize=128,
                )

                try:
                    metadata_json = (
                        source.get_storer(key)
                        .attrs.block_metadata_json
                    )
                except AttributeError:
                    metadata_json = None
                if metadata_json is not None:
                    target.get_storer(
                        key
                    ).attrs.block_metadata_json = metadata_json

        if removed_something:
            os.replace(tmp_path, chunk_file)
        else:
            tmp_path.unlink()


class SweepManager:
    """Execute, resume and merge a generic nested parameter sweep.

    ``measurement_func`` may return either a legacy sequence or a named
    mapping. Mapping block names and list positions must remain stable
    throughout one sweep; mapping insertion order is irrelevant.

    With ``provide_previous_result=True``, the function is called as
    ``measurement_func(params, previous_result)``. The second argument is
    ``None`` for the first point and otherwise a defensive copy of the most
    recent valid tagged result. Resume loads that result from the exact
    complete chunk referenced by the log before invoking the callback.
    """

    def __init__(
        self,
        measurement_func: Callable[..., MeasurementResult],
        ivars: pd.MultiIndex,
        meas_name: str,
        resume: bool = True,
        output_root: str | Path | None = None,
        chunk_size: int = 10,
        param_col_names: list[str] | None = None,
        critical_callback: Callable[[Exception], None] | None = None,
        metadata: Mapping[str, Any] | None = None,
        expected_result_schema: Mapping[str, Mapping[str, Any]] | None = None,
        strict_resume_contract: bool = True,
        provide_previous_result: bool = False,
    ) -> None:
        if not isinstance(ivars, pd.MultiIndex):
            raise TypeError("ivars must be a pandas MultiIndex.")
        if not ivars.is_unique:
            raise ValueError(
                "Sweep index contains duplicate parameter combinations."
            )

        self.measurement_func = measurement_func
        self.multi_index = ivars
        self.param_col_names = list(
            param_col_names or ivars.names
        )
        if any(name is None for name in self.param_col_names):
            raise ValueError("All global sweep levels must be named.")
        if len(self.param_col_names) != ivars.nlevels:
            raise ValueError(
                "param_col_names length must match ivars.nlevels."
            )
        if len(self.param_col_names) != len(set(self.param_col_names)):
            raise ValueError("Global sweep level names must be unique.")

        self.meas_name = str(meas_name)
        self.resume = bool(resume)
        self.critical_callback = critical_callback
        self._already_run = False
        self._block_mode: BlockMode | None = None
        self._expected_result_keys: tuple[str, ...] | None = None
        self._persisted_block_names: tuple[str, ...] | None = None
        self.strict_resume_contract = bool(strict_resume_contract)
        self.provide_previous_result = bool(provide_previous_result)
        self._declared_result_schema = self._normalise_declared_result_schema(
            expected_result_schema
        )
        self._persisted_result_schema: dict[str, dict[str, Any]] | None = None

        output_root_path = (
            Path(output_root)
            if output_root is not None
            else Path.cwd()
        )
        self.output_dir = output_root_path / self.meas_name
        self.log_file = self.output_dir / "simlog.tsv"
        self.errlog_file = self.output_dir / "traceback.log"
        self.contract_file = self.output_dir / "sweep_contract.json"

        existing_resume = self.resume and self.log_file.exists()
        if existing_resume:
            self._load_and_validate_sweep_contract()

        user_metadata = dict(metadata or {})
        user_blocks = user_metadata.pop("blocks", {})
        if not isinstance(user_blocks, Mapping):
            raise TypeError(
                "metadata['blocks'] must be a mapping when provided."
            )

        self._user_block_metadata: dict[str, dict[str, Any]] = {
            _normalise_block_name(str(key)): dict(values)
            for key, values in user_blocks.items()
        }
        self.block_metadata: dict[str, BlockMetadata] = {}

        self.metadata: dict[str, Any] = {
            "measurement_name": self.meas_name,
            "created_at": datetime.now().isoformat(),
        }
        self.metadata.update(user_metadata)

        self.param_list = self._create_param_list()
        self.data_manager = OnDiskChunkManager(
            chunk_size=chunk_size,
            output_dir=self.output_dir,
            resume=self.resume,
        )
        chunk_block_mode = self.data_manager.block_mode
        if (
            self._block_mode is not None
            and chunk_block_mode is not None
            and self._block_mode != chunk_block_mode
        ):
            raise ValueError(
                "Resume contract block mode differs from the existing chunks: "
                f"contract={self._block_mode!r}, chunks={chunk_block_mode!r}."
            )
        self._block_mode = chunk_block_mode or self._block_mode

        def on_chunk_written(
            chunk_idx: int,
            chunk_filename: str,
            param_keys: list[tuple[str, ...]],
        ) -> None:
            with self.log_file.open("a", encoding="utf-8") as log:
                for key in param_keys:
                    row = "\t".join(map(str, key))
                    log.write(
                        f"{row}\t{chunk_filename}\tComplete\n"
                    )
                    self.log_dict[key] = (chunk_filename, "Complete")

        self.data_manager._on_chunk_written = on_chunk_written
        self._prepare_log()
        if not existing_resume:
            self._write_sweep_contract(result_schema=self._declared_result_schema)

    @staticmethod
    def _normalise_declared_result_schema(
        schema: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, dict[str, Any]] | None:
        if schema is None:
            return None
        result: dict[str, dict[str, Any]] = {}
        for raw_name, raw_values in schema.items():
            name = _normalise_block_name(str(raw_name))
            values = dict(raw_values)
            unknown = set(values) - {"index_names", "columns", "dtypes"}
            if unknown:
                raise ValueError(
                    f"Unknown expected-result schema fields for {name!r}: "
                    f"{sorted(unknown)}"
                )
            if "index_names" not in values or "columns" not in values:
                raise ValueError(
                    f"Expected-result schema for {name!r} requires "
                    "index_names and columns"
                )
            item = {
                "index_names": [str(value) for value in values["index_names"]],
                "columns": [str(value) for value in values["columns"]],
            }
            if "dtypes" in values:
                item["dtypes"] = [str(value) for value in values["dtypes"]]
                if len(item["dtypes"]) != len(item["columns"]):
                    raise ValueError(
                        f"Expected-result dtypes for {name!r} do not match columns"
                    )
            result[name] = item
        if not result:
            raise ValueError("expected_result_schema must not be empty")
        return result

    def _sweep_grid_contract(self) -> dict[str, Any]:
        values = [
            [json.loads(json.dumps(value, default=json_default)) for value in row]
            for row in self.multi_index.tolist()
        ]
        return {
            "parameter_names": list(self.param_col_names),
            "level_dtypes": [
                str(self.multi_index.get_level_values(index).dtype)
                for index in range(self.multi_index.nlevels)
            ],
            "combination_count": len(values),
            "combinations": values,
        }

    def _contract_payload(
        self,
        *,
        result_schema: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        if self._expected_result_keys is not None:
            block_names = list(self._expected_result_keys)
            if self._block_mode == "mapping":
                block_names.sort()
        else:
            block_names = None

        return {
            "format_name": "pynst.sweep_contract",
            "format_version": SWEEP_CONTRACT_VERSION,
            "measurement_name": self.meas_name,
            "grid": self._sweep_grid_contract(),
            "block_mode": self._block_mode,
            "block_names": block_names,
            "result_schema": None if result_schema is None else dict(result_schema),
        }

    def _write_sweep_contract(
        self,
        *,
        result_schema: Mapping[str, Mapping[str, Any]] | None,
    ) -> None:
        payload = self._contract_payload(result_schema=result_schema)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.contract_file.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                default=json_default,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.contract_file)

    def _load_and_validate_sweep_contract(self) -> None:
        if not self.contract_file.is_file():
            if self.strict_resume_contract:
                raise RuntimeError(
                    "Cannot safely resume this legacy sweep: sweep_contract.json "
                    "is missing. Start a new run or explicitly set "
                    "strict_resume_contract=False after manual review."
                )
            return
        payload = json.loads(self.contract_file.read_text(encoding="utf-8"))
        if payload.get("format_name") != "pynst.sweep_contract":
            raise ValueError("Resume contract has an unknown format")
        format_version = int(payload.get("format_version", 0))
        if format_version not in SUPPORTED_SWEEP_CONTRACT_VERSIONS:
            raise ValueError("Resume contract has an unsupported version")
        if payload.get("measurement_name") != self.meas_name:
            raise ValueError("Resume measurement name differs from its contract")
        expected_grid = payload.get("grid")
        actual_grid = self._sweep_grid_contract()
        if expected_grid != actual_grid:
            details = []
            for key in ("parameter_names", "level_dtypes", "combination_count"):
                if expected_grid.get(key) != actual_grid.get(key):
                    details.append(
                        f"{key}: stored={expected_grid.get(key)!r}, "
                        f"current={actual_grid.get(key)!r}"
                    )
            if not details:
                details.append("parameter combinations/order changed")
            raise ValueError(
                "Refusing resume because the sweep grid changed: "
                + "; ".join(details)
            )
        stored_schema = payload.get("result_schema")
        self._persisted_result_schema = (
            None
            if stored_schema is None
            else self._normalise_declared_result_schema(stored_schema)
        )
        stored_mode = payload.get("block_mode")
        if stored_mode not in {None, "list", "mapping"}:
            raise ValueError(
                f"Resume contract contains invalid block mode {stored_mode!r}"
            )
        self._block_mode = stored_mode

        raw_block_names = payload.get("block_names")
        if (
            raw_block_names is None
            and format_version == 1
            and self._persisted_result_schema is not None
        ):
            # Version 1 did not persist list slots that contained None. Its
            # best available fallback is the set of materialised blocks.
            raw_block_names = list(self._persisted_result_schema)
        elif raw_block_names is None and stored_mode is not None:
            raise ValueError(
                "Resume contract is missing block_names for a known block mode"
            )
        if raw_block_names is not None:
            if not isinstance(raw_block_names, list):
                raise ValueError("Resume contract block_names must be a list")
            normalised_names = tuple(
                _normalise_block_name(str(name)) for name in raw_block_names
            )
            if len(normalised_names) != len(set(normalised_names)):
                raise ValueError("Resume contract contains duplicate block names")
            if stored_mode == "mapping":
                normalised_names = tuple(sorted(normalised_names))
            elif stored_mode == "list" and format_version == 1:
                normalised_names = tuple(
                    sorted(normalised_names, key=_block_sort_key)
                )
            elif stored_mode == "list":
                expected_list_names = tuple(
                    f"block_{index}" for index in range(len(normalised_names))
                )
                if normalised_names != expected_list_names:
                    raise ValueError(
                        "Resume contract list block positions are invalid: "
                        f"{normalised_names!r}"
                    )
            self._persisted_block_names = normalised_names

        if (
            self._declared_result_schema is not None
            and self._persisted_result_schema is not None
            and not self._schemas_compatible(
                self._declared_result_schema,
                self._persisted_result_schema,
            )
        ):
            raise ValueError(
                "Refusing resume because the declared measurement-result "
                "schema differs from the persisted schema"
            )
        if self._persisted_block_names is not None:
            self._expected_result_keys = self._persisted_block_names

    @staticmethod
    def _schemas_compatible(
        declared: Mapping[str, Mapping[str, Any]],
        persisted: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        """Compare a user declaration with a materialised result schema.

        A declaration may intentionally omit dtypes. Persisting the observed
        dtypes must not make that same declaration invalid on resume.
        """
        if set(declared) != set(persisted):
            return False
        for name, declared_block in declared.items():
            persisted_block = persisted[name]
            for field in ("index_names", "columns"):
                if declared_block[field] != persisted_block[field]:
                    return False
            if (
                "dtypes" in declared_block
                and declared_block["dtypes"] != persisted_block.get("dtypes")
            ):
                return False
        return True

    @staticmethod
    def _result_schema_from_blocks(
        blocks: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        schema: dict[str, dict[str, Any]] = {}
        for name, value in blocks.items():
            if value is None:
                continue
            if not isinstance(value, pd.DataFrame):
                raise TypeError(
                    f"Measurement block {name!r} must be a pandas DataFrame"
                )
            schema[name] = {
                "index_names": [
                    str(index_name) if index_name is not None else f"index_{index}"
                    for index, index_name in enumerate(value.index.names)
                ],
                "columns": [str(column) for column in value.columns],
                "dtypes": [str(value[column].dtype) for column in value.columns],
            }
        return schema

    def _validate_and_persist_result_schema(
        self,
        blocks: Mapping[str, Any],
    ) -> None:
        actual = self._result_schema_from_blocks(blocks)
        expected = self._persisted_result_schema or self._declared_result_schema
        if expected is not None:
            for name, expected_block in expected.items():
                if name not in actual:
                    raise ValueError(f"Required measurement block {name!r} is missing")
                actual_block = actual[name]
                for field in ("index_names", "columns"):
                    if expected_block[field] != actual_block[field]:
                        raise ValueError(
                            f"Measurement block {name!r} changed {field}: "
                            f"expected {expected_block[field]!r}, "
                            f"got {actual_block[field]!r}"
                        )
                if (
                    "dtypes" in expected_block
                    and expected_block["dtypes"] != actual_block["dtypes"]
                ):
                    raise ValueError(
                        f"Measurement block {name!r} changed dtypes: expected "
                        f"{expected_block['dtypes']!r}, got {actual_block['dtypes']!r}"
                    )
            if set(actual) != set(expected):
                raise ValueError(
                    "Measurement result block names changed: "
                    f"expected {tuple(expected)!r}, got {tuple(actual)!r}"
                )
        if self._persisted_result_schema is None:
            self._persisted_result_schema = actual
            self._write_sweep_contract(result_schema=actual)

    def _prepare_log(self) -> None:
        if self.resume and self.log_file.exists():
            self.log_dict = self._load_log()
            incomplete = [
                key
                for key, (_, status) in self.log_dict.items()
                if status != "Complete"
            ]
            if incomplete:
                print(
                    "Resume: validating and cleaning partial data for "
                    f"{len(incomplete)} combinations."
                )
                self.data_manager.remove_incomplete_params_from_chunks(
                    incomplete,
                    self.param_col_names,
                )
                for key in incomplete:
                    self.log_dict.pop(key, None)
            else:
                print("Resume: no incomplete logged combinations found.")
            return

        self.log_dict: dict[tuple[str, ...], tuple[str, str]] = {}
        if self.log_file.exists():
            self.log_file.unlink()
        if self.errlog_file.exists() and not self.resume:
            self.errlog_file.unlink()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("w", encoding="utf-8") as log:
            log.write(
                "# "
                + json.dumps(self.param_col_names, ensure_ascii=False)
                + "\n"
            )
            log.write(
                "\t".join(self.param_col_names)
                + "\tchunk_file\tstatus\n"
            )

    def _create_param_list(self) -> list[dict[str, Any]]:
        return [
            {
                name: value
                for name, value in zip(
                    self.param_col_names,
                    values,
                )
            }
            for values in self.multi_index
        ]

    def _load_log(
        self,
    ) -> dict[tuple[str, ...], tuple[str, str]]:
        results: dict[tuple[str, ...], tuple[str, str]] = {}
        with self.log_file.open("r", encoding="utf-8") as log:
            for line in log:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                parameter_count = len(self.param_col_names)
                if len(parts) < parameter_count + 2:
                    continue
                key = tuple(parts[:parameter_count])
                results[key] = (
                    parts[parameter_count],
                    parts[parameter_count + 1],
                )
        return results

    @staticmethod
    def _copy_measurement_result(
        result: MeasurementResult | None,
    ) -> MeasurementResult | None:
        """Return a defensive copy suitable for a measurement callback."""
        return copy.deepcopy(result)

    def _result_from_tagged_blocks(
        self,
        blocks: Mapping[str, pd.DataFrame],
        mode: BlockMode,
    ) -> MeasurementResult:
        if mode == "mapping":
            names = self._expected_result_keys or tuple(sorted(blocks))
            return {
                name: blocks[name].copy(deep=True)
                for name in names
                if name in blocks
            }

        names = self._expected_result_keys
        if names is None:
            names = tuple(sorted(blocks, key=_block_sort_key))
        return [
            blocks[name].copy(deep=True) if name in blocks else None
            for name in names
        ]

    def _load_persisted_result(
        self,
        params: Mapping[str, Any],
        key: tuple[str, ...],
    ) -> MeasurementResult:
        """Load one exact, complete predecessor from its referenced chunk."""
        log_entry = self.log_dict.get(key)
        if log_entry is None or log_entry[1] != "Complete":
            raise RuntimeError(
                f"Previous result {key!r} is not marked Complete in the log"
            )

        chunk_filename = log_entry[0]
        relative_chunk = Path(chunk_filename)
        if relative_chunk.is_absolute() or relative_chunk.name != chunk_filename:
            raise RuntimeError(
                f"Invalid chunk reference {chunk_filename!r} for {key!r}"
            )
        chunk_path = (self.output_dir / relative_chunk).resolve()
        if chunk_path.parent != self.output_dir.resolve():
            raise RuntimeError(
                f"Chunk reference {chunk_filename!r} escapes the sweep directory"
            )
        if not chunk_path.is_file():
            raise FileNotFoundError(
                f"Chunk {chunk_filename!r} referenced by {key!r} is missing"
            )

        selected_blocks: BlockMap = {}
        with pd.HDFStore(chunk_path, mode="r") as store:
            try:
                chunk_mode = str(store.root._v_attrs.pynst_block_mode)
            except AttributeError:
                chunk_mode = None
            if (
                self._block_mode is not None
                and chunk_mode is not None
                and chunk_mode != self._block_mode
            ):
                raise RuntimeError(
                    "Previous-result chunk block mode differs from the resume "
                    f"contract: {chunk_mode!r} != {self._block_mode!r}"
                )

            available_names = {
                hdf_key.strip("/")
                for hdf_key in store.keys()
                if not hdf_key.startswith("/__metadata__/")
            }
            schema = (
                self._persisted_result_schema
                or self._declared_result_schema
            )
            expected_materialised = (
                set(schema) if schema is not None else available_names
            )
            if schema is not None and available_names != expected_materialised:
                raise RuntimeError(
                    f"Chunk {chunk_filename!r} contains blocks "
                    f"{sorted(available_names)!r}, expected "
                    f"{sorted(expected_materialised)!r}"
                )

            for block_name in sorted(
                expected_materialised,
                key=_block_sort_key,
            ):
                frame = store[block_name]
                missing_levels = [
                    name
                    for name in self.param_col_names
                    if name not in frame.index.names
                ]
                if missing_levels:
                    raise RuntimeError(
                        f"Block {block_name!r} in {chunk_filename!r} is missing "
                        f"sweep index levels {missing_levels!r}"
                    )

                index_frame = frame.index.to_frame(index=False)
                mask = np.ones(len(frame), dtype=bool)
                for name in self.param_col_names:
                    value = params[name]
                    if pd.isna(value):
                        equal = index_frame[name].isna()
                    else:
                        equal = index_frame[name].eq(value).fillna(False)
                    mask &= np.asarray(equal, dtype=bool)
                selected = frame.loc[mask]
                if selected.empty:
                    raise RuntimeError(
                        f"Complete result {key!r} has no rows in block "
                        f"{block_name!r} of {chunk_filename!r}"
                    )
                selected_blocks[block_name] = selected.copy(deep=True)

        mode = self._block_mode
        if mode is None:
            mode = (
                "list"
                if selected_blocks
                and all(_is_legacy_block_name(name) for name in selected_blocks)
                else "mapping"
            )
        return self._result_from_tagged_blocks(selected_blocks, mode)

    def _normalise_measurement_result(
        self,
        result: MeasurementResult,
    ) -> tuple[BlockMode, dict[str, Any]]:
        if isinstance(result, Mapping):
            mode: BlockMode = "mapping"
            blocks: dict[str, Any] = {}
            for raw_name, value in result.items():
                name = _normalise_block_name(raw_name)
                if name in blocks:
                    raise ValueError(
                        f"Duplicate normalised block name {name!r}."
                    )
                blocks[name] = value
        elif isinstance(result, Sequence) and not isinstance(
            result,
            (str, bytes, bytearray),
        ):
            mode = "list"
            blocks = {
                f"block_{index}": value
                for index, value in enumerate(result)
            }
        else:
            raise TypeError(
                "measurement_func must return a list/tuple or a mapping "
                "of block names to DataFrames."
            )

        keys = tuple(blocks)
        comparable_keys = (
            tuple(sorted(keys)) if mode == "mapping" else keys
        )
        if self._block_mode is None:
            self._block_mode = mode
        elif self._block_mode != mode:
            raise ValueError(
                "measurement_func changed its return container type "
                f"from {self._block_mode!r} to {mode!r}."
            )

        if self._expected_result_keys is None:
            self._expected_result_keys = comparable_keys
        elif self._expected_result_keys != comparable_keys:
            structure = (
                "block names" if mode == "mapping" else "block positions"
            )
            raise ValueError(
                f"measurement_func changed its {structure}. "
                f"Expected {self._expected_result_keys!r}, "
                f"got {comparable_keys!r}."
            )

        return mode, blocks

    @staticmethod
    def _normalise_local_index_names(
        frame: pd.DataFrame,
        block_name: str,
    ) -> list[str]:
        names = [
            str(name) if name is not None else f"index_{index}"
            for index, name in enumerate(frame.index.names)
        ]
        if len(names) != len(set(names)):
            raise ValueError(
                f"{block_name!r} contains duplicate local index names: "
                f"{names!r}."
            )
        return names

    def _tag_with_params(
        self,
        blocks: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> BlockMap:
        """Prepend global sweep ivars to each local measurement block."""
        tagged: BlockMap = {}

        for block_name, value in blocks.items():
            if value is None:
                # None remains useful in legacy lists as an intentionally
                # absent block position, but nothing is written for it.
                continue
            if not isinstance(value, pd.DataFrame):
                raise TypeError(
                    f"Measurement block {block_name!r} must be a "
                    f"DataFrame or None, got {type(value).__name__}."
                )

            local_ivars = self._normalise_local_index_names(
                value,
                block_name,
            )
            dvars = [str(column) for column in value.columns]
            if len(dvars) != len(set(dvars)):
                raise ValueError(
                    f"{block_name!r} contains duplicate dvar names."
                )
            if any(
                not isinstance(column, str)
                for column in value.columns
            ):
                raise TypeError(
                    f"{block_name!r}: DataFrame columns must be strings."
                )

            conflicts = (
                set(self.param_col_names) & set(local_ivars)
                | set(self.param_col_names) & set(dvars)
                | set(local_ivars) & set(dvars)
            )
            if conflicts:
                raise ValueError(
                    f"{block_name!r} reuses variable names across sweep "
                    f"ivars, local ivars or dvars: {sorted(conflicts)!r}."
                )

            frame = value.copy(deep=False)
            frame.index = frame.index.set_names(local_ivars)
            flat = frame.reset_index()

            # Global sweep values are added after reset_index so no dependent
            # variable can be overwritten silently.
            for name in self.param_col_names:
                flat[name] = params[name]

            ivars = list(self.param_col_names) + local_ivars
            tagged_frame = flat.set_index(ivars)

            if not tagged_frame.index.is_unique:
                raise ValueError(
                    f"{block_name!r} contains duplicate combinations of "
                    "global and local independent variables."
                )

            self._register_block_metadata(
                block_name,
                tagged_frame,
                local_ivars,
            )
            tagged[block_name] = tagged_frame

        return tagged

    def _register_block_metadata(
        self,
        block_name: str,
        frame: pd.DataFrame,
        local_ivars: list[str],
    ) -> None:
        """Create or validate ADS-like ivar/dvar metadata for one block."""
        user_values = dict(
            self._user_block_metadata.get(block_name, {})
        )
        user_variables = dict(
            user_values.pop("variables", {}) or {}
        )

        ivars = list(frame.index.names)
        dvars = [str(column) for column in frame.columns]
        structural = {
            "hdf_key": block_name,
            "ivars": ivars,
            "sweep_ivars": list(self.param_col_names),
            "local_ivars": list(local_ivars),
            "dvars": dvars,
        }

        # Hand-written structural fields are allowed only when they agree with
        # the actual DataFrame. This catches stale metadata early.
        for field, actual in structural.items():
            if field in user_values and user_values[field] != actual:
                raise ValueError(
                    f"User metadata for {block_name!r} defines "
                    f"{field}={user_values[field]!r}, but the stored "
                    f"structure is {actual!r}."
                )
            user_values[field] = actual

        variable_values: dict[str, dict[str, Any]] = {
            str(name): dict(values)
            for name, values in user_variables.items()
        }

        for name in ivars:
            values = variable_values.setdefault(name, {})
            values.setdefault(
                "dtype",
                str(frame.index.get_level_values(name).dtype),
            )

        for name in dvars:
            values = variable_values.setdefault(name, {})
            values.setdefault("dtype", str(frame[name].dtype))

        user_values["variables"] = variable_values
        candidate = BlockMetadata.from_dict(
            user_values,
            default_key=block_name,
        )

        existing = self.block_metadata.get(block_name)
        if existing is not None:
            for field in (
                "ivars",
                "sweep_ivars",
                "local_ivars",
                "dvars",
            ):
                if getattr(existing, field) != getattr(candidate, field):
                    raise ValueError(
                        f"Structure of {block_name!r} changed during "
                        f"the sweep: {field} differs."
                    )

            # Dtypes may legitimately be promoted by pandas. Keep the first
            # schema but merge newly supplied optional semantic metadata.
            for name, variable in candidate.variables.items():
                existing.variables.setdefault(name, variable)
            return

        self.block_metadata[block_name] = candidate

    def _block_metadata_dict(self) -> dict[str, dict[str, Any]]:
        return {
            name: metadata.to_dict()
            for name, metadata in self.block_metadata.items()
        }

    def run(self) -> None:
        if self._already_run:
            raise RuntimeError(
                "SweepManager.run() was already called. Create a new "
                "manager to execute the sweep again."
            )
        self._already_run = True

        completed_count = sum(
            1
            for params in self.param_list
            if self.log_dict.get(
                tuple(str(params[name]) for name in self.param_col_names),
                (None, None),
            )[1] == "Complete"
        )
        progress = tqdm(
            total=len(self.param_list),
            initial=completed_count,
            desc="Sweep",
            leave=True,
        )

        previous_result: MeasurementResult | None = None
        previous_reference: tuple[
            dict[str, Any], tuple[str, ...]
        ] | None = None

        for params in self.param_list:
            key = tuple(
                str(params[name])
                for name in self.param_col_names
            )
            if (
                key in self.log_dict
                and self.log_dict[key][1] == "Complete"
            ):
                if self.provide_previous_result:
                    # Defer disk I/O until a later point really needs this
                    # predecessor. A fully complete resume performs no reads.
                    previous_result = None
                    previous_reference = (dict(params), key)
                continue

            description = " | ".join(
                f"{name}={value}"
                for name, value in params.items()
            )
            progress.set_description(description)

            try:
                if (
                    self.provide_previous_result
                    and previous_reference is not None
                ):
                    predecessor_params, predecessor_key = previous_reference
                    try:
                        previous_result = self._load_persisted_result(
                            predecessor_params,
                            predecessor_key,
                        )
                    except Exception as error:
                        raise CriticalMeasurementError(
                            "Cannot safely load the previous complete result "
                            "before the next measurement; aborting before the "
                            f"hardware callback: {error}"
                        ) from error
                    previous_reference = None

                if self.provide_previous_result:
                    raw_result = self.measurement_func(
                        params,
                        self._copy_measurement_result(previous_result),
                    )
                else:
                    raw_result = self.measurement_func(params)
                try:
                    mode, blocks = self._normalise_measurement_result(
                        raw_result
                    )
                    self._validate_and_persist_result_schema(blocks)
                    tagged = self._tag_with_params(blocks, params)
                except Exception as error:
                    raise CriticalMeasurementError(
                        "Measurement result normalization/tagging failed "
                        "after the measurement function returned; aborting "
                        f"the sweep: {error}"
                    ) from error

                self.data_manager.add_worker_data(
                    tagged,
                    metadata={
                        "block_mode": mode,
                        "blocks": self._block_metadata_dict(),
                    },
                    param_keys=[key],
                )
                if self.provide_previous_result:
                    previous_result = self._result_from_tagged_blocks(
                        tagged,
                        mode,
                    )
                    previous_reference = None

            except CriticalMeasurementError as error:
                progress.write(f"Critical error: {error}")
                self.data_manager.finalize()
                self._write_exception(
                    "Critical Exception",
                    description,
                )
                if self.critical_callback:
                    self.critical_callback(error)
                progress.close()
                return

            except Exception as error:
                progress.write(
                    "Exception occurred, measurement skipped: "
                    f"{error}"
                )
                with self.log_file.open("a", encoding="utf-8") as log:
                    log.write(
                        "\t".join(key)
                        + "\tNA\tFailed\n"
                    )
                self.log_dict[key] = ("NA", "Failed")
                self._write_exception("Exception", description)

            progress.update(1)

        progress.close()
        self.data_manager.finalize()
        print("Measurement run completed.")

    def _write_exception(
        self,
        heading: str,
        description: str,
    ) -> None:
        with self.errlog_file.open("a", encoding="utf-8") as error_log:
            error_log.write(
                f"--- {heading} at {description}; "
                f"{datetime.now().isoformat()} ---\n"
            )
            traceback.print_exc(file=error_log)
            error_log.write("\n")

    def get_results(
        self,
        include_nan: bool = True,
    ) -> list[Any] | dict[str, pd.DataFrame]:
        return self.data_manager.get_results(
            include_nan=include_nan
        )

    def update_metadata(
        self,
        values: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Update global or optional per-block semantic metadata."""
        incoming = dict(values or {})
        incoming.update(kwargs)

        block_updates = incoming.pop("blocks", None)
        self.metadata.update(incoming)

        if block_updates is None:
            return
        if not isinstance(block_updates, Mapping):
            raise TypeError("metadata['blocks'] must be a mapping.")

        for raw_name, raw_values in block_updates.items():
            block_name = _normalise_block_name(str(raw_name))
            values = dict(raw_values)

            user = dict(
                self._user_block_metadata.get(block_name, {})
            )
            user_variables = dict(user.pop("variables", {}) or {})
            new_variables = dict(values.pop("variables", {}) or {})

            user.update(values)
            for variable_name, variable_values in new_variables.items():
                merged_variable = dict(
                    user_variables.get(variable_name, {})
                )
                merged_variable.update(dict(variable_values))
                user_variables[variable_name] = merged_variable

            if user_variables:
                user["variables"] = user_variables
            self._user_block_metadata[block_name] = user

            # If the block has already been observed, apply semantic updates
            # immediately without changing structural fields.
            existing = self.block_metadata.get(block_name)
            if existing is not None:
                for field in (
                    "required",
                    "default_load",
                    "description",
                ):
                    if field in values:
                        setattr(existing, field, values[field])

                structural_fields = {
                    "schema_version",
                    "hdf_key",
                    "ivars",
                    "sweep_ivars",
                    "local_ivars",
                    "dvars",
                    "required",
                    "default_load",
                    "description",
                }
                existing.extra.update(
                    {
                        key: value
                        for key, value in values.items()
                        if key not in structural_fields
                    }
                )

                for variable_name, variable_values in new_variables.items():
                    current = existing.variables.get(
                        variable_name,
                        VariableMetadata(),
                    )
                    merged = current.to_dict()
                    merged.update(dict(variable_values))
                    existing.variables[variable_name] = (
                        VariableMetadata.from_dict(merged)
                    )

    def _ensure_block_metadata_from_chunks(self) -> None:
        """Recover schemas when a resumed run executes no new measurement."""
        chunk_files = self.data_manager._chunk_files()

        for chunk_file in chunk_files:
            with pd.HDFStore(chunk_file, mode="r") as store:
                if self._block_mode is None:
                    try:
                        self._block_mode = str(
                            store.root._v_attrs.pynst_block_mode
                        )  # type: ignore[assignment]
                    except AttributeError:
                        pass

                for key in store.keys():
                    if key.startswith("/__metadata__/"):
                        continue

                    block_name = key.strip("/")
                    if block_name in self.block_metadata:
                        continue

                    try:
                        metadata_json = (
                            store.get_storer(key)
                            .attrs.block_metadata_json
                        )
                    except AttributeError:
                        metadata_json = None

                    if metadata_json:
                        self.block_metadata[block_name] = (
                            BlockMetadata.from_dict(
                                json.loads(metadata_json),
                                default_key=block_name,
                            )
                        )
                        continue

                    try:
                        sample = store.select(
                            key,
                            start=0,
                            stop=1,
                        )
                    except (
                        TypeError,
                        ValueError,
                        NotImplementedError,
                    ):
                        sample = store[key].head(1)

                    local_ivars = [
                        name
                        for name in sample.index.names
                        if name not in self.param_col_names
                    ]
                    self._register_block_metadata(
                        block_name,
                        sample,
                        local_ivars,
                    )

    def _build_sweep_metadata(self) -> SweepMetadata:
        self._ensure_block_metadata_from_chunks()

        reserved = {
            "schema_version",
            "measurement_name",
            "nested_sweep_levels",
            "blocks",
            "created_at",
            "merged_at",
            "resume_enabled",
        }
        extra = {
            key: value
            for key, value in self.metadata.items()
            if key not in reserved
        }

        metadata = SweepMetadata(
            measurement_name=self.meas_name,
            nested_sweep_levels=list(self.param_col_names),
            blocks=dict(self.block_metadata),
            created_at=self.metadata.get("created_at"),
            merged_at=datetime.now().isoformat(),
            resume_enabled=self.resume,
            schema_version=2,
            extra=extra,
        )
        metadata.validate()
        return metadata

    def _write_metadata(self, store: pd.HDFStore) -> None:
        metadata = self._build_sweep_metadata()

        # Direct block attributes make a block self-describing. The central
        # config remains the authoritative complete file schema.
        for key in store.keys():
            if key.startswith("/__metadata__/"):
                continue
            block_name = key.strip("/")
            block = metadata.blocks.get(block_name)
            if block is None:
                continue
            store.get_storer(key).attrs.block_metadata_json = json.dumps(
                block.to_dict(),
                default=json_default,
                ensure_ascii=False,
            )

        config = pd.DataFrame(
            {"json": [dumps_metadata(metadata)]}
        )
        store.put(
            "/__metadata__/config",
            config,
            format="fixed",
        )

        if self.log_file.exists():
            log_frame = pd.read_csv(
                self.log_file,
                sep="\t",
                comment="#",
                dtype=str,
                keep_default_na=False,
            )
            store.put(
                "/__metadata__/log",
                log_frame,
                format="table",
                data_columns=True,
            )

        if self.errlog_file.exists():
            text = self.errlog_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
            if text:
                store.put(
                    "/__metadata__/traceback",
                    pd.DataFrame({"text": [text]}),
                    format="fixed",
                )

        store.root._v_attrs.pynst_schema_version = 2
        store.root._v_attrs.pynst_block_mode = (
            self._block_mode
            or self.data_manager.block_mode
            or "mapping"
        )

    def _prepare_merge_target(
        self,
        merged_file: str | Path,
        *,
        overwrite: bool,
    ) -> tuple[Path, Path]:
        target = Path(merged_file).expanduser().resolve()
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Output file {target} already exists. Set "
                "force_merge_into_existing=True to rebuild it."
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            target.name + ".pynst_tmp"
        )
        if temporary.exists():
            temporary.unlink()
        return target, temporary

    def partial_merge(
        self,
        merged_file: str | Path,
        remove_chunks: bool = False,
        force_merge_into_existing: bool = False,
        drop_columns: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Stream chunks into one table-format HDF file.

        The output is built in a temporary file and atomically moved into
        place. ``force_merge_into_existing=True`` rebuilds an existing target;
        it does not append duplicate chunks to it. ``drop_columns`` projects
        named blocks to a stable schema while reading without modifying the
        source chunks. Block names may be supplied with or without a leading
        slash, and columns absent from a block are ignored.
        """
        self.data_manager.finalize()
        chunk_files = self.data_manager._chunk_files()
        if not chunk_files:
            raise FileNotFoundError("No chunk files found.")

        target, temporary = self._prepare_merge_target(
            merged_file,
            overwrite=force_merge_into_existing,
        )
        projected_columns = {
            "/" + block_name.strip("/"): tuple(columns)
            for block_name, columns in (drop_columns or {}).items()
        }

        progress = tqdm(
            total=len(chunk_files),
            desc="Merging chunks",
            leave=True,
        )
        try:
            with pd.HDFStore(
                temporary,
                mode="w",
                complevel=9,
                complib="blosc",
            ) as output:
                for chunk_file in chunk_files:
                    with pd.HDFStore(
                        chunk_file,
                        mode="r",
                    ) as source:
                        for key in source.keys():
                            if key.startswith("/__metadata__/"):
                                continue
                            frame = source[key]
                            columns_to_drop = projected_columns.get(key, ())
                            if columns_to_drop:
                                frame = frame.drop(
                                    columns=list(columns_to_drop),
                                    errors="ignore",
                                )
                            output.append(
                                key,
                                frame,
                                format="table",
                                data_columns=list(frame.index.names),
                                complevel=9,
                                complib="blosc",
                                min_itemsize=128,
                            )
                    progress.update(1)

                self._write_metadata(output)

            os.replace(temporary, target)

        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        finally:
            progress.close()

        if remove_chunks:
            for chunk_file in chunk_files:
                chunk_file.unlink()

        print(
            f"Partial merge done -> {target}, "
            f"remove_chunks={remove_chunks}"
        )

    def merge(
        self,
        merged_file: str | Path,
        remove_chunks: bool = False,
        *,
        overwrite: bool = False,
    ) -> None:
        """Merge each complete block in memory and write fixed-format HDF.

        Unlike the historical implementation, this method preserves the full
        MultiIndex by never using ``ignore_index=True``.
        """
        self.data_manager.finalize()
        chunk_files = self.data_manager._chunk_files()
        if not chunk_files:
            raise FileNotFoundError("No chunk files found.")

        target, temporary = self._prepare_merge_target(
            merged_file,
            overwrite=overwrite,
        )

        block_names: list[str] = []
        for chunk_file in chunk_files:
            with pd.HDFStore(chunk_file, mode="r") as store:
                for key in store.keys():
                    if key.startswith("/__metadata__/"):
                        continue
                    name = key.strip("/")
                    if name not in block_names:
                        block_names.append(name)

        progress = tqdm(
            block_names,
            desc="Merging by block",
            leave=True,
        )
        try:
            with pd.HDFStore(temporary, mode="w") as output:
                for block_name in progress:
                    frames: list[pd.DataFrame] = []
                    hdf_key = f"/{block_name}"

                    for chunk_file in chunk_files:
                        with pd.HDFStore(
                            chunk_file,
                            mode="r",
                        ) as source:
                            if hdf_key in source.keys():
                                frames.append(source[hdf_key])

                    if not frames:
                        continue

                    merged = pd.concat(
                        frames,
                        axis=0,
                        join="outer",
                    ).sort_index()

                    output.put(
                        block_name,
                        merged,
                        format="fixed",
                    )
                    del merged, frames
                    gc.collect()

                self._write_metadata(output)

            os.replace(temporary, target)

        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        finally:
            progress.close()

        if remove_chunks:
            for chunk_file in chunk_files:
                chunk_file.unlink()

        print(f"Optimized merge done -> {target}")

    @staticmethod
    def read_merged_metadata(
        file_path: str | Path,
    ) -> SweepMetadata:
        with pd.HDFStore(file_path, mode="r") as store:
            key = "/__metadata__/config"
            if key not in store.keys():
                raise KeyError(
                    f"Merged HDF file does not contain {key!r}."
                )
            values = json.loads(store[key].iloc[0]["json"])
        return SweepMetadata.from_dict(values)

    @staticmethod
    def read_merged_log(
        file_path: str | Path,
    ) -> pd.DataFrame:
        with pd.HDFStore(file_path, mode="r") as store:
            key = "/__metadata__/log"
            if key not in store.keys():
                return pd.DataFrame()
            return store[key]

    @staticmethod
    def read_merged_block_metadata(
        file_path: str | Path,
        block_name: str,
    ) -> BlockMetadata:
        key = _normalise_block_name(block_name)
        with pd.HDFStore(file_path, mode="r") as store:
            hdf_key = f"/{key}"
            if hdf_key not in store.keys():
                raise KeyError(
                    f"Merged HDF file does not contain {hdf_key!r}."
                )
            try:
                metadata_json = (
                    store.get_storer(hdf_key)
                    .attrs.block_metadata_json
                )
            except AttributeError as exc:
                raise KeyError(
                    f"Block {key!r} contains no block metadata."
                ) from exc

        return BlockMetadata.from_dict(
            json.loads(metadata_json),
            default_key=key,
        )
