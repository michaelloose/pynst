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

import copy
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from typing import Any, Callable, Literal
import uuid
import warnings
import weakref

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..data.model import (
    BlockMetadata,
    SweepMetadata,
    VariableMetadata,
    dumps_metadata,
    json_default,
)
from ..storage.chunks import BlockMode, DataManager, OnDiskChunkManager
from ..storage.hdf import (
    _block_sort_key,
    _canonical_index_value,
    _data_block_keys,
    _dtype_spec,
    _hdf_object_dtype_issue,
    _is_legacy_block_name,
    _normalise_block_name,
)
from ..storage.persistence import (
    _RunLock,
    _atomic_write_json,
    _atomic_write_text,
    _fsync_file,
    _json_text,
    _lock_identity,
    _replace_file,
    _sha256_file,
)
from .errors import CriticalMeasurementError, StorageCommitError


MergeStrategy = Literal["fixed", "streaming"]
MeasurementResult = Sequence[Any] | Mapping[str, Any]
BlockMap = dict[str, pd.DataFrame]
SWEEP_CONTRACT_VERSION = 3
SUPPORTED_SWEEP_CONTRACT_VERSIONS = {1, 2, SWEEP_CONTRACT_VERSION}
RUN_MANIFEST_VERSION = 1
_FIXED_MERGE_MEMORY_FACTOR = 4.0
_FIXED_MERGE_BASE_OVERHEAD_BYTES = 256 * 1024**2
_FIXED_MERGE_INDEX_OVERHEAD_PER_ROW = 256
_FIXED_MERGE_MIN_RESERVE_BYTES = 512 * 1024**2
_FIXED_MERGE_RESERVE_FRACTION = 0.15
_FIXED_MERGE_SCAN_ROWS = 100_000
_STREAMING_MERGE_BATCH_ROWS = 100_000
_CHUNK_VALIDATION_BATCH_ROWS = 100_000


def _available_memory_bytes() -> int | None:
    """Return currently available physical memory using only the stdlib."""
    if os.name == "nt":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        try:
            success = ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)
            )
        except (AttributeError, OSError):
            return None
        return int(status.ullAvailPhys) if success else None

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if page_size <= 0 or available_pages <= 0:
        return None
    return page_size * available_pages


def _format_bytes(value: int) -> str:
    amount = float(max(0, value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")

class SweepManager:
    """Execute, resume and merge a generic nested parameter sweep.

    ``measurement_func`` may return either a legacy sequence or a named
    mapping. Mapping block names and list positions must remain stable
    throughout one sweep; mapping insertion order is irrelevant.

    With ``provide_previous_result=True``, the function is called as
    ``measurement_func(params, previous_result)``. The second argument is
    ``None`` for the first point and otherwise a defensive copy of the most
    recent valid tagged result. Resume loads that result from the exact
    complete chunk referenced by the run manifest before invoking the callback.
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
        if len(ivars) == 0:
            raise ValueError("Sweep index must contain at least one combination.")
        if not ivars.is_unique:
            raise ValueError(
                "Sweep index contains duplicate parameter combinations."
            )

        self.measurement_func = measurement_func
        self.multi_index = ivars
        self.param_col_names = list(param_col_names or ivars.names)
        if any(name is None for name in self.param_col_names):
            raise ValueError("All global sweep levels must be named.")
        if any(not isinstance(name, str) for name in self.param_col_names):
            raise TypeError("All global sweep level names must be strings.")
        if any(
            any(character in name for character in "\t\r\n")
            for name in self.param_col_names
        ):
            raise ValueError(
                "Global sweep level names must not contain tabs or newlines."
            )
        if len(self.param_col_names) != ivars.nlevels:
            raise ValueError(
                "param_col_names length must match ivars.nlevels."
            )
        if len(self.param_col_names) != len(set(self.param_col_names)):
            raise ValueError("Global sweep level names must be unique.")
        unsupported_grid_dtypes = [
            f"{self.param_col_names[level]}="
            f"{ivars.get_level_values(level).dtype}"
            for level in range(ivars.nlevels)
            if isinstance(
                ivars.get_level_values(level).dtype,
                pd.api.extensions.ExtensionDtype,
            )
            and not isinstance(
                ivars.get_level_values(level).dtype,
                pd.CategoricalDtype,
            )
        ]
        if unsupported_grid_dtypes:
            raise TypeError(
                "pandas HDF storage does not support these sweep-level "
                "extension dtypes; convert them before creating the manager: "
                f"{unsupported_grid_dtypes!r}."
            )
        unsupported_grid_indices = [
            f"{self.param_col_names[level]}=uint64"
            for level in range(ivars.nlevels)
            if ivars.get_level_values(level).dtype == np.dtype("uint64")
        ]
        if unsupported_grid_indices:
            raise TypeError(
                "pandas HDF table indices do not support uint64 sweep "
                "levels; use a checked int64 representation instead: "
                f"{unsupported_grid_indices!r}."
            )
        unsupported_grid_objects = [
            issue
            for level in range(ivars.nlevels)
            if (
                issue := _hdf_object_dtype_issue(
                    ivars.get_level_values(level),
                    self.param_col_names[level],
                )
            )
            is not None
        ]
        if unsupported_grid_objects:
            raise TypeError(
                "pandas HDF storage requires homogeneous numeric, temporal, "
                "categorical or string sweep levels; mixed/non-string object "
                f"levels are unsupported: {unsupported_grid_objects!r}."
            )
        try:
            _json_text({"grid": self._sweep_grid_contract()})
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Sweep grid values must be finite and JSON-serializable by "
                "pynst.data_model.json_default."
            ) from error

        self.meas_name = self._validate_measurement_name(meas_name)
        self.resume = bool(resume)
        self.critical_callback = critical_callback
        self._already_run = False
        self._storage_compromised = False
        self._block_mode: BlockMode | None = None
        self._expected_result_keys: tuple[str, ...] | None = None
        self._persisted_block_names: tuple[str, ...] | None = None
        self.strict_resume_contract = bool(strict_resume_contract)
        self.provide_previous_result = bool(provide_previous_result)
        self._declared_result_schema = self._normalise_declared_result_schema(
            expected_result_schema
        )
        self._persisted_result_schema: dict[str, dict[str, Any]] | None = None
        self._loaded_contract_version = SWEEP_CONTRACT_VERSION
        self._contract_has_run_uuid = False
        self.run_uuid = str(uuid.uuid4())
        self.contract_sha256 = ""
        self._manifest: dict[str, Any] = {}

        output_root_path = Path(
            output_root if output_root is not None else Path.cwd()
        ).expanduser().resolve()
        output_root_path.mkdir(parents=True, exist_ok=True)
        self.output_dir = (output_root_path / self.meas_name).resolve()
        if self.output_dir.parent != output_root_path:
            raise ValueError("Measurement path escapes output_root.")

        lock_digest = hashlib.sha256(
            _lock_identity(self.output_dir).encode("utf-8")
        ).hexdigest()
        self._run_lock = _RunLock(
            output_root_path / ".pynst_locks" / f"{lock_digest}.lock"
        )
        self._run_lock.acquire()

        self.log_file = self.output_dir / "simlog.tsv"
        self.errlog_file = self.output_dir / "traceback.log"
        self.contract_file = self.output_dir / "sweep_contract.json"
        self.manifest_file = self.output_dir / "run_manifest.json"

        try:
            if not self.resume and self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            existing_artifacts = any(
                (
                    self.contract_file.exists(),
                    self.log_file.exists(),
                    self.manifest_file.exists(),
                    any(
                        not path.name.endswith(".tmp.h5")
                        for path in self.output_dir.glob("chunk_*.h5")
                    ),
                )
            )
            if self.resume and not existing_artifacts:
                entries = list(self.output_dir.iterdir())
                control_temps = {
                    self.contract_file.with_name(
                        self.contract_file.name + ".tmp"
                    ),
                    self.manifest_file.with_name(
                        self.manifest_file.name + ".tmp"
                    ),
                    self.log_file.with_name(self.log_file.name + ".tmp"),
                }
                known_uncommitted = [
                    path
                    for path in entries
                    if path in control_temps
                    or (
                        path.is_file()
                        and path.name.startswith("chunk_")
                        and path.name.endswith(".tmp.h5")
                    )
                ]
                if entries and len(known_uncommitted) == len(entries):
                    # No measurement can be authoritative without a final
                    # contract/manifest/chunk. These are known torn-write
                    # remnants and can be retired before starting anew.
                    for path in known_uncommitted:
                        path.unlink()
            existing_resume = self.resume and existing_artifacts
            if existing_resume:
                self._load_and_validate_sweep_contract()
            elif self.resume and any(self.output_dir.iterdir()):
                raise RuntimeError(
                    "Output directory contains unknown files but no valid "
                    "PyNST run artifacts."
                )

            incoming_metadata = dict(metadata or {})
            incoming_blocks = incoming_metadata.pop("blocks", {})
            if not isinstance(incoming_blocks, Mapping):
                raise TypeError(
                    "metadata['blocks'] must be a mapping when provided."
                )

            stored_manifest = (
                self._read_run_manifest()
                if self.manifest_file.is_file()
                else None
            )
            if stored_manifest is not None:
                if stored_manifest.get("state") == "archived":
                    raise RuntimeError(
                        "This PyNST run is archived and its chunks were retired; "
                        "it cannot be resumed."
                    )
                manifest_uuid = str(stored_manifest.get("run_uuid", ""))
                if (
                    manifest_uuid
                    and self._contract_has_run_uuid
                    and manifest_uuid != self.run_uuid
                ):
                    raise RuntimeError(
                        "Run manifest UUID differs from the sweep contract."
                    )
                self.run_uuid = manifest_uuid or self.run_uuid
                stored_metadata = dict(stored_manifest.get("metadata", {}))
                stored_blocks = dict(
                    stored_manifest.get("user_block_metadata", {})
                )
            else:
                stored_metadata = {}
                stored_blocks = {}

            self.metadata = stored_metadata or {
                "measurement_name": self.meas_name,
                "created_at": datetime.now().isoformat(),
            }
            self.metadata["measurement_name"] = self.meas_name
            self.metadata.setdefault("created_at", datetime.now().isoformat())
            self.metadata.update(incoming_metadata)

            stored_blocks.update(
                {
                    _normalise_block_name(str(key)): dict(values)
                    for key, values in incoming_blocks.items()
                }
            )
            self._user_block_metadata = stored_blocks
            self.block_metadata: dict[str, BlockMetadata] = {}
            self.param_list = self._create_param_list()

            self.data_manager = OnDiskChunkManager(
                chunk_size=chunk_size,
                output_dir=self.output_dir,
                resume=existing_resume,
            )
            chunk_block_mode = self.data_manager.block_mode
            if (
                self._block_mode is not None
                and chunk_block_mode is not None
                and self._block_mode != chunk_block_mode
            ):
                raise ValueError(
                    "Resume contract block mode differs from existing chunks: "
                    f"contract={self._block_mode!r}, "
                    f"chunks={chunk_block_mode!r}."
                )
            self._block_mode = chunk_block_mode or self._block_mode
            # Avoid a strong ``self -> data_manager -> bound method -> self``
            # cycle.  Apart from making destruction needlessly delayed, that
            # cycle could keep the OS run lock alive after a temporary manager
            # object had already gone out of scope.
            owner_ref = weakref.ref(self)

            def commit_chunk(
                chunk_idx: int,
                chunk_filename: str,
                sequence_indices: list[int],
            ) -> None:
                owner = owner_ref()
                if owner is None:
                    raise StorageCommitError(
                        "SweepManager vanished before the chunk commit."
                    )
                owner._on_chunk_written(
                    chunk_idx,
                    chunk_filename,
                    sequence_indices,
                )

            self.data_manager._on_chunk_written = commit_chunk

            if not existing_resume:
                self._write_sweep_contract(
                    result_schema=self._declared_result_schema
                )
                self._manifest = self._new_run_manifest()
                self._write_run_manifest()
                self.log_dict: dict[int, tuple[str, str]] = {}
                self._write_log_from_state()
            else:
                self._manifest = stored_manifest or {}
                if self._manifest.get("state") == "migrating_legacy":
                    # A previous process may have stopped after publishing
                    # either side of the v2 -> v3 contract transition.  The
                    # manifest contains both fingerprints and the complete
                    # target payload, so finishing it is deterministic.
                    self._recover_legacy_contract_migration()
                    stored_manifest = self._manifest

                if self._loaded_contract_version < SWEEP_CONTRACT_VERSION:
                    self._infer_legacy_contract_from_chunks()
                    # A contract-less legacy run first receives a complete
                    # v2 contract.  If the process stops here, the next resume
                    # can still identify the chunks as legacy and repeat the
                    # migration safely.
                    self._ensure_provisional_legacy_contract()
                    self._manifest = (
                        stored_manifest or self._new_run_manifest()
                    )
                    self._reconcile_persistence()
                    self._begin_legacy_contract_migration()
                else:
                    self.contract_sha256 = _sha256_file(self.contract_file)
                    self._manifest = (
                        stored_manifest or self._new_run_manifest()
                    )
                    self._reconcile_persistence()

            self.data_manager.configure_run_identity(
                run_uuid=self.run_uuid,
                contract_sha256=self.contract_sha256,
                block_names=self._expected_result_keys,
            )
        except BaseException:
            self._run_lock.release()
            raise

    @staticmethod
    def _validate_measurement_name(value: Any) -> str:
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError("Measurement name must be a path-like string.")
        name = str(value)
        if not name or name in {".", ".."}:
            raise ValueError(
                "Measurement name must be one non-empty relative component."
            )
        if name.casefold() in {".pynst_locks", ".pynst_merge_locks"}:
            raise ValueError(
                f"Measurement name {name!r} is reserved for PyNST locks."
            )
        if Path(name).is_absolute() or "/" in name or "\\" in name:
            raise ValueError(
                "Measurement name must be one safe relative path component."
            )
        return name

    def _acquire_current_run(self) -> bool:
        """Acquire the run lock and refresh state after an unlocked period.

        Returns ``True`` when this call acquired the lock, and ``False`` when
        the manager already held it from construction.
        """
        acquired_here = not self._run_lock.acquired
        self._run_lock.acquire()
        if not acquired_here:
            return False
        try:
            expected_run_uuid = self.run_uuid
            self._load_and_validate_sweep_contract()
            if self.run_uuid != expected_run_uuid:
                replacement_uuid = self.run_uuid
                self.run_uuid = expected_run_uuid
                raise RuntimeError(
                    "The run directory was replaced while this manager was "
                    "unlocked: expected run UUID "
                    f"{expected_run_uuid!r}, found {replacement_uuid!r}."
                )
            self.contract_sha256 = _sha256_file(self.contract_file)
            manifest = self._read_run_manifest()
            manifest_uuid = str(manifest.get("run_uuid", ""))
            if manifest_uuid and manifest_uuid != self.run_uuid:
                raise RuntimeError(
                    "Run manifest UUID differs from the sweep contract."
                )
            self.run_uuid = manifest_uuid or self.run_uuid
            self._manifest = manifest
            self.metadata = dict(manifest.get("metadata", {}))
            self.metadata.setdefault("measurement_name", self.meas_name)
            self._user_block_metadata = dict(
                manifest.get("user_block_metadata", {})
            )
            self._reconcile_persistence()
            self.data_manager.chunks_written = (
                self.data_manager._get_max_existing_chunk_index() + 1
            )
            self.data_manager.block_mode = (
                self._block_mode or self.data_manager._read_existing_block_mode()
            )
            self.data_manager.configure_run_identity(
                run_uuid=self.run_uuid,
                contract_sha256=self.contract_sha256,
                block_names=self._expected_result_keys,
            )
        except BaseException:
            self._run_lock.release()
            raise
        return True

    def _ensure_storage_usable(self) -> None:
        if self._storage_compromised:
            raise StorageCommitError(
                "This SweepManager encountered an ambiguous storage commit "
                "and is terminal. Close it and create a new manager with "
                "resume=True so persisted chunks can be reconciled safely."
            )

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
            unknown = set(values) - {
                "index_names",
                "columns",
                "index_dtypes",
                "dtypes",
                "index_dtype_specs",
                "dtype_specs",
            }
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
            if "index_dtypes" in values:
                item["index_dtypes"] = [
                    str(value) for value in values["index_dtypes"]
                ]
                if len(item["index_dtypes"]) != len(item["index_names"]):
                    raise ValueError(
                        f"Expected-result index_dtypes for {name!r} do not "
                        "match index_names"
                    )
            for field, names_field in (
                ("index_dtype_specs", "index_names"),
                ("dtype_specs", "columns"),
            ):
                if field not in values:
                    continue
                specs = [dict(spec) for spec in values[field]]
                if len(specs) != len(item[names_field]):
                    raise ValueError(
                        f"Expected-result {field} for {name!r} do not match "
                        f"{names_field}"
                    )
                if any("dtype" not in spec for spec in specs):
                    raise ValueError(
                        f"Every {field} entry for {name!r} requires dtype"
                    )
                item[field] = specs
            result[name] = item
        if not result:
            raise ValueError("expected_result_schema must not be empty")
        return result

    def _sweep_grid_contract(self) -> dict[str, Any]:
        values = [
            [json.loads(json.dumps(value, default=json_default)) for value in row]
            for row in self.multi_index.tolist()
        ]
        dtype_specs: list[dict[str, Any]] = []
        for index in range(self.multi_index.nlevels):
            dtype = self.multi_index.get_level_values(index).dtype
            dtype_specs.append(_dtype_spec(dtype))
        return {
            "parameter_names": list(self.param_col_names),
            "level_dtypes": [
                str(self.multi_index.get_level_values(index).dtype)
                for index in range(self.multi_index.nlevels)
            ],
            "level_dtype_specs": dtype_specs,
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
            "run_uuid": self.run_uuid,
            "measurement_name": self.meas_name,
            "grid": self._sweep_grid_contract(),
            "block_mode": self._block_mode,
            "block_names": block_names,
            "result_schema": None if result_schema is None else dict(result_schema),
        }

    def _replace_contract_payload(
        self,
        payload: Mapping[str, Any],
    ) -> str:
        text = _json_text(payload)
        temporary = self.contract_file.with_name(
            self.contract_file.name + ".tmp"
        )
        # ``Path.write_text`` performs platform newline translation.  Hashing
        # the JSON text before that write therefore disagrees with the bytes
        # on disk on Windows.  Persist exact UTF-8 bytes instead.
        temporary.write_bytes(text.encode("utf-8"))
        _fsync_file(temporary)
        digest = _sha256_file(temporary)
        _replace_file(temporary, self.contract_file)
        self.contract_sha256 = digest
        self._contract_has_run_uuid = payload.get("run_uuid") is not None
        return digest

    def _ensure_provisional_legacy_contract(self) -> None:
        if self.contract_file.is_file():
            self.contract_sha256 = _sha256_file(self.contract_file)
            return
        payload = self._contract_payload(
            result_schema=self._persisted_result_schema
        )
        payload["format_version"] = 2
        self._replace_contract_payload(payload)
        self._loaded_contract_version = 2

    def _begin_legacy_contract_migration(self) -> None:
        target_payload = self._contract_payload(
            result_schema=self._persisted_result_schema
        )
        target_text = _json_text(target_payload)
        target_hash = hashlib.sha256(target_text.encode("utf-8")).hexdigest()
        source_hash = self.contract_sha256

        for record in self._manifest.get("chunks", []):
            record["legacy"] = True
        self._manifest["state"] = "migrating_legacy"
        self._manifest["migration"] = {
            "source_contract_sha256": source_hash,
            "target_contract_sha256": target_hash,
            "target_contract": target_payload,
        }
        self._write_run_manifest()

        written_hash = self._replace_contract_payload(target_payload)
        if written_hash != target_hash:
            raise StorageCommitError(
                "Legacy migration produced an unexpected contract hash."
            )
        self._loaded_contract_version = SWEEP_CONTRACT_VERSION
        self._manifest["contract_sha256"] = target_hash
        self._manifest["state"] = "active"
        self._manifest.pop("migration", None)
        self._write_run_manifest()

    def _recover_legacy_contract_migration(self) -> None:
        migration = self._manifest.get("migration")
        if not isinstance(migration, Mapping):
            raise RuntimeError(
                "Legacy migration manifest is missing its recovery payload."
            )
        target_payload = dict(migration.get("target_contract", {}))
        if not target_payload:
            raise RuntimeError("Legacy migration target contract is missing.")
        target_text = _json_text(target_payload)
        target_hash = hashlib.sha256(target_text.encode("utf-8")).hexdigest()
        if target_hash != migration.get("target_contract_sha256"):
            raise RuntimeError("Legacy migration target fingerprint is corrupt.")

        current_hash = (
            _sha256_file(self.contract_file)
            if self.contract_file.is_file()
            else ""
        )
        source_hash = str(migration.get("source_contract_sha256", ""))
        if current_hash == source_hash:
            self._replace_contract_payload(target_payload)
        elif current_hash != target_hash:
            raise RuntimeError(
                "Legacy migration found neither its source nor target contract."
            )
        else:
            self.contract_sha256 = current_hash

        self._load_and_validate_sweep_contract()
        self.contract_sha256 = target_hash
        self._manifest["contract_sha256"] = target_hash
        self._manifest["state"] = "active"
        self._manifest.pop("migration", None)
        self._write_run_manifest()

    def _write_sweep_contract(
        self,
        *,
        result_schema: Mapping[str, Mapping[str, Any]] | None,
    ) -> None:
        payload = self._contract_payload(result_schema=result_schema)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        text = _json_text(payload)
        new_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if self._manifest.get("chunks"):
            old_hash = self._manifest.get("contract_sha256")
            if old_hash != new_hash:
                raise RuntimeError(
                    "Refusing to change sweep contract after chunks exist."
                )
        written_hash = self._replace_contract_payload(payload)
        if written_hash != new_hash:
            raise StorageCommitError(
                "Sweep contract fingerprint changed while being written."
            )
        if hasattr(self, "data_manager"):
            self.data_manager.configure_run_identity(
                run_uuid=self.run_uuid,
                contract_sha256=self.contract_sha256,
                block_names=self._expected_result_keys,
            )
        if self._manifest:
            self._manifest["contract_sha256"] = self.contract_sha256
            try:
                self._write_run_manifest()
            except Exception as error:
                raise StorageCommitError(
                    "Sweep contract was written, but its manifest fingerprint "
                    f"could not be committed: {error}"
                ) from error

    def _load_and_validate_sweep_contract(self) -> None:
        if not self.contract_file.is_file():
            if self.strict_resume_contract:
                raise RuntimeError(
                    "Cannot safely resume this legacy sweep: sweep_contract.json "
                    "is missing. Start a new run or explicitly set "
                    "strict_resume_contract=False after manual review."
                )
            self._loaded_contract_version = 0
            return
        payload = json.loads(self.contract_file.read_text(encoding="utf-8"))
        if payload.get("format_name") != "pynst.sweep_contract":
            raise ValueError("Resume contract has an unknown format")
        format_version = int(payload.get("format_version", 0))
        if format_version not in SUPPORTED_SWEEP_CONTRACT_VERSIONS:
            raise ValueError("Resume contract has an unsupported version")
        self._loaded_contract_version = format_version
        stored_uuid = payload.get("run_uuid")
        self._contract_has_run_uuid = stored_uuid is not None
        if stored_uuid is not None:
            self.run_uuid = str(stored_uuid)
        if payload.get("measurement_name") != self.meas_name:
            raise ValueError("Resume measurement name differs from its contract")
        expected_grid = payload.get("grid")
        actual_grid = self._sweep_grid_contract()
        if not isinstance(expected_grid, Mapping):
            raise ValueError("Resume contract contains no valid sweep grid")
        expected_grid = dict(expected_grid)
        if "level_dtype_specs" not in expected_grid:
            # Contracts written before full CategoricalDtype support cannot
            # prove category vocabulary/order.  Non-categorical dtypes have
            # a lossless legacy representation and can be normalised safely.
            legacy_dtypes = list(expected_grid.get("level_dtypes", []))
            if "category" in legacy_dtypes:
                raise ValueError(
                    "Legacy sweep contract does not describe categorical "
                    "levels completely; start a new run to bind categories "
                    "and ordering safely."
                )
            expected_grid["level_dtype_specs"] = [
                {"dtype": str(dtype)} for dtype in legacy_dtypes
            ]
        if expected_grid != actual_grid:
            details = []
            for key in (
                "parameter_names",
                "level_dtypes",
                "level_dtype_specs",
                "combination_count",
            ):
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
            for field in (
                "index_dtypes",
                "dtypes",
                "index_dtype_specs",
                "dtype_specs",
            ):
                if (
                    field in declared_block
                    and declared_block[field] != persisted_block.get(field)
                ):
                    return False
        return True

    def _infer_legacy_contract_from_chunks(self) -> None:
        """Complete a pre-v3 contract before adopting any legacy chunks.

        Older runs did not persist enough return-structure information to
        make a rewritten contract immediately self-consistent.  Infer the
        missing, mechanically observable structure once while the run lock is
        held.  The resulting v3 contract then remains immutable.
        """
        chunk_files = self.data_manager._chunk_files()
        if not chunk_files:
            return

        observed_mode: BlockMode | None = self._block_mode
        observed_names: tuple[str, ...] | None = None
        observed_schema: dict[str, dict[str, Any]] | None = None

        for chunk_file in chunk_files:
            with pd.HDFStore(chunk_file, mode="r") as store:
                try:
                    chunk_mode = str(store.root._v_attrs.pynst_block_mode)
                except AttributeError:
                    keys_for_mode = [
                        key.strip("/") for key in _data_block_keys(store)
                    ]
                    chunk_mode = (
                        "list"
                        if keys_for_mode
                        and all(
                            _is_legacy_block_name(name)
                            for name in keys_for_mode
                        )
                        else "mapping"
                    )
                if chunk_mode not in {"list", "mapping"}:
                    raise RuntimeError(
                        f"Legacy chunk {chunk_file.name!r} has invalid block "
                        f"mode {chunk_mode!r}."
                    )
                if observed_mode is None:
                    observed_mode = chunk_mode  # type: ignore[assignment]
                elif observed_mode != chunk_mode:
                    raise RuntimeError(
                        "Legacy chunks disagree about the measurement return "
                        "container type."
                    )

                materialised_names = tuple(
                    sorted(
                        (key.strip("/") for key in _data_block_keys(store)),
                        key=_block_sort_key,
                    )
                )
                if not materialised_names:
                    raise RuntimeError(
                        f"Legacy chunk {chunk_file.name!r} contains no data."
                    )
                if observed_names is None:
                    observed_names = materialised_names
                elif observed_names != materialised_names:
                    raise RuntimeError(
                        "Legacy chunks contain different materialised blocks."
                    )

                chunk_schema: dict[str, dict[str, Any]] = {}
                for block_name in materialised_names:
                    frame = store[block_name]
                    if frame.empty:
                        raise RuntimeError(
                            f"Legacy chunk {chunk_file.name!r} block "
                            f"{block_name!r} is empty."
                        )
                    local_names = [
                        str(name)
                        for name in frame.index.names
                        if name not in self.param_col_names
                    ]
                    local_levels = [
                        frame.index.get_level_values(name)
                        for name in local_names
                    ]
                    chunk_schema[block_name] = {
                        "index_names": local_names,
                        "index_dtypes": [
                            str(level.dtype) for level in local_levels
                        ],
                        "index_dtype_specs": [
                            _dtype_spec(level.dtype) for level in local_levels
                        ],
                        "columns": [str(name) for name in frame.columns],
                        "dtypes": [str(dtype) for dtype in frame.dtypes],
                        "dtype_specs": [
                            _dtype_spec(frame[column].dtype)
                            for column in frame.columns
                        ],
                    }
                if observed_schema is None:
                    observed_schema = chunk_schema
                elif observed_schema != chunk_schema:
                    raise RuntimeError(
                        "Legacy chunks disagree about their DataFrame schema."
                    )

        if observed_mode is None or observed_names is None or observed_schema is None:
            raise RuntimeError("Could not infer the legacy measurement schema.")

        if self._persisted_result_schema is not None:
            if not self._schemas_compatible(
                self._persisted_result_schema,
                observed_schema,
            ):
                raise RuntimeError(
                    "Legacy chunks disagree with the stored result schema."
                )
            # Upgrade a compatible but structurally incomplete v1/v2 schema
            # to the fully observed v3 representation (including local-index
            # and categorical dtype semantics).
            self._persisted_result_schema = observed_schema
        else:
            self._persisted_result_schema = observed_schema

        if self._expected_result_keys is None:
            if observed_mode == "mapping":
                expected_names = tuple(sorted(observed_names))
            else:
                indices = [
                    int(name.removeprefix("block_"))
                    for name in observed_names
                    if _is_legacy_block_name(name)
                ]
                if len(indices) != len(observed_names):
                    raise RuntimeError(
                        "Legacy list-mode chunks contain invalid block names."
                    )
                expected_names = tuple(
                    f"block_{index}" for index in range(max(indices) + 1)
                )
            self._expected_result_keys = expected_names
        self._persisted_block_names = self._expected_result_keys
        self._block_mode = observed_mode

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
                "index_dtypes": [
                    str(value.index.get_level_values(index).dtype)
                    for index in range(value.index.nlevels)
                ],
                "index_dtype_specs": [
                    _dtype_spec(value.index.get_level_values(index).dtype)
                    for index in range(value.index.nlevels)
                ],
                "columns": [str(column) for column in value.columns],
                "dtypes": [str(value[column].dtype) for column in value.columns],
                "dtype_specs": [
                    _dtype_spec(value[column].dtype) for column in value.columns
                ],
            }
        return schema

    @staticmethod
    def _validate_hdf_compatible_blocks(
        blocks: Mapping[str, Any],
    ) -> None:
        """Reject extension dtypes that pandas cannot persist in HDF5.

        CategoricalDtype has a native pandas table representation. Nullable
        integer/boolean/string and other extension arrays currently have no
        reliable pandas HDF representation; accepting them would bind the run
        contract to a schema that can never produce a valid chunk.
        """
        unsupported: list[str] = []
        for block_name, value in blocks.items():
            if not isinstance(value, pd.DataFrame):
                continue
            for level in range(value.index.nlevels):
                level_values = value.index.get_level_values(level)
                dtype = level_values.dtype
                if (
                    isinstance(dtype, pd.api.extensions.ExtensionDtype)
                    and not isinstance(dtype, pd.CategoricalDtype)
                ):
                    unsupported.append(
                        f"{block_name}/index[{level}]={dtype}"
                    )
                elif (
                    isinstance(dtype, np.dtype)
                    and np.issubdtype(dtype, np.complexfloating)
                ):
                    unsupported.append(
                        f"{block_name}/index[{level}]={dtype} "
                        "(split complex index values into real/imag levels)"
                    )
                elif dtype == np.dtype("uint64"):
                    unsupported.append(
                        f"{block_name}/index[{level}]=uint64 "
                        "(use a checked int64 representation)"
                    )
                object_issue = _hdf_object_dtype_issue(
                    level_values,
                    f"{block_name}/index[{level}]",
                )
                if object_issue is not None:
                    unsupported.append(object_issue)
            for column in value.columns:
                dtype = value[column].dtype
                if (
                    isinstance(dtype, pd.api.extensions.ExtensionDtype)
                    and not isinstance(dtype, pd.CategoricalDtype)
                ):
                    unsupported.append(
                        f"{block_name}/{column}={dtype}"
                    )
                object_issue = _hdf_object_dtype_issue(
                    value[column],
                    f"{block_name}/{column}",
                    allow_empty=True,
                )
                if object_issue is not None:
                    unsupported.append(object_issue)
        if unsupported:
            raise TypeError(
                "pandas HDF storage does not support these extension dtypes; "
                "convert them to NumPy dtypes or homogeneous string values "
                "before returning the measurement result: "
                f"{unsupported!r}."
            )

    def _validate_and_persist_result_schema(
        self,
        blocks: Mapping[str, Any],
    ) -> None:
        actual = self._result_schema_from_blocks(blocks)
        if not actual:
            raise ValueError(
                "Measurement result must contain at least one DataFrame block."
            )
        empty_blocks = [
            name
            for name, value in blocks.items()
            if isinstance(value, pd.DataFrame) and value.empty
        ]
        if empty_blocks:
            raise ValueError(
                "Measurement DataFrame blocks must not be empty: "
                f"{empty_blocks!r}."
            )
        index_only_blocks = [
            name
            for name, value in blocks.items()
            if isinstance(value, pd.DataFrame) and len(value.columns) == 0
        ]
        if index_only_blocks:
            raise ValueError(
                "Measurement DataFrame blocks require at least one dependent "
                f"variable column: {index_only_blocks!r}."
            )
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
                for field in (
                    "index_dtypes",
                    "dtypes",
                    "index_dtype_specs",
                    "dtype_specs",
                ):
                    if (
                        field in expected_block
                        and expected_block[field] != actual_block[field]
                    ):
                        raise ValueError(
                            f"Measurement block {name!r} changed {field}: "
                            f"expected {expected_block[field]!r}, got "
                            f"{actual_block[field]!r}"
                        )
            if set(actual) != set(expected):
                raise ValueError(
                    "Measurement result block names changed: "
                    f"expected {tuple(expected)!r}, got {tuple(actual)!r}"
                )
        if self._persisted_result_schema is None:
            self._persisted_result_schema = actual
            self._write_sweep_contract(result_schema=actual)

    def _new_run_manifest(self) -> dict[str, Any]:
        return {
            "format_name": "pynst.run_manifest",
            "format_version": RUN_MANIFEST_VERSION,
            "run_uuid": self.run_uuid,
            "contract_sha256": self.contract_sha256,
            "state": "active",
            "metadata": dict(self.metadata),
            "user_block_metadata": dict(self._user_block_metadata),
            "chunks": [],
            "failed_sequences": [],
            "archived_artifact": None,
        }

    def _read_run_manifest(self) -> dict[str, Any]:
        payload = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        if payload.get("format_name") != "pynst.run_manifest":
            raise RuntimeError("Unknown PyNST run-manifest format.")
        if int(payload.get("format_version", 0)) != RUN_MANIFEST_VERSION:
            raise RuntimeError("Unsupported PyNST run-manifest version.")
        return dict(payload)

    def _write_run_manifest(self) -> None:
        self._manifest["metadata"] = dict(self.metadata)
        self._manifest["user_block_metadata"] = dict(
            self._user_block_metadata
        )
        _atomic_write_json(self.manifest_file, self._manifest)

    @staticmethod
    def _log_value(value: Any) -> str:
        return json.dumps(
            value,
            default=json_default,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

    def _write_log_from_state(self) -> None:
        complete = {
            int(sequence): str(record["filename"])
            for record in self._manifest.get("chunks", [])
            for sequence in record.get("sequence_indices", [])
        }
        failed = {
            int(sequence)
            for sequence in self._manifest.get("failed_sequences", [])
        } - set(complete)
        lines = [
            "# " + json.dumps(self.param_col_names, ensure_ascii=False),
            "\t".join(self.param_col_names)
            + "\tchunk_file\tstatus\tsequence_index",
        ]
        self.log_dict = {}
        for sequence_index, params in enumerate(self.param_list):
            if sequence_index in complete:
                chunk, status = complete[sequence_index], "Complete"
            elif sequence_index in failed:
                chunk, status = "NA", "Failed"
            else:
                continue
            display = "\t".join(
                self._log_value(params[name])
                for name in self.param_col_names
            )
            lines.append(
                f"{display}\t{chunk}\t{status}\t{sequence_index}"
            )
            self.log_dict[sequence_index] = (chunk, status)
        _atomic_write_text(self.log_file, "\n".join(lines) + "\n")

    def _load_diagnostic_log(self) -> dict[int, tuple[str, str]]:
        if not self.log_file.is_file():
            return {}
        results: dict[int, tuple[str, str]] = {}
        parameter_count = len(self.param_col_names)
        with self.log_file.open("r", encoding="utf-8") as log:
            header_seen = False
            has_sequence_column = False
            for line_number, line in enumerate(log, start=1):
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if not header_seen:
                    legacy_header = self.param_col_names + [
                        "chunk_file",
                        "status",
                    ]
                    current_header = legacy_header + ["sequence_index"]
                    if parts not in (legacy_header, current_header):
                        raise RuntimeError(
                            "Diagnostic log has an invalid header at line "
                            f"{line_number}."
                        )
                    header_seen = True
                    has_sequence_column = parts == current_header
                    continue

                expected_length = parameter_count + (
                    3 if has_sequence_column else 2
                )
                if len(parts) != expected_length:
                    raise RuntimeError(
                        "Diagnostic log has a malformed row at line "
                        f"{line_number}."
                    )
                if has_sequence_column:
                    try:
                        sequence_index = int(parts[parameter_count + 2])
                    except ValueError as error:
                        raise RuntimeError(
                            "Diagnostic log has an invalid sequence_index at "
                            f"line {line_number}."
                        ) from error
                else:
                    legacy_key = tuple(parts[:parameter_count])
                    candidates = [
                        index
                        for index, params in enumerate(self.param_list)
                        if tuple(
                            str(params[name]) for name in self.param_col_names
                        )
                        == legacy_key
                    ]
                    if len(candidates) != 1:
                        raise RuntimeError(
                            "Legacy log parameter identity is ambiguous; "
                            "manual migration is required."
                        )
                    sequence_index = candidates[0]

                if not 0 <= sequence_index < len(self.param_list):
                    raise RuntimeError(
                        f"Log contains invalid sequence_index {sequence_index}."
                    )
                chunk = parts[parameter_count]
                status = parts[parameter_count + 1]
                if status not in {"Complete", "Failed"}:
                    raise RuntimeError(
                        f"Diagnostic log has invalid status {status!r} at "
                        f"line {line_number}."
                    )
                if Path(chunk).name != chunk or chunk in {"", ".", ".."}:
                    raise RuntimeError(
                        f"Diagnostic log has unsafe chunk name {chunk!r} at "
                        f"line {line_number}."
                    )
                if sequence_index in results:
                    raise RuntimeError(
                        "Diagnostic log repeats sequence_index "
                        f"{sequence_index}."
                    )
                results[sequence_index] = (chunk, status)
        return results

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

    @staticmethod
    def _scalar_equal(left: Any, right: Any) -> bool:
        try:
            if bool(pd.isna(left)) and bool(pd.isna(right)):
                return True
        except (TypeError, ValueError):
            pass
        try:
            return type(left) is type(right) and bool(left == right)
        except (TypeError, ValueError):
            return False

    def _sequence_for_values(self, values: Sequence[Any]) -> int:
        candidates = []
        for sequence_index, params in enumerate(self.param_list):
            if all(
                self._scalar_equal(params[name], value)
                for name, value in zip(self.param_col_names, values)
            ):
                candidates.append(sequence_index)
        if len(candidates) != 1:
            raise RuntimeError(
                "Chunk parameter values do not map uniquely to the sweep "
                f"contract: {tuple(values)!r}"
            )
        return candidates[0]

    def _chunk_record(
        self,
        path: Path,
        *,
        allow_legacy_metadata: bool = False,
    ) -> dict[str, Any]:
        try:
            chunk_index = int(path.stem.removeprefix("chunk_"))
        except ValueError as error:
            raise RuntimeError(f"Invalid chunk filename {path.name!r}") from error

        require_commit_metadata = (
            self._loaded_contract_version >= SWEEP_CONTRACT_VERSION
            and not allow_legacy_metadata
        )
        legacy_metadata = False
        with pd.HDFStore(path, mode="r") as store:
            try:
                internal_index = int(
                    store.root._v_attrs.pynst_chunk_index
                )
            except AttributeError:
                legacy_metadata = True
                if require_commit_metadata:
                    raise RuntimeError(
                        f"Chunk {path.name!r} has no internal chunk index."
                    )
                internal_index = chunk_index
            if internal_index != chunk_index:
                raise RuntimeError(
                    f"Chunk {path.name!r} has internal index {internal_index}."
                )

            try:
                chunk_uuid = str(store.root._v_attrs.pynst_run_uuid)
            except AttributeError:
                chunk_uuid = ""
                legacy_metadata = True
            if require_commit_metadata and not chunk_uuid:
                raise RuntimeError(
                    f"Chunk {path.name!r} has no run UUID."
                )
            if chunk_uuid and chunk_uuid != self.run_uuid:
                raise RuntimeError(
                    f"Chunk {path.name!r} belongs to a different run UUID."
                )
            try:
                chunk_contract = str(
                    store.root._v_attrs.pynst_contract_sha256
                )
            except AttributeError:
                chunk_contract = ""
                legacy_metadata = True
            if require_commit_metadata and not chunk_contract:
                raise RuntimeError(
                    f"Chunk {path.name!r} has no contract fingerprint."
                )
            if chunk_contract and chunk_contract != self.contract_sha256:
                if allow_legacy_metadata:
                    legacy_metadata = True
                else:
                    raise RuntimeError(
                        f"Chunk {path.name!r} references a different contract."
                    )

            try:
                mode = str(store.root._v_attrs.pynst_block_mode)
            except AttributeError:
                legacy_metadata = True
                if require_commit_metadata:
                    raise RuntimeError(
                        f"Chunk {path.name!r} has no block mode."
                    )
                mode = self.data_manager.block_mode or "mapping"
            if self._block_mode is not None and mode != self._block_mode:
                raise RuntimeError(
                    f"Chunk {path.name!r} has block mode {mode!r}, expected "
                    f"{self._block_mode!r}."
                )

            block_names = {
                key.strip("/") for key in _data_block_keys(store)
            }
            try:
                declared_block_names = tuple(
                    _normalise_block_name(str(name))
                    for name in json.loads(
                        str(
                            store.root._v_attrs
                            .pynst_block_names_json
                        )
                    )
                )
            except AttributeError:
                declared_block_names = ()
                legacy_metadata = True
            if require_commit_metadata and not declared_block_names:
                raise RuntimeError(
                    f"Chunk {path.name!r} has no block-name manifest."
                )
            if self._expected_result_keys is not None and declared_block_names:
                expected_names = self._expected_result_keys
                if self._block_mode == "mapping":
                    declared_block_names = tuple(sorted(declared_block_names))
                if declared_block_names != expected_names:
                    raise RuntimeError(
                        f"Chunk {path.name!r} declares block names "
                        f"{declared_block_names!r}, expected {expected_names!r}."
                    )
            expected_materialised = (
                set(self._persisted_result_schema)
                if self._persisted_result_schema is not None
                else block_names
            )
            if block_names != expected_materialised:
                raise RuntimeError(
                    f"Chunk {path.name!r} contains blocks "
                    f"{sorted(block_names)!r}, expected "
                    f"{sorted(expected_materialised)!r}."
                )

            try:
                sequence_indices = [
                    int(value)
                    for value in json.loads(
                        str(
                            store.root._v_attrs
                            .pynst_sequence_indices_json
                        )
                    )
                ]
            except AttributeError:
                sequence_indices = []
                legacy_metadata = True
            if require_commit_metadata and not sequence_indices:
                raise RuntimeError(
                    f"Chunk {path.name!r} has no sequence-index manifest."
                )

            inferred_by_block: list[set[int]] = []
            for block_name in sorted(block_names, key=_block_sort_key):
                storer = store.get_storer(block_name)
                is_table = bool(storer.is_table)
                if is_table:
                    frames: Iterable[pd.DataFrame] = store.select(
                        block_name,
                        chunksize=_CHUNK_VALIDATION_BATCH_ROWS,
                    )
                else:
                    frames = (store[block_name],)

                observed_rows = 0
                block_sequences: set[int] = set()
                seen_index_values: set[Any] = set()
                for frame in frames:
                    observed_rows += len(frame)
                    if not frame.index.is_unique:
                        raise RuntimeError(
                            f"Chunk {path.name!r} block {block_name!r} has "
                            "duplicate index rows."
                        )
                    if is_table:
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
                            raise RuntimeError(
                                f"Chunk {path.name!r} block "
                                f"{block_name!r} has unhashable index "
                                "values."
                            ) from error
                        if seen_index_values & index_values:
                            raise RuntimeError(
                                f"Chunk {path.name!r} block "
                                f"{block_name!r} has duplicate index rows "
                                "across validation batches."
                            )
                        seen_index_values.update(index_values)

                    missing = [
                        name
                        for name in self.param_col_names
                        if name not in frame.index.names
                    ]
                    if missing:
                        raise RuntimeError(
                            f"Chunk {path.name!r} block {block_name!r} is "
                            f"missing sweep levels {missing!r}."
                        )
                    if self._persisted_result_schema is not None:
                        expected = self._persisted_result_schema[block_name]
                        local_names = [
                            name
                            for name in frame.index.names
                            if name not in self.param_col_names
                        ]
                        if local_names != expected["index_names"]:
                            raise RuntimeError(
                                f"Chunk {path.name!r} block {block_name!r} "
                                "has incompatible local index names."
                            )
                        actual_index_dtypes = [
                            str(frame.index.get_level_values(name).dtype)
                            for name in local_names
                        ]
                        expected_index_dtypes = expected.get("index_dtypes")
                        if (
                            expected_index_dtypes is not None
                            and actual_index_dtypes != expected_index_dtypes
                        ):
                            raise RuntimeError(
                                f"Chunk {path.name!r} block {block_name!r} "
                                "has incompatible local index dtypes."
                            )
                        actual_index_specs = [
                            _dtype_spec(
                                frame.index.get_level_values(name).dtype
                            )
                            for name in local_names
                        ]
                        expected_index_specs = expected.get(
                            "index_dtype_specs"
                        )
                        if (
                            expected_index_specs is not None
                            and actual_index_specs != expected_index_specs
                        ):
                            raise RuntimeError(
                                f"Chunk {path.name!r} block {block_name!r} "
                                "has incompatible local index dtype "
                                "semantics."
                            )
                        if (
                            [str(name) for name in frame.columns]
                            != expected["columns"]
                        ):
                            raise RuntimeError(
                                f"Chunk {path.name!r} block {block_name!r} "
                                "has incompatible columns."
                            )
                        expected_dtypes = expected.get("dtypes")
                        if (
                            expected_dtypes is not None
                            and [str(dtype) for dtype in frame.dtypes]
                            != expected_dtypes
                        ):
                            raise RuntimeError(
                                f"Chunk {path.name!r} block {block_name!r} "
                                "has incompatible dtypes."
                            )
                        expected_dtype_specs = expected.get("dtype_specs")
                        actual_dtype_specs = [
                            _dtype_spec(frame[column].dtype)
                            for column in frame.columns
                        ]
                        if (
                            expected_dtype_specs is not None
                            and actual_dtype_specs != expected_dtype_specs
                        ):
                            raise RuntimeError(
                                f"Chunk {path.name!r} block {block_name!r} "
                                "has incompatible dtype semantics."
                            )
                    parameter_rows = frame.index.to_frame(index=False)[
                        self.param_col_names
                    ].drop_duplicates()
                    block_sequences.update(
                        self._sequence_for_values(row)
                        for row in parameter_rows.itertuples(
                            index=False,
                            name=None,
                        )
                    )
                if observed_rows == 0:
                    raise RuntimeError(
                        f"Chunk {path.name!r} block {block_name!r} is empty."
                    )
                inferred_by_block.append(block_sequences)

        if not sequence_indices:
            if not inferred_by_block:
                raise RuntimeError(f"Chunk {path.name!r} contains no data.")
            first = inferred_by_block[0]
            if any(values != first for values in inferred_by_block[1:]):
                raise RuntimeError(
                    f"Chunk {path.name!r} blocks contain different sequences."
                )
            sequence_indices = sorted(first)
        expected_set = set(sequence_indices)
        if len(expected_set) != len(sequence_indices):
            raise RuntimeError(
                f"Chunk {path.name!r} manifest repeats a sequence index."
            )
        if any(values != expected_set for values in inferred_by_block):
            raise RuntimeError(
                f"Chunk {path.name!r} data disagree with its sequence manifest."
            )
        if not expected_set or min(expected_set) < 0 or max(expected_set) >= len(
            self.param_list
        ):
            raise RuntimeError(
                f"Chunk {path.name!r} contains invalid sequence indices."
            )

        return {
            "index": chunk_index,
            "filename": path.name,
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
            "sequence_indices": sequence_indices,
            "block_mode": mode,
            "legacy": legacy_metadata,
        }

    def _reconcile_persistence(self) -> None:
        if self._manifest.get("state", "active") != "active":
            raise RuntimeError("Only active PyNST runs can be resumed.")
        manifest_hash = self._manifest.get("contract_sha256")
        if manifest_hash and manifest_hash != self.contract_sha256:
            if self._manifest.get("chunks"):
                raise RuntimeError(
                    "Run manifest references a different sweep contract."
                )
            # Before the first chunk, observing the result schema legitimately
            # evolves the contract. A crash between the contract replace and
            # manifest replace is recoverable because no data is bound to the
            # old fingerprint yet.
        self._manifest["contract_sha256"] = self.contract_sha256
        self._manifest["run_uuid"] = self.run_uuid

        listed_records = [
            dict(record) for record in self._manifest.get("chunks", [])
        ]
        listed_filenames = [
            str(record.get("filename", "")) for record in listed_records
        ]
        if (
            any(not filename for filename in listed_filenames)
            or len(listed_filenames) != len(set(listed_filenames))
        ):
            raise RuntimeError(
                "Run manifest contains missing or duplicate chunk filenames."
            )
        listed = {
            filename: record
            for filename, record in zip(listed_filenames, listed_records)
        }
        scanned: dict[str, dict[str, Any]] = {}
        scanned_indices: set[int] = set()
        for path in self.data_manager._chunk_files():
            expected = listed.get(path.name, {})
            allow_legacy = (
                self._loaded_contract_version < SWEEP_CONTRACT_VERSION
                or bool(expected.get("legacy", False))
            )
            record = self._chunk_record(
                path,
                allow_legacy_metadata=allow_legacy,
            )
            if int(record["index"]) in scanned_indices:
                raise RuntimeError(
                    f"Multiple final chunks use index {record['index']}."
                )
            scanned_indices.add(int(record["index"]))
            scanned[record["filename"]] = record
        for filename, expected in listed.items():
            actual = scanned.get(filename)
            if actual is None:
                raise RuntimeError(
                    f"Manifest references missing chunk {filename!r}."
                )
            for field in ("sha256", "size", "sequence_indices"):
                if expected.get(field) != actual.get(field):
                    raise RuntimeError(
                        f"Chunk {filename!r} failed manifest field {field!r}."
                    )

        owners: dict[int, str] = {}
        for filename, record in scanned.items():
            for sequence_index in record["sequence_indices"]:
                previous = owners.get(sequence_index)
                if previous is not None:
                    raise RuntimeError(
                        "Multiple chunks claim sequence_index "
                        f"{sequence_index}: {previous!r}, {filename!r}."
                    )
                owners[sequence_index] = filename

        if self._loaded_contract_version >= SWEEP_CONTRACT_VERSION:
            # Version-3 logs are diagnostic only. Contract, manifest and
            # validated chunks remain sufficient to rebuild a missing, torn
            # or non-decodable log, so it is deliberately not parsed here.
            diagnostic: dict[int, tuple[str, str]] = {}
        else:
            diagnostic = self._load_diagnostic_log()
        if not listed:
            for sequence_index, (filename, status) in diagnostic.items():
                if status == "Complete":
                    owner = owners.get(sequence_index)
                    if owner is None:
                        raise RuntimeError(
                            "Legacy log marks sequence_index "
                            f"{sequence_index} Complete, but its chunk is missing."
                        )
                    if filename not in {"", "NA", owner}:
                        raise RuntimeError(
                            "Legacy log points sequence_index "
                            f"{sequence_index} to {filename!r}, found {owner!r}."
                        )

        recovered = [
            record for filename, record in scanned.items() if filename not in listed
        ]
        if recovered:
            print(
                "Resume: adopting "
                f"{len(recovered)} validated orphan chunk(s)."
            )
        all_records = list(listed.values()) + recovered
        all_records.sort(key=lambda record: int(record["index"]))
        self._manifest["chunks"] = all_records

        complete = set(owners)
        failed = {
            int(value)
            for value in self._manifest.get("failed_sequences", [])
        }
        failed.update(
            sequence_index
            for sequence_index, (_, status) in diagnostic.items()
            if status == "Failed"
        )
        self._manifest["failed_sequences"] = sorted(failed - complete)
        self._write_run_manifest()
        self._write_log_from_state()

    def _on_chunk_written(
        self,
        chunk_idx: int,
        chunk_filename: str,
        sequence_indices: list[int],
    ) -> None:
        path = self.output_dir / chunk_filename
        record = self._chunk_record(path)
        if record["index"] != chunk_idx:
            raise RuntimeError("Chunk callback index does not match chunk file.")
        if record["sequence_indices"] != sequence_indices:
            raise RuntimeError(
                "Chunk callback sequences do not match its internal manifest."
            )
        existing = {
            int(sequence)
            for item in self._manifest.get("chunks", [])
            for sequence in item.get("sequence_indices", [])
        }
        overlap = existing & set(sequence_indices)
        if overlap:
            raise RuntimeError(
                f"Chunk commit would duplicate sequences {sorted(overlap)!r}."
            )
        self._manifest.setdefault("chunks", []).append(record)
        failed = {
            int(value)
            for value in self._manifest.get("failed_sequences", [])
        } - set(sequence_indices)
        self._manifest["failed_sequences"] = sorted(failed)
        self._write_run_manifest()
        self._write_log_from_state()

    def _record_failed_sequence(self, sequence_index: int) -> None:
        failed = {
            int(value)
            for value in self._manifest.get("failed_sequences", [])
        }
        failed.add(int(sequence_index))
        self._manifest["failed_sequences"] = sorted(failed)
        self._write_run_manifest()
        self._write_log_from_state()

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
                name: (
                    blocks[name].copy(deep=True)
                    if name in blocks
                    else None
                )
                for name in names
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
        sequence_index: int,
    ) -> MeasurementResult:
        """Load one exact, complete predecessor from its referenced chunk."""
        log_entry = self.log_dict.get(sequence_index)
        if log_entry is None or log_entry[1] != "Complete":
            raise RuntimeError(
                "Previous sequence_index "
                f"{sequence_index} is not marked Complete in the manifest"
            )

        chunk_filename = log_entry[0]
        relative_chunk = Path(chunk_filename)
        if relative_chunk.is_absolute() or relative_chunk.name != chunk_filename:
            raise RuntimeError(
                f"Invalid chunk reference {chunk_filename!r} for "
                f"sequence_index {sequence_index}"
            )
        chunk_path = (self.output_dir / relative_chunk).resolve()
        if chunk_path.parent != self.output_dir.resolve():
            raise RuntimeError(
                f"Chunk reference {chunk_filename!r} escapes the sweep directory"
            )
        if not chunk_path.is_file():
            raise FileNotFoundError(
                f"Chunk {chunk_filename!r} referenced by sequence_index "
                f"{sequence_index} is missing"
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
                hdf_key.strip("/") for hdf_key in _data_block_keys(store)
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
                        f"Complete sequence_index {sequence_index} has no rows "
                        "in block "
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
            for level, name in enumerate(self.param_col_names):
                # A value obtained by iterating a MultiIndex is normally a
                # Python scalar.  Assigning that scalar directly makes pandas
                # widen e.g. int32/float32 to int64/float64 and turns a
                # categorical level into object.  Construct a typed Series so
                # chunk storage and the declared sweep metadata stay equal.
                level_dtype = self.multi_index.get_level_values(level).dtype
                repeated = [params[name]] * len(flat)
                if isinstance(level_dtype, pd.CategoricalDtype):
                    values = pd.Categorical(repeated, dtype=level_dtype)
                    flat[name] = pd.Series(values, index=flat.index)
                else:
                    flat[name] = pd.Series(
                        repeated,
                        index=flat.index,
                        dtype=level_dtype,
                    )

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
            if name in self.param_col_names:
                level = self.param_col_names.index(name)
                dtype = self.multi_index.get_level_values(level).dtype
            else:
                dtype = frame.index.get_level_values(name).dtype
            actual_spec = _dtype_spec(dtype)
            actual_dtype = actual_spec["dtype"]
            declared_dtype = values.get("dtype")
            if declared_dtype is not None and declared_dtype != actual_dtype:
                raise ValueError(
                    f"User metadata for {block_name!r}/{name!r} declares "
                    f"dtype {declared_dtype!r}, observed {actual_dtype!r}."
                )
            for field in ("categories", "ordered"):
                if field in values and values[field] != actual_spec.get(field):
                    raise ValueError(
                        f"User metadata for {block_name!r}/{name!r} declares "
                        f"{field}={values[field]!r}, observed "
                        f"{actual_spec.get(field)!r}."
                    )
                if field not in actual_spec:
                    values.pop(field, None)
            values.update(actual_spec)

        for name in dvars:
            values = variable_values.setdefault(name, {})
            actual_spec = _dtype_spec(frame[name].dtype)
            actual_dtype = actual_spec["dtype"]
            declared_dtype = values.get("dtype")
            if declared_dtype is not None and declared_dtype != actual_dtype:
                raise ValueError(
                    f"User metadata for {block_name!r}/{name!r} declares "
                    f"dtype {declared_dtype!r}, observed {actual_dtype!r}."
                )
            for field in ("categories", "ordered"):
                if field in values and values[field] != actual_spec.get(field):
                    raise ValueError(
                        f"User metadata for {block_name!r}/{name!r} declares "
                        f"{field}={values[field]!r}, observed "
                        f"{actual_spec.get(field)!r}."
                    )
                if field not in actual_spec:
                    values.pop(field, None)
            values.update(actual_spec)

        user_values["variables"] = variable_values
        candidate = BlockMetadata.from_dict(
            user_values,
            default_key=block_name,
        )
        self._apply_user_block_metadata(block_name, candidate)

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

            for name in existing.ivars + existing.dvars:
                old_variable = existing.variables.get(
                    name, VariableMetadata()
                )
                new_variable = candidate.variables.get(
                    name, VariableMetadata()
                )
                old_spec = {
                    "dtype": old_variable.dtype,
                    **{
                        field: old_variable.extra[field]
                        for field in ("categories", "ordered")
                        if field in old_variable.extra
                    },
                }
                new_spec = {
                    "dtype": new_variable.dtype,
                    **{
                        field: new_variable.extra[field]
                        for field in ("categories", "ordered")
                        if field in new_variable.extra
                    },
                }
                if old_spec != new_spec:
                    raise ValueError(
                        f"Structure of {block_name!r} changed during the "
                        f"sweep: dtype semantics for {name!r} differ."
                    )

            # Merge newly supplied optional semantic metadata.
            for name, variable in candidate.variables.items():
                existing.variables.setdefault(name, variable)
            return

        self.block_metadata[block_name] = candidate

    def _block_metadata_dict(self) -> dict[str, dict[str, Any]]:
        return {
            name: metadata.to_dict()
            for name, metadata in self.block_metadata.items()
        }

    def _apply_user_block_metadata(
        self,
        block_name: str,
        block: BlockMetadata,
    ) -> None:
        """Overlay persisted semantic metadata on observed block structure."""
        user = dict(self._user_block_metadata.get(block_name, {}))
        variable_updates = dict(user.pop("variables", {}) or {})
        structural_fields = (
            "schema_version",
            "hdf_key",
            "ivars",
            "sweep_ivars",
            "local_ivars",
            "dvars",
        )
        for field in structural_fields:
            if field in user and user[field] != getattr(block, field):
                raise ValueError(
                    f"User metadata for {block_name!r} changes structural "
                    f"field {field!r}."
                )

        for field in ("required", "default_load", "description"):
            if field in user:
                setattr(block, field, user[field])

        non_extra = set(structural_fields) | {
            "required",
            "default_load",
            "description",
        }
        block.extra.update(
            {key: value for key, value in user.items() if key not in non_extra}
        )

        known_variables = set(block.ivars) | set(block.dvars)
        unknown_variables = set(variable_updates) - known_variables
        if unknown_variables:
            raise ValueError(
                f"User metadata for {block_name!r} references unknown "
                f"variables {sorted(unknown_variables)!r}."
            )
        for variable_name, raw_values in variable_updates.items():
            values = dict(raw_values)
            current = block.variables.get(variable_name, VariableMetadata())
            declared_dtype = values.get("dtype")
            if (
                declared_dtype is not None
                and current.dtype is not None
                and declared_dtype != current.dtype
            ):
                raise ValueError(
                    f"User metadata for {block_name!r}/{variable_name!r} "
                    "changes the observed dtype."
                )
            for field in ("categories", "ordered"):
                if (
                    field in values
                    and values[field] != current.extra.get(field)
                ):
                    raise ValueError(
                        f"User metadata for {block_name!r}/"
                        f"{variable_name!r} changes observed categorical "
                        f"{field}."
                    )
            merged = current.to_dict()
            merged.update(values)
            block.variables[variable_name] = VariableMetadata.from_dict(merged)
        block.validate()

    def run(self) -> bool:
        """Execute pending sequences and return whether the run is complete."""
        self._ensure_storage_usable()
        if self._already_run:
            raise RuntimeError(
                "SweepManager.run() was already called. Create a new "
                "manager to execute the sweep again."
            )
        self._acquire_current_run()
        self._already_run = True

        completed_count = sum(
            1
            for sequence_index in range(len(self.param_list))
            if self.log_dict.get(sequence_index, (None, None))[1] == "Complete"
        )
        progress = tqdm(
            total=len(self.param_list),
            initial=completed_count,
            desc="Sweep",
            leave=True,
        )

        previous_result: MeasurementResult | None = None
        previous_reference: tuple[
            dict[str, Any], int
        ] | None = None
        try:
            for sequence_index, params in enumerate(self.param_list):
                if self.log_dict.get(sequence_index, (None, None))[1] == "Complete":
                    if self.provide_previous_result:
                        # Defer disk I/O until a pending point needs it.
                        previous_result = None
                        previous_reference = (dict(params), sequence_index)
                    continue

                description = " | ".join(
                    f"{name}={value}" for name, value in params.items()
                )
                progress.set_description(description)

                try:
                    if (
                        self.provide_previous_result
                        and previous_reference is not None
                    ):
                        predecessor_params, predecessor_index = (
                            previous_reference
                        )
                        try:
                            previous_result = self._load_persisted_result(
                                predecessor_params,
                                predecessor_index,
                            )
                        except Exception as error:
                            raise CriticalMeasurementError(
                                "Cannot safely load the previous complete result "
                                "before the next measurement; aborting before "
                                f"the hardware callback: {error}"
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
                        self._validate_hdf_compatible_blocks(blocks)
                        tagged = self._tag_with_params(blocks, params)
                        # Persist the first observed schema only after all
                        # name-collision, index and metadata checks passed.
                        # Otherwise a structurally invalid callback result
                        # could permanently bind an empty run to an unusable
                        # contract.
                        self._validate_and_persist_result_schema(blocks)
                    except StorageCommitError:
                        raise
                    except Exception as error:
                        raise CriticalMeasurementError(
                            "Measurement result normalization/tagging failed "
                            "after the measurement function returned; aborting "
                            f"the sweep: {error}"
                        ) from error

                    try:
                        self.data_manager.add_worker_data(
                            tagged,
                            metadata={
                                "block_mode": mode,
                                "block_names": self._expected_result_keys,
                                "blocks": self._block_metadata_dict(),
                            },
                            sequence_indices=[sequence_index],
                        )
                    except Exception as error:
                        raise StorageCommitError(
                            "Measurement storage commit failed; aborting the "
                            f"sweep: {error}"
                        ) from error

                    if self.provide_previous_result:
                        previous_result = self._result_from_tagged_blocks(
                            tagged,
                            mode,
                        )
                        previous_reference = None

                except CriticalMeasurementError as error:
                    progress.write(f"Critical error: {error}")
                    if not isinstance(error, StorageCommitError):
                        try:
                            self.data_manager.finalize()
                        except Exception as finalize_error:
                            error = StorageCommitError(
                                "Finalizing prior accepted measurements failed: "
                                f"{finalize_error}"
                            )
                    log_error = self._best_effort_write_exception(
                        "Critical Exception",
                        description,
                    )
                    if log_error is not None:
                        error = StorageCommitError(
                            f"{error}; additionally could not persist the "
                            f"traceback: {log_error}"
                        )
                    if isinstance(error, StorageCommitError):
                        self._storage_compromised = True
                    if self.critical_callback:
                        self.critical_callback(error)
                    print("Measurement run aborted after a critical error.")
                    return False

                except Exception as error:
                    progress.write(
                        "Exception occurred, measurement skipped: "
                        f"{error}"
                    )
                    try:
                        self._record_failed_sequence(sequence_index)
                        self._write_exception("Exception", description)
                    except Exception as storage_error:
                        self._storage_compromised = True
                        critical = StorageCommitError(
                            "Could not persist a failed measurement state; "
                            f"aborting the sweep: {storage_error}"
                        )
                        try:
                            self.data_manager.finalize()
                        except Exception as finalize_error:
                            critical = StorageCommitError(
                                f"{critical}; finalizing prior accepted data "
                                f"also failed: {finalize_error}"
                            )
                        self._best_effort_write_exception(
                            "Critical Storage Exception",
                            description,
                        )
                        if self.critical_callback:
                            self.critical_callback(critical)
                        print(
                            "Measurement run aborted while persisting a "
                            "failed point."
                        )
                        return False

                progress.update(1)

            try:
                self.data_manager.finalize()
            except Exception as error:
                self._storage_compromised = True
                critical = StorageCommitError(
                    f"Final measurement storage commit failed: {error}"
                )
                self._best_effort_write_exception(
                    "Critical Storage Exception",
                    "finalize",
                )
                if self.critical_callback:
                    self.critical_callback(critical)
                print("Measurement run aborted during final storage commit.")
                return False

            failed_count = sum(
                1
                for sequence_index in range(len(self.param_list))
                if self.log_dict.get(sequence_index, (None, None))[1]
                != "Complete"
            )
            if failed_count:
                print(
                    "Measurement run incomplete: "
                    f"{failed_count}/{len(self.param_list)} combinations "
                    "are not complete."
                )
                return False
            print("Measurement run completed.")
            return True

        except (KeyboardInterrupt, SystemExit):
            try:
                self.data_manager.finalize()
            except Exception:
                self._storage_compromised = True
                self._best_effort_write_exception(
                    "Storage Exception while handling interruption",
                    "interrupt",
                )
            raise
        finally:
            progress.close()
            self._run_lock.release()

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

    def _best_effort_write_exception(
        self,
        heading: str,
        description: str,
    ) -> Exception | None:
        try:
            self._write_exception(heading, description)
        except Exception as error:
            return error
        return None

    def get_results(
        self,
        include_nan: bool = True,
    ) -> list[Any] | dict[str, pd.DataFrame | None]:
        self._ensure_storage_usable()
        acquired_here = self._acquire_current_run()
        try:
            if self._manifest.get("state") == "archived":
                raise RuntimeError(
                    "Run chunks were archived; read the merged artifact "
                    "instead."
                )
            return self.data_manager.get_results(
                include_nan=include_nan
            )
        finally:
            if acquired_here:
                self._run_lock.release()

    def update_metadata(
        self,
        values: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Atomically update global or per-block semantic metadata."""
        self._ensure_storage_usable()
        acquired_here = self._acquire_current_run()
        snapshots = (
            copy.deepcopy(self.metadata),
            copy.deepcopy(self._user_block_metadata),
            copy.deepcopy(self.block_metadata),
            copy.deepcopy(self._manifest),
        )
        try:
            self._update_metadata_locked(values, **kwargs)
        except BaseException:
            (
                self.metadata,
                self._user_block_metadata,
                self.block_metadata,
                self._manifest,
            ) = snapshots
            raise
        finally:
            if acquired_here:
                self._run_lock.release()

    def _update_metadata_locked(
        self,
        values: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        incoming = dict(values or {})
        incoming.update(kwargs)

        block_updates = incoming.pop("blocks", None)
        self.metadata.update(incoming)

        if block_updates is None:
            if self._manifest:
                self._write_run_manifest()
            return
        if not isinstance(block_updates, Mapping):
            raise TypeError("metadata['blocks'] must be a mapping.")
        if self.data_manager._chunk_files():
            self._ensure_block_metadata_from_chunks()

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
                self._apply_user_block_metadata(block_name, existing)

        if self._manifest:
            self._write_run_manifest()

    def close(self) -> None:
        """Release this manager's run-directory lock without running it."""
        if hasattr(self, "_run_lock"):
            self._run_lock.release()

    def __enter__(self) -> "SweepManager":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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

                for key in _data_block_keys(store):

                    block_name = key.strip("/")
                    if block_name in self.block_metadata:
                        self._apply_user_block_metadata(
                            block_name,
                            self.block_metadata[block_name],
                        )
                        continue

                    try:
                        metadata_json = (
                            store.get_storer(key)
                            .attrs.block_metadata_json
                        )
                    except AttributeError:
                        metadata_json = None

                    if metadata_json:
                        block = BlockMetadata.from_dict(
                            json.loads(metadata_json),
                            default_key=block_name,
                        )
                        self._apply_user_block_metadata(block_name, block)
                        self.block_metadata[block_name] = block
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

    def _build_sweep_metadata(
        self,
        *,
        drop_columns: Mapping[str, Sequence[str]] | None = None,
    ) -> SweepMetadata:
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
        extra.setdefault("pynst_run_uuid", self.run_uuid)
        extra.setdefault("pynst_contract_sha256", self.contract_sha256)
        extra.setdefault("pynst_manifest_version", RUN_MANIFEST_VERSION)
        extra.setdefault(
            "pynst_result_block_names",
            list(self._expected_result_keys or tuple(self.block_metadata)),
        )
        extra.setdefault(
            "pynst_block_mode",
            self._block_mode or self.data_manager.block_mode or "mapping",
        )

        metadata = SweepMetadata(
            measurement_name=self.meas_name,
            nested_sweep_levels=list(self.param_col_names),
            blocks={
                name: BlockMetadata.from_dict(block.to_dict())
                for name, block in self.block_metadata.items()
            },
            created_at=self.metadata.get("created_at"),
            merged_at=datetime.now().isoformat(),
            resume_enabled=self.resume,
            schema_version=2,
            extra=extra,
        )
        for raw_name, columns in (drop_columns or {}).items():
            block_name = _normalise_block_name(str(raw_name))
            block = metadata.blocks.get(block_name)
            if block is None:
                continue
            removed = {str(column) for column in columns}
            block.dvars = [name for name in block.dvars if name not in removed]
            if not block.dvars:
                raise ValueError(
                    f"drop_columns would remove every dependent variable "
                    f"from block {block_name!r}."
                )
            for name in removed:
                if name not in block.ivars:
                    block.variables.pop(name, None)
        metadata.validate()
        return metadata

    def _write_metadata(
        self,
        store: pd.HDFStore,
        *,
        drop_columns: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        metadata = self._build_sweep_metadata(drop_columns=drop_columns)

        # Direct block attributes make a block self-describing. The central
        # config remains the authoritative complete file schema.
        for key in _data_block_keys(store):
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
            log_rows = []
            for sequence_index, (chunk, status) in sorted(
                self.log_dict.items()
            ):
                params = self.param_list[sequence_index]
                log_rows.append(
                    {
                        **{
                            f"parameter:{name}": self._log_value(params[name])
                            for name in self.param_col_names
                        },
                        "pynst:chunk_file": chunk,
                        "pynst:status": status,
                        "pynst:sequence_index": sequence_index,
                    }
                )
            log_frame = pd.DataFrame(log_rows)
            store.put(
                "/__metadata__/log",
                log_frame,
                # The diagnostic log is always read as one small frame.
                # Fixed storage accepts arbitrary user parameter names
                # without PyTables NaturalNameWarning noise.
                format="fixed",
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
        store.root._v_attrs.pynst_block_names_json = json.dumps(
            list(self._expected_result_keys or tuple(self.block_metadata)),
            separators=(",", ":"),
        )

    def _validate_merge_target_location(
        self,
        merged_file: str | Path,
    ) -> Path:
        target = Path(merged_file).expanduser().resolve()
        reserved_artifacts = {
            self.contract_file.resolve(),
            self.manifest_file.resolve(),
            self.log_file.resolve(),
            self.errlog_file.resolve(),
        }
        reserved_artifacts.update(
            path.with_name(path.name + ".tmp")
            for path in tuple(reserved_artifacts)
        )
        if target == self.output_dir.resolve():
            raise ValueError("Merged output must be a file, not the run directory.")
        if target in reserved_artifacts:
            raise ValueError(
                f"Merged output {target} would overwrite a PyNST run artifact."
            )
        if (
            target.parent == self.output_dir.resolve()
            and target.name.startswith("chunk_")
            and target.suffix.lower() == ".h5"
        ):
            raise ValueError(
                f"Merged output {target} uses the reserved chunk namespace."
            )
        return target

    def _validate_merge_target_ready(
        self,
        merged_file: str | Path,
        *,
        overwrite: bool,
    ) -> Path:
        """Check target policy without creating a directory or temp file."""
        target = self._validate_merge_target_location(merged_file)
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"Output file {target} already exists. Set "
                "overwrite=True to rebuild it."
            )
        return target

    @staticmethod
    def _create_merge_temporary(target: Path) -> Path:
        """Create a private temporary file beside a validated target."""
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".pynst_tmp",
            dir=target.parent,
        )
        os.close(descriptor)
        return Path(temporary_name)

    def _validate_retirement_target(self, merged_file: str | Path) -> Path:
        target = self._validate_merge_target_location(merged_file)
        run_directory = self.output_dir.resolve()
        if target == run_directory or run_directory in target.parents:
            raise ValueError(
                "remove_chunks=True requires an archive target outside the "
                "run directory; otherwise a later resume=False run could "
                "delete the only retained measurement artifact."
            )
        return target

    @staticmethod
    def _merge_target_lock(merged_file: str | Path) -> _RunLock:
        target = Path(merged_file).expanduser().resolve()
        digest = hashlib.sha256(
            _lock_identity(target).encode("utf-8")
        ).hexdigest()
        return _RunLock(
            target.parent / ".pynst_merge_locks" / f"{digest}.lock"
        )

    def _prepare_chunks_for_merge(
        self,
        *,
        require_complete: bool,
    ) -> list[Path]:
        if self._manifest.get("state") == "archived":
            raise RuntimeError("Archived PyNST runs cannot be merged again.")
        try:
            self.data_manager.finalize()
        except Exception as error:
            raise StorageCommitError(
                f"Could not finalize buffered data before merge: {error}"
            ) from error
        self._reconcile_persistence()
        completed = {
            int(sequence)
            for record in self._manifest.get("chunks", [])
            for sequence in record.get("sequence_indices", [])
        }
        expected = set(range(len(self.param_list)))
        if require_complete and completed != expected:
            missing = sorted(expected - completed)
            raise RuntimeError(
                "Cannot merge incomplete sweep; missing sequence indices: "
                f"{missing[:20]!r}"
                + (" ..." if len(missing) > 20 else "")
            )
        records = sorted(
            self._manifest.get("chunks", []),
            key=lambda record: int(record["index"]),
        )
        chunk_files = [
            self.output_dir / str(record["filename"])
            for record in records
        ]
        if not chunk_files:
            raise FileNotFoundError("No committed chunk files found.")
        return chunk_files

    def _archive_after_merge(
        self,
        target: Path,
        chunk_files: Sequence[Path],
    ) -> None:
        from ..data.dataset import GenericSweepDataset

        GenericSweepDataset(target).validate_storage(deep=True)
        self._manifest["state"] = "archived"
        self._manifest["archived_artifact"] = {
            "path": str(target),
            "sha256": _sha256_file(target),
            "size": target.stat().st_size,
            "archived_at": datetime.now().isoformat(),
            "retired_chunks": [path.name for path in chunk_files],
        }
        self._write_run_manifest()
        for chunk_file in chunk_files:
            if chunk_file.exists():
                chunk_file.unlink()

    @staticmethod
    def _streaming_min_itemsize(
        chunk_files: Sequence[Path],
        projected_columns: Mapping[str, Sequence[str]],
    ) -> dict[str, dict[str, int]]:
        """Determine stable string widths before the first table append."""
        required: dict[str, dict[str, int]] = defaultdict(dict)
        for chunk_file in chunk_files:
            with pd.HDFStore(chunk_file, mode="r") as source:
                for key in _data_block_keys(source):
                    storer = source.get_storer(key)
                    if storer.is_table:
                        frames: Iterable[pd.DataFrame] = source.select(
                            key,
                            chunksize=_STREAMING_MERGE_BATCH_ROWS,
                        )
                    else:
                        frames = (source[key],)
                    for frame in frames:
                        columns_to_drop = projected_columns.get(key, ())
                        if columns_to_drop:
                            frame = frame.drop(
                                columns=list(columns_to_drop),
                                errors="ignore",
                            )
                        flat = frame.reset_index()
                        block_sizes = required[key]
                        for column in flat.columns:
                            series = flat[column]
                            if series.dtype != np.dtype("object"):
                                continue
                            non_missing = series.dropna()
                            observed = max(
                                (
                                    len(str(value).encode("utf-8"))
                                    for value in non_missing
                                ),
                                default=0,
                            )
                            block_sizes[str(column)] = max(
                                block_sizes.get(str(column), 128),
                                observed,
                            )
        return dict(required)

    @staticmethod
    def _fixed_merge_memory_estimate(
        chunk_files: Sequence[Path],
        projected_columns: Mapping[str, Sequence[str]],
    ) -> tuple[str, int, int]:
        """Estimate peak RAM for the largest block of a fixed merge.

        Table chunks are scanned in bounded slices so object payloads and
        MultiIndex memory are included in ``DataFrame.memory_usage(deep=True)``.
        Legacy fixed chunks are read one chunk at a time.  The peak estimate
        accounts for the source frame list, concatenated frame, HDF
        serialisation, index validation and a fixed pandas/PyTables overhead.
        """
        logical_bytes: dict[str, int] = defaultdict(int)
        rows_by_block: dict[str, int] = defaultdict(int)
        for chunk_file in chunk_files:
            with pd.HDFStore(chunk_file, mode="r") as source:
                for key in _data_block_keys(source):
                    storer = source.get_storer(key)
                    if storer.is_table:
                        frames: Iterable[pd.DataFrame] = source.select(
                            key,
                            chunksize=_FIXED_MERGE_SCAN_ROWS,
                        )
                    else:
                        frames = (source[key],)
                    for frame in frames:
                        columns_to_drop = projected_columns.get(key, ())
                        if columns_to_drop:
                            frame = frame.drop(
                                columns=list(columns_to_drop),
                                errors="ignore",
                            )
                        logical_bytes[key.strip("/")] += int(
                            frame.memory_usage(
                                index=True,
                                deep=True,
                            ).sum()
                        )
                        rows_by_block[key.strip("/")] += len(frame)

        if not logical_bytes:
            raise ValueError("No measurement blocks found in committed chunks.")
        required_by_block = {
            name: (
                int(np.ceil(block_bytes * _FIXED_MERGE_MEMORY_FACTOR))
                + rows_by_block[name]
                * _FIXED_MERGE_INDEX_OVERHEAD_PER_ROW
                + _FIXED_MERGE_BASE_OVERHEAD_BYTES
            )
            for name, block_bytes in logical_bytes.items()
        }
        block_name, required_bytes = max(
            required_by_block.items(),
            key=lambda item: item[1],
        )
        block_bytes = logical_bytes[block_name]
        return block_name, block_bytes, required_bytes

    def _ensure_fixed_merge_memory(
        self,
        chunk_files: Sequence[Path],
        projected_columns: Mapping[str, Sequence[str]],
    ) -> tuple[str, int, int, int]:
        """Reject a fixed merge that cannot retain safe RAM headroom."""
        try:
            block_name, block_bytes, required_bytes = (
                self._fixed_merge_memory_estimate(
                    chunk_files,
                    projected_columns,
                )
            )
        except MemoryError as error:
            raise MemoryError(
                "Could not estimate fixed-merge memory without exhausting "
                "RAM. Retry with merge(..., strategy='streaming')."
            ) from error

        available_bytes = _available_memory_bytes()
        if available_bytes is None:
            raise MemoryError(
                "Fixed merge cannot determine currently available physical "
                "memory on this platform, so its allocation safety cannot "
                "be verified. Retry with merge(..., strategy='streaming')."
            )

        reserve_bytes = max(
            _FIXED_MERGE_MIN_RESERVE_BYTES,
            int(available_bytes * _FIXED_MERGE_RESERVE_FRACTION),
        )
        usable_bytes = max(0, available_bytes - reserve_bytes)
        if required_bytes > usable_bytes:
            raise MemoryError(
                "Fixed merge is estimated to need "
                f"{_format_bytes(required_bytes)} working memory for block "
                f"{block_name!r} ({_format_bytes(block_bytes)} logical "
                f"source data), but only {_format_bytes(available_bytes)} "
                "is currently available and "
                f"{_format_bytes(reserve_bytes)} is retained as safety "
                "headroom. Retry with merge(..., strategy='streaming'); "
                "source chunks were not modified."
            )
        return (
            block_name,
            block_bytes,
            required_bytes,
            available_bytes,
        )

    def partial_merge(
        self,
        merged_file: str | Path,
        remove_chunks: bool = False,
        force_merge_into_existing: bool = False,
        drop_columns: Mapping[str, Sequence[str]] | None = None,
        *,
        require_complete: bool = True,
    ) -> None:
        """Deprecated alias for ``merge(strategy='streaming')``."""
        warnings.warn(
            "partial_merge() is deprecated; use merge(..., "
            "strategy='streaming', overwrite=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.merge(
            merged_file,
            remove_chunks=remove_chunks,
            strategy="streaming",
            overwrite=force_merge_into_existing,
            require_complete=require_complete,
            drop_columns=drop_columns,
        )

    def _merge_streaming_locked(
        self,
        merged_file: str | Path,
        *,
        remove_chunks: bool,
        overwrite: bool,
        drop_columns: Mapping[str, Sequence[str]] | None,
        require_complete: bool,
    ) -> None:
        """Stream chunks into one table-format HDF file.

        The output is built in a temporary file and atomically moved into
        place. ``overwrite=True`` rebuilds an existing target; it does not
        append duplicate chunks to it. ``drop_columns`` projects named blocks
        to a stable schema while reading without modifying the source chunks.
        Block names may be supplied with or without a leading slash, and
        columns absent from a block are ignored.
        """
        if remove_chunks and not require_complete:
            raise ValueError(
                "remove_chunks=True requires a complete, validated merge."
            )
        if remove_chunks:
            self._validate_retirement_target(merged_file)
        target = self._validate_merge_target_ready(
            merged_file,
            overwrite=overwrite,
        )
        chunk_files = self._prepare_chunks_for_merge(
            require_complete=require_complete
        )
        # Validate the requested projection before creating a temporary file
        # or scanning all chunk payloads.
        self._build_sweep_metadata(drop_columns=drop_columns)

        projected_columns = {
            "/" + block_name.strip("/"): tuple(columns)
            for block_name, columns in (drop_columns or {}).items()
        }
        min_itemsize = self._streaming_min_itemsize(
            chunk_files,
            projected_columns,
        )
        temporary = self._create_merge_temporary(target)

        progress: Any | None = None
        try:
            progress = tqdm(
                total=len(chunk_files),
                desc="Merging chunks",
                leave=True,
            )
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
                        for key in _data_block_keys(source):
                            storer = source.get_storer(key)
                            if storer.is_table:
                                frames = source.select(
                                    key,
                                    chunksize=_STREAMING_MERGE_BATCH_ROWS,
                                )
                            else:
                                frames = (source[key],)
                            for frame in frames:
                                columns_to_drop = projected_columns.get(
                                    key,
                                    (),
                                )
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
                                    min_itemsize=min_itemsize.get(key, {}),
                                )
                    progress.update(1)

                self._write_metadata(
                    output,
                    drop_columns=drop_columns,
                )

            from ..data.dataset import GenericSweepDataset

            GenericSweepDataset(temporary).validate_storage(deep=True)
            _fsync_file(temporary)
            _replace_file(temporary, target)

        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise
        finally:
            if progress is not None:
                progress.close()

        if remove_chunks:
            self._archive_after_merge(target, chunk_files)

        print(
            f"Streaming merge done -> {target}, "
            f"remove_chunks={remove_chunks}"
        )

    def merge(
        self,
        merged_file: str | Path,
        remove_chunks: bool = False,
        *,
        strategy: MergeStrategy = "fixed",
        overwrite: bool = False,
        require_complete: bool = True,
        drop_columns: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Merge committed chunks using a fixed or streaming strategy.

        ``strategy='fixed'`` is the default and merges one complete block in
        memory before writing it once.  A conservative RAM preflight runs
        before the output file is created.  Pandas extension dtypes that fixed
        HDF cannot represent transparently use table storage for that block.

        ``strategy='streaming'`` appends bounded source batches to HDF tables
        and never materialises a complete result block.  Peak RAM still
        depends on the configured source chunk size and the exact index set
        retained by deep output validation.
        """
        if not isinstance(strategy, str) or strategy not in (
            "fixed",
            "streaming",
        ):
            raise ValueError(
                "merge strategy must be 'fixed' or 'streaming', found "
                f"{strategy!r}."
            )
        self._ensure_storage_usable()
        self._validate_merge_target_location(merged_file)
        if remove_chunks:
            self._validate_retirement_target(merged_file)
        self._acquire_current_run()
        target_lock = self._merge_target_lock(merged_file)
        try:
            target_lock.acquire()
            if strategy == "fixed":
                self._merge_fixed_locked(
                    merged_file,
                    remove_chunks=remove_chunks,
                    overwrite=overwrite,
                    require_complete=require_complete,
                    drop_columns=drop_columns,
                )
            else:
                self._merge_streaming_locked(
                    merged_file,
                    remove_chunks=remove_chunks,
                    overwrite=overwrite,
                    drop_columns=drop_columns,
                    require_complete=require_complete,
                )
        finally:
            target_lock.release()
            self._run_lock.release()

    def _merge_fixed_locked(
        self,
        merged_file: str | Path,
        *,
        remove_chunks: bool,
        overwrite: bool,
        require_complete: bool,
        drop_columns: Mapping[str, Sequence[str]] | None,
    ) -> None:
        """Merge each complete block in memory into an optimized HDF file.

        Unlike the historical implementation, this method preserves the full
        MultiIndex by never using ``ignore_index=True``. Most blocks use fixed
        storage; pandas extension dtypes transparently use table storage.
        """
        if remove_chunks and not require_complete:
            raise ValueError(
                "remove_chunks=True requires a complete, validated merge."
            )
        if remove_chunks:
            self._validate_retirement_target(merged_file)
        target = self._validate_merge_target_ready(
            merged_file,
            overwrite=overwrite,
        )
        chunk_files = self._prepare_chunks_for_merge(
            require_complete=require_complete
        )
        merge_metadata = self._build_sweep_metadata(
            drop_columns=drop_columns
        )
        projected_columns = {
            "/" + block_name.strip("/"): tuple(columns)
            for block_name, columns in (drop_columns or {}).items()
        }
        (
            largest_block,
            _logical_bytes,
            required_bytes,
            available_bytes,
        ) = self._ensure_fixed_merge_memory(
            chunk_files,
            projected_columns,
        )
        print(
            "Fixed merge RAM preflight: "
            f"{_format_bytes(required_bytes)} estimated for block "
            f"{largest_block!r}; {_format_bytes(available_bytes)} "
            "currently available."
        )

        temporary = self._create_merge_temporary(target)
        block_names = list(merge_metadata.blocks)

        progress = None
        try:
            progress = tqdm(
                block_names,
                desc="Merging by block",
                leave=True,
            )
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
                                frame = source[hdf_key]
                                columns_to_drop = projected_columns.get(
                                    hdf_key,
                                    (),
                                )
                                if columns_to_drop:
                                    frame = frame.drop(
                                        columns=list(columns_to_drop),
                                        errors="ignore",
                                    )
                                frames.append(frame)

                    if not frames:
                        continue

                    merged = pd.concat(
                        frames,
                        axis=0,
                        join="outer",
                    )

                    has_extension_dtype = any(
                        isinstance(
                            merged.index.get_level_values(level).dtype,
                            pd.api.extensions.ExtensionDtype,
                        )
                        for level in range(merged.index.nlevels)
                    ) or any(
                        isinstance(dtype, pd.api.extensions.ExtensionDtype)
                        for dtype in merged.dtypes
                    )
                    if has_extension_dtype:
                        # pandas fixed HDF storage cannot serialise a
                        # MultiIndex containing extension dtypes (notably
                        # CategoricalDtype).  Table storage preserves those
                        # semantics and remains transparent to the dataset API.
                        output.put(
                            block_name,
                            merged,
                            format="table",
                            data_columns=list(merged.index.names),
                        )
                    else:
                        output.put(
                            block_name,
                            merged,
                            format="fixed",
                        )
                    del merged, frames
                    gc.collect()

                self._write_metadata(
                    output,
                    drop_columns=drop_columns,
                )

            from ..data.dataset import GenericSweepDataset

            GenericSweepDataset(temporary).validate_storage(deep=True)
            _fsync_file(temporary)
            _replace_file(temporary, target)

        except MemoryError as error:
            if temporary.exists():
                temporary.unlink()
            raise MemoryError(
                "Fixed merge exhausted memory despite the conservative "
                "preflight. Retry with merge(..., strategy='streaming'); "
                "source chunks were not modified."
            ) from error
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise
        finally:
            if progress is not None:
                progress.close()

        if remove_chunks:
            self._archive_after_merge(target, chunk_files)

        print(f"Fixed merge done -> {target}")

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

__all__ = [
    "CriticalMeasurementError",
    "DataManager",
    "OnDiskChunkManager",
    "StorageCommitError",
    "SweepManager",
]
