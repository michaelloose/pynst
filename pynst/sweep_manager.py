import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import json
import traceback
from datetime import datetime

from collections import defaultdict
from typing import List, Any, Callable, Optional, Dict

from tqdm.auto import tqdm


# -------------------------------
# Custom Exceptions
# -------------------------------
class CriticalMeasurementError(Exception):
    """Schwerwiegender Messfehler: Bricht gesamte Messung ab."""
    pass


# -------------------------------
# DataManager (Abstract)
# -------------------------------
class DataManager:
    def add_worker_data(self, block_list: List[Any], metadata: Any, param_keys: List[Any]) -> None:
        raise NotImplementedError

    def finalize(self) -> None:
        raise NotImplementedError

    def get_results(self, include_nan: bool = True) -> List[Any]:
        raise NotImplementedError

    def remove_incomplete_params_from_chunks(self, incomplete_keys: List[tuple], param_col_names: List[str]) -> None:
        """
        Entfernt selektiv Zeilen aus bereits geschriebenen Chunks,
        die zu unvollständigen Param-Kombinationen (incomplete_keys) gehören.
        """
        raise NotImplementedError


# -------------------------------
# OnDiskChunkManager
# -------------------------------
class OnDiskChunkManager(DataManager):
    """
    - Verwaltet DataFrames in Chunks (mit flush, .tmp -> .h5 rename).
    - remove_incomplete_params_from_chunks: Filtert nur die 'schlechten' Param-Kombinationen raus.
    - Bietet optionalen Resume, ohne alle Chunks zu löschen.
    """
    def __init__(
        self,
        chunk_size: int = 10,
        output_dir: str = "chunks",
        resume: bool = False,
    ):
        self.chunk_size = chunk_size
        self.output_dir = Path(output_dir)
        self.resume = resume

        # Verzeichnis vorbereiten
        if not self.resume and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.worker_count = 0
        # Sammle DataFrames nach "block_idx"
        self.df_blocks = defaultdict(list)
        # Speichert ParamKeys der aktuellen Chunk-Sammlung
        self.pending_param_keys = []
        # Bestimme bereits existierende Chunks
        self.chunks_written = self._get_max_existing_chunk_index() + 1 if resume else 0

    def _get_max_existing_chunk_index(self) -> int:
        indices = []
        for f in self.output_dir.glob("chunk_*.h5"):
            try:
                idx = int(f.stem.split("_")[1])
                indices.append(idx)
            except (IndexError, ValueError):
                pass
        return max(indices) if indices else -1

    def add_worker_data(self, block_list: List[Any], metadata: Any, param_keys: List[Any]) -> None:
        for block_idx, block in enumerate(block_list):
            if isinstance(block, pd.DataFrame):
                self.df_blocks[block_idx].append(block)

        self.pending_param_keys.extend(param_keys)
        self.worker_count += 1

        if self.worker_count >= self.chunk_size:
            self._flush_chunk()

    def _flush_chunk(self):
        if not self.df_blocks:
            return

        tmp_filename = f"chunk_{self.chunks_written}.tmp.h5"
        tmp_path = self.output_dir / tmp_filename

        # schreibe DataFrames in .tmp
        with pd.HDFStore(tmp_path, mode='w') as store:
            for block_idx, df_list in self.df_blocks.items():
                combined = pd.concat(df_list, axis=0, join='outer')
                store.put(f"block_{block_idx}", combined, format='table')

        final_filename = f"chunk_{self.chunks_written}.h5"
        final_path = self.output_dir / final_filename
        os.rename(tmp_path, final_path)

        done_keys = self.pending_param_keys[:]
        self.pending_param_keys.clear()
        self.df_blocks.clear()
        self.worker_count = 0
        chunk_idx = self.chunks_written
        self.chunks_written += 1

        # Callback -> Loggen der param_keys
        self._on_chunk_written(chunk_idx, final_filename, done_keys)

    def _on_chunk_written(self, chunk_idx: int, chunk_filename: str, param_keys: List[Any]):
        """Wird vom SweepManager überschrieben, um ins Log zu schreiben."""
        pass

    def finalize(self) -> None:
        if self.worker_count > 0:
            self._flush_chunk()

    def get_results(self, include_nan: bool = True) -> List[Any]:
        """
        Lädt alle Chunks, konkateniert blockweise.
        """
        block_map = defaultdict(list)
        chunk_files = sorted(self.output_dir.glob("chunk_*.h5"))
        for cf in chunk_files:
            with pd.HDFStore(cf, 'r') as store:
                for key in store.keys():
                    idx = int(key.strip("/").split("_")[1])
                    df = store[key]
                    block_map[idx].append(df)

        max_block_idx = max(block_map.keys()) if block_map else -1
        out_list = []
        for i in range(max_block_idx + 1):
            df_list = block_map.get(i)
            if not df_list:
                # kein block => None
                if include_nan:
                    out_list.append(None)
            else:
                combined = pd.concat(df_list, axis=0, join='outer')
                if combined.isna().all().all() and not include_nan:
                    continue
                out_list.append(combined)

        return out_list

    def remove_incomplete_params_from_chunks(self, incomplete_keys: List[tuple], param_col_names: List[str]) -> None:
        """
        Öffnet jede chunk_*.h5 und entfernt Zeilen, deren Parameter-Kombination in incomplete_keys steht.
        Mit Fortschrittsbalken.
        """
        chunk_files = sorted(self.output_dir.glob("chunk_*.h5"))
        pbar = tqdm(total=len(chunk_files), desc="Removing incomplete data", leave=True)
        for chunk_file in chunk_files:
            self._filter_incomplete_in_file(chunk_file, incomplete_keys, param_col_names)
            pbar.update(1)
        pbar.close()

    def _filter_incomplete_in_file(self, chunk_file: Path, incomplete_keys: List[tuple], param_col_names: List[str]) -> None:
        tmp_path = chunk_file.with_suffix(".tmp.h5")
        removed_something = False

        with pd.HDFStore(chunk_file, 'r') as store_in, pd.HDFStore(tmp_path, 'w') as store_out:
            for key in store_in.keys():
                df = store_in[key]
                if df.index.nlevels < len(param_col_names):
                    # Falls MultiIndex nicht so groß ist wie param_col_names => 
                    # Hier ggf. anpassen oder ignorieren. Minimal:
                    pass

                keep_mask = [True]*len(df)
                n_params = len(param_col_names)
                for i, idx_tuple in enumerate(df.index):
                    # param_tuple = erster n_params Teil des Index
                    param_tuple = idx_tuple[:n_params]
                    if param_tuple in incomplete_keys:
                        keep_mask[i] = False
                        removed_something = True

                filtered_df = df.loc[keep_mask]
                if not filtered_df.empty:
                    store_out.put(key, filtered_df, format='table')

        if removed_something:
            chunk_file.unlink()
            tmp_path.rename(chunk_file)
        else:
            tmp_path.unlink()


# -------------------------------
# SweepManager
# -------------------------------
class SweepManager:
    """
    Manages chunk-based measurement sweeps across a multi-dimensional parameter space.

    This class processes parameter combinations provided as a pandas MultiIndex, runs a
    user-defined measurement function on each combination, and stores the results in chunked
    HDF5 files. It maintains a log file that tracks completed or failed parameter sets,
    allowing you to resume an interrupted sweep by skipping already finished entries and
    removing incomplete ones.

    Args:
        measurement_func (Callable[[dict], List[pd.DataFrame]]):
            A function that takes a dict of parameters and returns one or more DataFrames
            containing measurement results.
        ivars (pd.MultiIndex):
            A MultiIndex representing all parameter combinations to be swept.
        meas_name (str):
            A name or path segment used for output directories and files.
        resume (bool, optional):
            Whether to resume from a previous log. Defaults to False.
        chunk_size (int, optional):
            The number of parameter combinations to store per chunk file. Defaults to 10.
        param_col_names (List[str], optional):
            Column names for the parameters. If None, uses `ivars.names`.
        critical_callback (Callable[[Exception], None], optional):
            A callback function for handling critical exceptions during measurement.

    Attributes:
        log_file (Path):
            Path to the TSV log tracking each parameter combination and its status.
        errlog_file (Path):
            Path to a log file containing stack traces for any exceptions encountered.
        param_list (List[dict]):
            A list of parameter dictionaries derived from the MultiIndex.

    Methods:
        run():
            Executes the measurement process over all parameter combinations. Skips
            completed ones if `resume` is True. Logs successful or failed attempts,
            and aborts on critical errors.
        get_results(include_nan: bool = True) -> List[Any]:
            Retrieves all results (optionally including NaN entries) from the chunk files.
        partial_merge(merged_file: str, remove_chunks: bool = False):
            Merges all chunked HDF5 files into a single file, with an option to remove the
            chunk files afterward. Useful for consolidating results in one place.

    Raises:
        ValueError:
            If the provided MultiIndex is not unique.
        RuntimeError:
            If `run()` is called multiple times without resetting or creating a new instance.

    Example:
        >>> sm = SweepManager(measurement_func=my_measurement, ivars=my_index, meas_name="test_run")
        >>> sm.run()
        >>> sm.partial_merge("merged_results.h5", remove_chunks=True)
    """

    def __init__(
        self,
        measurement_func: Callable[[dict], List[pd.DataFrame]],
        ivars: pd.MultiIndex,
        meas_name: str, 
        resume: bool = False,
        chunk_size: int = 10,
        param_col_names: Optional[List[str]] = None,
        critical_callback: Optional[Callable[[Exception], None]] = None
    ):
        
        if not ivars.is_unique:
            raise ValueError("Sweep index contains duplicate values. Please sanitize, e.g. using .unique() before passing")
        
        output_dir = Path(meas_name)
        self.log_file = output_dir/"simlog.tsv"
        self.errlog_file = output_dir/"traceback.log"

        self.measurement_func = measurement_func
        self.multi_index = ivars
        self.param_col_names = param_col_names or list(ivars.names)
        self.resume = resume
        self.critical_callback = critical_callback
        self._already_run = False  # Guard

        self.param_list = self._create_param_list()
        self.data_manager = OnDiskChunkManager(
            chunk_size=chunk_size,
            output_dir=output_dir,
            resume=resume
        )

        # Callback nach Flush
        def on_chunk_written(chunk_idx, chunk_filename, param_keys):
            with open(self.log_file, 'a') as f:
                for pk in param_keys:
                    pk_str = "\t".join(map(str, pk))
                    line = f"{pk_str}\t{chunk_filename}\tComplete\n"
                    f.write(line)

        self.data_manager._on_chunk_written = on_chunk_written

        # Log vorbereiten/laden
        if resume and self.log_file.exists():
            self.log_dict = self._load_log()
            # incomplete/failed => filtern
            incomplete_keys = [k for k, (ch, st) in self.log_dict.items() if st != "Complete"]
            if incomplete_keys:
                print(f"Resume: removing partial data for {len(incomplete_keys)} combos.")
                self.data_manager.remove_incomplete_params_from_chunks(incomplete_keys, self.param_col_names)
                # Incomplete keys aus log_dict entfernen
                for k in incomplete_keys:
                    del self.log_dict[k]
            else:
                print("Resume: all combos in log are complete.")
        else:
            self.log_dict = {}
            if self.log_file.exists() and not resume:
                self.log_file.unlink()
            if not self.log_file.exists():
                with open(self.log_file, 'w') as f:
                    f.write("# " + json.dumps(list(ivars.names)) + "\n")
                    f.write("\t".join(ivars.names) + "\tchunk_file\tstatus\n")

    def _create_param_list(self) -> List[dict]:
        out = []
        for tup in self.multi_index:
            d = {n: v for n, v in zip(self.multi_index.names, tup)}
            out.append(d)
        return out

    def _load_log(self) -> Dict[tuple, tuple]:
        """
        param_tuple -> (chunk_file, status)
        """
        res = {}
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                param_len = len(self.multi_index.names)
                key_tuple = tuple(parts[:param_len])
                chunk_file = parts[param_len]
                status = parts[param_len + 1]
                res[key_tuple] = (chunk_file, status)
        return res

    def run(self) -> None:
        # Guard, damit man nicht versehentlich doppelt run() aufruft
        if self._already_run:
            raise RuntimeError(
                "SweepManager.run() was already called. To run again, you must create a new SweepManager "
                "or explicitly reset the existing one."
            )
        self._already_run = True

        total = len(self.param_list)
        pbar = tqdm(total=total, desc="Sweep", leave=True)

        for i, param_dict in enumerate(self.param_list):
            key_tuple = tuple(str(param_dict[n]) for n in self.multi_index.names)
            if key_tuple in self.log_dict and self.log_dict[key_tuple][1] == "Complete":
                # Überspringen
                pbar.update(1)
                continue

            desc = " | ".join(f"{k}={v}" for k, v in param_dict.items())
            #desc += f" || {i+1}/{total}"
            pbar.set_description(desc)

            # Messung
            try:
                dfs = self.measurement_func(param_dict)
                tagged = self._tag_with_params(dfs, param_dict)
                self.data_manager.add_worker_data(tagged, metadata=None, param_keys=[key_tuple])
            except CriticalMeasurementError as ce:
                pbar.write(f"Critical error: {ce}")
                self.data_manager.finalize()
                if self.critical_callback:
                    self.critical_callback(ce)
                # Dump Traceback: 
                with open(self.errlog_file, 'a') as f:
                    f.write(f"--- Critical Exception at Index {desc} Timestamp: {datetime.now().isoformat()} ---\n")
                    traceback.print_exc(file=f)
                    f.write("\n")

                return  # Abbruch
            except Exception as ex:
                pbar.write(f"Exception occured, measurement skipped: {ex}")
                # Log => Failed
                with open(self.log_file, 'a') as f:
                    line = "\t".join(key_tuple) + "\tNA\tFailed\n"
                    f.write(line)
                # Dump Traceback: 
                with open(self.errlog_file, 'a') as f:
                    f.write(f"--- Exception at Index {desc} Timestamp: {datetime.now().isoformat()} ---\n")
                    traceback.print_exc(file=f)
                    f.write("\n") 
                            
                

            pbar.update(1)

        pbar.close()
        self.data_manager.finalize()
        print("Measurement run completed.")

    def _tag_with_params(self, dfs: List[pd.DataFrame], params: dict) -> List[pd.DataFrame]:
        """
        Fügt param-Spalten ein, reset_index, 
        MultiIndex => param_col_names + Ursprungs-Index
        """
        out = []
        for df in dfs:
            if not isinstance(df, pd.DataFrame):
                out.append(df)
                continue
            df_assigned = df.assign(**params)
            df_flat = df_assigned.reset_index()
            orig_names = df.index.names
            if all(x is None for x in orig_names):
                orig_names = [f"index_{i}" for i in range(df.index.nlevels)]
            new_index_names = self.param_col_names + list(orig_names)
            missing = [x for x in new_index_names if x not in df_flat.columns]
            # TODO Leere indizes werfen hier Fehler. Das muss behandelt werden, notfalls durch das erzwungene setzen eines Namens. Besser: Das Level leer lassen. sonst ist ein unnötiges Level im Endgültigen DF
            if missing:
                raise KeyError(f"Missing columns {missing} for new index.")
            df_new = df_flat.set_index(new_index_names)
            out.append(df_new)
        return out

    def get_results(self, include_nan: bool = True) -> List[Any]:
        return self.data_manager.get_results(include_nan=include_nan)

    def partial_merge(self, merged_file: str, remove_chunks: bool = False) -> None:
        """
        Liest alle Chunks, appends sie in 'merged_file', 
        zeigt dabei einen Fortschrittsbalken.
        """
        # TODO: Die Ausgabe dieser Funktion ist durch das Append unnötig groß. Das sollte irgendwie defragmentiert werden. Am besten ohne alles in den ram zu laden
        self.data_manager.finalize()
        chunk_files = sorted(Path(self.data_manager.output_dir).glob("chunk_*.h5"))
        pbar = tqdm(total=len(chunk_files), desc="Merging chunks", leave=True)
        with pd.HDFStore(merged_file, mode='a') as out_store:
            for cf in chunk_files:
                with pd.HDFStore(cf, 'r') as in_store:
                    for key in in_store.keys():
                        df = in_store[key]
                        out_store.append(key, df, format='table')
                if remove_chunks:
                    cf.unlink()
                pbar.update(1)
        pbar.close()
        print(f"Partial merge done -> {merged_file}, remove_chunks={remove_chunks}")


    def merge(self, merged_file: str, remove_chunks: bool = False) -> None:
        """
        ### Not Fully Tested!###
        Reads all chunk files, concatenates DataFrames by key, and writes them once in 'fixed' format.
        This avoids multiple append operations that bloat the output file.

        Args:
            merged_file (str): Path to the merged output HDF5 file.
            remove_chunks (bool): If True, deletes chunk files after merging.

        Raises:
            MemoryError: If concatenation of all chunks per key exceeds available RAM.
        """
        import gc
        from tqdm import tqdm
        import pandas as pd

        # Ensure all chunks are finalized
        self.data_manager.finalize()
        chunk_files = sorted(Path(self.data_manager.output_dir).glob("chunk_*.h5"))
        if not chunk_files:
            print("No chunk files found. Nothing to merge.")
            return

        # 1) Alle Keys aus den Chunk-Dateien ermitteln
        all_keys = set()
        for cf in chunk_files:
            with pd.HDFStore(cf, 'r') as store:
                all_keys.update(store.keys())

        print(f"Found {len(all_keys)} unique keys across {len(chunk_files)} chunk files.")

        # 2) Für jeden Key Daten aus allen Chunks laden und in einen DataFrame mergen
        with pd.HDFStore(merged_file, mode='w') as out_store:
            pbar = tqdm(all_keys, desc="Merging by key")
            for key in pbar:
                df_list = []
                for cf in chunk_files:
                    with pd.HDFStore(cf, 'r') as in_store:
                        if key in in_store.keys():
                            df_list.append(in_store[key])
                if df_list:
                    merged_df = pd.concat(df_list, ignore_index=True)
                    # optional: merged_df.drop_duplicates(...) bei Bedarf
                    # optional: Sortierung / Umwandlung / Kompression etc.
                    out_store.put(key, merged_df, format='fixed')
                    # Speicher frei geben
                    del merged_df, df_list
                    gc.collect()
            pbar.close()

        # 3) Chunk-Dateien entfernen
        if remove_chunks:
            for cf in chunk_files:
                cf.unlink()
            print("All chunk files have been removed.")

        print(f"Optimized merge done -> {merged_file}")
