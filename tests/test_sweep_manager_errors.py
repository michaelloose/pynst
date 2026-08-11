from __future__ import annotations

import pandas as pd
import pytest

from pynst import CriticalMeasurementError, SweepManager
from pynst import sweep_manager as sweep_manager_module


def test_tagging_error_aborts_before_next_measurement(tmp_path):
    calls: list[int] = []
    critical_errors: list[Exception] = []

    def measurement(params):
        calls.append(params["frequency"])
        return {
            "tuner_state_df": pd.DataFrame(
                {"frequency": [params["frequency"]]},
                index=pd.Index([0], name="tuner_state"),
            )
        }

    ivars = pd.MultiIndex.from_tuples(
        [(1,), (2,)],
        names=["frequency"],
    )
    manager = SweepManager(
        measurement_func=measurement,
        ivars=ivars,
        meas_name="fatal_tagging_error",
        output_root=tmp_path,
        chunk_size=10,
        resume=False,
        critical_callback=critical_errors.append,
    )

    manager.run()

    assert calls == [1]
    assert len(critical_errors) == 1
    assert "normalization/tagging failed" in str(critical_errors[0])
    assert "reuses variable names" in str(critical_errors[0])
    assert not list(manager.output_dir.glob("chunk_*.h5"))

    traceback_text = manager.errlog_file.read_text(encoding="utf-8")
    assert "Critical Exception" in traceback_text
    assert "reuses variable names" in traceback_text
    assert "Measurement run completed" not in traceback_text


def test_tagging_error_flushes_prior_buffered_measurements(tmp_path):
    calls: list[int] = []

    def measurement(params):
        value = params["frequency"]
        calls.append(value)
        if value == 2:
            return {
                "tuner_state_df": pd.DataFrame(
                    {"frequency": [value]},
                    index=pd.Index([0], name="tuner_state"),
                )
            }
        return {
            "tuner_state_df": pd.DataFrame(
                {"value": [42.0]},
                index=pd.Index([0], name="sample"),
            )
        }

    ivars = pd.MultiIndex.from_tuples(
        [(1,), (2,), (3,)],
        names=["frequency"],
    )
    manager = SweepManager(
        measurement_func=measurement,
        ivars=ivars,
        meas_name="fatal_tagging_error_after_valid_point",
        output_root=tmp_path,
        chunk_size=10,
        resume=False,
    )

    manager.run()

    assert calls == [1, 2]
    chunk_files = list(manager.output_dir.glob("chunk_*.h5"))
    assert len(chunk_files) == 1
    with pd.HDFStore(chunk_files[0], mode="r") as store:
        stored = store["tuner_state_df"]
    assert stored.index.get_level_values("frequency").tolist() == [1]


def test_measurement_error_remains_point_local(tmp_path):
    calls: list[int] = []

    def measurement(params):
        value = params["frequency"]
        calls.append(value)
        if value == 1:
            raise RuntimeError("transient point failure")
        return {
            "meas_df": pd.DataFrame(
                {"value": [42.0]},
                index=pd.Index([0], name="sample"),
            )
        }

    ivars = pd.MultiIndex.from_tuples(
        [(1,), (2,)],
        names=["frequency"],
    )
    manager = SweepManager(
        measurement_func=measurement,
        ivars=ivars,
        meas_name="recoverable_measurement_error",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    )

    manager.run()

    assert calls == [1, 2]
    assert len(list(manager.output_dir.glob("chunk_*.h5"))) == 1
    log_text = manager.log_file.read_text(encoding="utf-8")
    assert "1\tNA\tFailed" in log_text
    assert "2\tchunk_0.h5\tComplete" in log_text


def test_resume_contract_rejects_changed_grid_before_measurement(tmp_path):
    calls: list[int] = []

    def measurement(params):
        calls.append(params["frequency"])
        return {
            "meas": pd.DataFrame(
                {"value": [1.0]}, index=pd.Index([0], name="sample")
            )
        }

    first = SweepManager(
        measurement_func=measurement,
        ivars=pd.MultiIndex.from_tuples([(1,), (2,)], names=["frequency"]),
        meas_name="contract_grid",
        output_root=tmp_path,
        resume=False,
        chunk_size=1,
        expected_result_schema={
            "meas": {"index_names": ["sample"], "columns": ["value"]}
        },
    )
    first.run()
    assert calls == [1, 2]

    with pytest.raises(ValueError, match="sweep grid changed"):
        SweepManager(
            measurement_func=measurement,
            ivars=pd.MultiIndex.from_tuples([(1,), (3,)], names=["frequency"]),
            meas_name="contract_grid",
            output_root=tmp_path,
            resume=True,
            expected_result_schema={
                "meas": {"index_names": ["sample"], "columns": ["value"]}
            },
        )
    assert calls == [1, 2]


def test_resume_contract_rejects_declared_result_change_before_measurement(tmp_path):
    def measurement(params):
        return {
            "meas": pd.DataFrame(
                {"value": [1.0]}, index=pd.Index([0], name="sample")
            )
        }

    grid = pd.MultiIndex.from_tuples([(1,)], names=["frequency"])
    manager = SweepManager(
        measurement_func=measurement,
        ivars=grid,
        meas_name="contract_schema",
        output_root=tmp_path,
        resume=False,
        expected_result_schema={
            "meas": {"index_names": ["sample"], "columns": ["value"]}
        },
    )
    manager.run()

    with pytest.raises(ValueError, match="declared measurement-result schema"):
        SweepManager(
            measurement_func=measurement,
            ivars=grid,
            meas_name="contract_schema",
            output_root=tmp_path,
            resume=True,
            expected_result_schema={
                "meas": {"index_names": ["sample"], "columns": ["other"]}
            },
        )


def test_strict_resume_refuses_legacy_run_without_contract(tmp_path):
    output = tmp_path / "legacy"
    output.mkdir()
    (output / "simlog.tsv").write_text(
        '# ["frequency"]\nfrequency\tchunk_file\tstatus\n',
        encoding="utf-8",
    )
    grid = pd.MultiIndex.from_tuples([(1,)], names=["frequency"])

    with pytest.raises(RuntimeError, match="legacy sweep"):
        SweepManager(
            measurement_func=lambda params: {},
            ivars=grid,
            meas_name="legacy",
            output_root=tmp_path,
            resume=True,
        )


def test_mapping_block_order_is_irrelevant_within_a_run(tmp_path):
    def measurement(params):
        first = pd.DataFrame(
            {"value_a": [float(params["frequency"])]},
            index=pd.Index([0], name="sample"),
        )
        second = pd.DataFrame(
            {"value_b": [float(params["frequency"])]},
            index=pd.Index([0], name="sample"),
        )
        if params["frequency"] == 1:
            return {"a": first, "b": second}
        return {"b": second, "a": first}

    manager = SweepManager(
        measurement_func=measurement,
        ivars=pd.MultiIndex.from_tuples([(1,), (2,)], names=["frequency"]),
        meas_name="mapping_order",
        output_root=tmp_path,
        resume=False,
        chunk_size=1,
    )
    manager.run()

    assert len(list(manager.output_dir.glob("chunk_*.h5"))) == 2
    if manager.errlog_file.exists():
        assert "Critical Exception" not in manager.errlog_file.read_text(
            encoding="utf-8"
        )


def test_resume_accepts_schema_declaration_without_observed_dtypes(tmp_path):
    grid = pd.MultiIndex.from_tuples([(1,), (2,)], names=["frequency"])
    schema = {
        "a": {"index_names": ["sample"], "columns": ["value_a"]},
        "b": {"index_names": ["sample"], "columns": ["value_b"]},
    }

    def first_measurement(params):
        if params["frequency"] == 2:
            raise CriticalMeasurementError("intentional stop")
        return {
            "a": pd.DataFrame(
                {"value_a": [1.0]}, index=pd.Index([0], name="sample")
            ),
            "b": pd.DataFrame(
                {"value_b": [2.0]}, index=pd.Index([0], name="sample")
            ),
        }

    first = SweepManager(
        measurement_func=first_measurement,
        ivars=grid,
        meas_name="schema_without_dtypes",
        output_root=tmp_path,
        resume=False,
        chunk_size=1,
        expected_result_schema=schema,
    )
    first.run()

    resumed_calls: list[int] = []

    def resumed_measurement(params):
        resumed_calls.append(params["frequency"])
        # Deliberately reverse mapping insertion order.
        return {
            "b": pd.DataFrame(
                {"value_b": [2.0]}, index=pd.Index([0], name="sample")
            ),
            "a": pd.DataFrame(
                {"value_a": [1.0]}, index=pd.Index([0], name="sample")
            ),
        }

    resumed = SweepManager(
        measurement_func=resumed_measurement,
        ivars=grid,
        meas_name="schema_without_dtypes",
        output_root=tmp_path,
        resume=True,
        chunk_size=1,
        expected_result_schema={"b": schema["b"], "a": schema["a"]},
    )
    resumed.run()
    assert resumed_calls == [2]


def test_resume_before_first_result_does_not_infer_list_order_from_schema(tmp_path):
    grid = pd.MultiIndex.from_tuples([(1,)], names=["frequency"])
    schema = {
        f"block_{index}": {
            "index_names": ["sample"],
            "columns": [f"value_{index}"],
        }
        for index in range(11)
    }

    # Constructing the first manager creates the log and generic contract, but
    # no return container type has been observed yet.
    SweepManager(
        measurement_func=lambda params: [],
        ivars=grid,
        meas_name="resume_before_first_result",
        output_root=tmp_path,
        resume=False,
        expected_result_schema=schema,
    )

    resumed = SweepManager(
        measurement_func=lambda params: [
            pd.DataFrame(
                {f"value_{index}": [float(index)]},
                index=pd.Index([0], name="sample"),
            )
            for index in range(11)
        ],
        ivars=grid,
        meas_name="resume_before_first_result",
        output_root=tmp_path,
        resume=True,
        chunk_size=1,
        expected_result_schema=schema,
    )
    resumed.run()

    assert resumed._block_mode == "list"
    assert resumed._expected_result_keys == tuple(
        f"block_{index}" for index in range(11)
    )


def test_list_block_positions_remain_structural(tmp_path):
    calls: list[int] = []

    def measurement(params):
        calls.append(params["frequency"])
        a = pd.DataFrame(
            {"a": [1.0]}, index=pd.Index([0], name="sample")
        )
        b = pd.DataFrame(
            {"b": [2.0]}, index=pd.Index([0], name="sample")
        )
        return [a, b] if params["frequency"] == 1 else [b, a]

    manager = SweepManager(
        measurement_func=measurement,
        ivars=pd.MultiIndex.from_tuples([(1,), (2,), (3,)], names=["frequency"]),
        meas_name="list_positions",
        output_root=tmp_path,
        resume=False,
        chunk_size=1,
    )
    manager.run()

    assert calls == [1, 2]
    error_text = manager.errlog_file.read_text(encoding="utf-8")
    assert "changed" in error_text
    assert "dvars" in error_text or "columns" in error_text


def test_resume_rejects_contract_chunk_block_mode_mismatch(tmp_path):
    grid = pd.MultiIndex.from_tuples([(1,)], names=["frequency"])

    manager = SweepManager(
        measurement_func=lambda params: {
            "meas": pd.DataFrame(
                {"value": [1.0]}, index=pd.Index([0], name="sample")
            )
        },
        ivars=grid,
        meas_name="block_mode_mismatch",
        output_root=tmp_path,
        resume=False,
        chunk_size=1,
    )
    manager.run()
    chunk = next(manager.output_dir.glob("chunk_*.h5"))
    with pd.HDFStore(chunk, mode="a") as store:
        store.root._v_attrs.pynst_block_mode = "list"

    with pytest.raises(ValueError, match="block mode differs"):
        SweepManager(
            measurement_func=lambda params: {},
            ivars=grid,
            meas_name="block_mode_mismatch",
            output_root=tmp_path,
            resume=True,
        )


def test_previous_result_is_passed_during_normal_run(tmp_path):
    received: list[object] = []

    def measurement(params, previous_result):
        received.append(previous_result)
        return {
            "meas": pd.DataFrame(
                {"value": [float(params["frequency"])]},
                index=pd.Index([0], name="sample"),
            )
        }

    manager = SweepManager(
        measurement_func=measurement,
        ivars=pd.MultiIndex.from_tuples([(1,), (2,)], names=["frequency"]),
        meas_name="previous_normal",
        output_root=tmp_path,
        resume=False,
        chunk_size=10,
        provide_previous_result=True,
    )
    manager.run()

    assert received[0] is None
    previous = received[1]
    assert isinstance(previous, dict)
    assert previous["meas"].index.names == ["frequency", "sample"]
    assert previous["meas"].index.get_level_values("frequency").tolist() == [1]
    assert previous["meas"]["value"].tolist() == [1.0]


def test_previous_result_is_restored_from_complete_chunk_on_resume(tmp_path):
    grid = pd.MultiIndex.from_tuples([(1,), (2,)], names=["frequency"])

    def first_measurement(params, previous_result):
        if params["frequency"] == 2:
            raise CriticalMeasurementError("intentional stop")
        return {
            "meas": pd.DataFrame(
                {"value": [1.0]}, index=pd.Index([0], name="sample")
            )
        }

    first = SweepManager(
        measurement_func=first_measurement,
        ivars=grid,
        meas_name="previous_resume",
        output_root=tmp_path,
        resume=False,
        chunk_size=1,
        provide_previous_result=True,
    )
    first.run()

    received: list[object] = []

    def resumed_measurement(params, previous_result):
        received.append(previous_result)
        return {
            "meas": pd.DataFrame(
                {"value": [2.0]}, index=pd.Index([0], name="sample")
            )
        }

    resumed = SweepManager(
        measurement_func=resumed_measurement,
        ivars=grid,
        meas_name="previous_resume",
        output_root=tmp_path,
        resume=True,
        chunk_size=1,
        provide_previous_result=True,
    )
    resumed.run()

    assert len(received) == 1
    previous = received[0]
    assert isinstance(previous, dict)
    assert previous["meas"].index.get_level_values("frequency").tolist() == [1]
    assert previous["meas"]["value"].tolist() == [1.0]


def test_previous_list_result_preserves_none_positions_on_resume(tmp_path):
    grid = pd.MultiIndex.from_tuples([(1,), (2,)], names=["frequency"])

    def first_measurement(params, previous_result):
        if params["frequency"] == 2:
            raise CriticalMeasurementError("intentional stop")
        return [
            pd.DataFrame(
                {"value": [1.0]}, index=pd.Index([0], name="sample")
            ),
            None,
        ]

    first = SweepManager(
        measurement_func=first_measurement,
        ivars=grid,
        meas_name="previous_list_resume",
        output_root=tmp_path,
        resume=False,
        chunk_size=1,
        provide_previous_result=True,
    )
    first.run()

    received: list[object] = []

    def resumed_measurement(params, previous_result):
        received.append(previous_result)
        return [
            pd.DataFrame(
                {"value": [2.0]}, index=pd.Index([0], name="sample")
            ),
            None,
        ]

    resumed = SweepManager(
        measurement_func=resumed_measurement,
        ivars=grid,
        meas_name="previous_list_resume",
        output_root=tmp_path,
        resume=True,
        chunk_size=1,
        provide_previous_result=True,
    )
    resumed.run()

    assert len(received) == 1
    previous = received[0]
    assert isinstance(previous, list)
    assert len(previous) == 2
    assert previous[1] is None
    assert previous[0].index.get_level_values("frequency").tolist() == [1]


def test_failed_point_does_not_replace_previous_valid_result(tmp_path):
    received: dict[int, object] = {}

    def measurement(params, previous_result):
        frequency = params["frequency"]
        received[frequency] = previous_result
        if frequency == 2:
            raise RuntimeError("point-local failure")
        return {
            "meas": pd.DataFrame(
                {"value": [float(frequency)]},
                index=pd.Index([0], name="sample"),
            )
        }

    manager = SweepManager(
        measurement_func=measurement,
        ivars=pd.MultiIndex.from_tuples([(1,), (2,), (3,)], names=["frequency"]),
        meas_name="previous_gap",
        output_root=tmp_path,
        resume=False,
        chunk_size=10,
        provide_previous_result=True,
    )
    manager.run()

    previous = received[3]
    assert isinstance(previous, dict)
    assert previous["meas"].index.get_level_values("frequency").tolist() == [1]


def test_missing_previous_chunk_aborts_before_measurement_callback(tmp_path):
    grid = pd.MultiIndex.from_tuples([(1,), (2,)], names=["frequency"])

    def first_measurement(params, previous_result):
        if params["frequency"] == 2:
            raise CriticalMeasurementError("intentional stop")
        return {
            "meas": pd.DataFrame(
                {"value": [1.0]}, index=pd.Index([0], name="sample")
            )
        }

    first = SweepManager(
        measurement_func=first_measurement,
        ivars=grid,
        meas_name="missing_previous_chunk",
        output_root=tmp_path,
        resume=False,
        chunk_size=1,
        provide_previous_result=True,
    )
    first.run()
    next(first.output_dir.glob("chunk_*.h5")).unlink()

    calls: list[int] = []

    # Persistence reconciliation is deliberately part of construction and
    # therefore fails before a manager capable of issuing callbacks exists.
    # A manifest entry without its committed chunk must never be treated as a
    # point that can silently be remeasured.
    with pytest.raises(RuntimeError, match="missing chunk"):
        SweepManager(
            measurement_func=lambda params, previous: calls.append(
                params["frequency"]
            ),
            ivars=grid,
            meas_name="missing_previous_chunk",
            output_root=tmp_path,
            resume=True,
            provide_previous_result=True,
        )

    assert calls == []


def test_resume_progress_starts_at_completed_count(tmp_path, monkeypatch):
    grid = pd.MultiIndex.from_tuples([(1,), (2,)], names=["frequency"])

    def first_measurement(params):
        if params["frequency"] == 2:
            raise CriticalMeasurementError("intentional stop")
        return {
            "meas": pd.DataFrame(
                {"value": [1.0]}, index=pd.Index([0], name="sample")
            )
        }

    first = SweepManager(
        measurement_func=first_measurement,
        ivars=grid,
        meas_name="resume_progress",
        output_root=tmp_path,
        resume=False,
        chunk_size=1,
    )
    first.run()

    class FakeProgress:
        instances: list["FakeProgress"] = []

        def __init__(self, *, total, initial=0, **kwargs):
            self.total = total
            self.initial = initial
            self.updates = 0
            self.instances.append(self)

        def update(self, count):
            self.updates += count

        def set_description(self, description):
            pass

        def write(self, message):
            pass

        def close(self):
            pass

    monkeypatch.setattr(sweep_manager_module, "tqdm", FakeProgress)

    resumed = SweepManager(
        measurement_func=lambda params: {
            "meas": pd.DataFrame(
                {"value": [2.0]}, index=pd.Index([0], name="sample")
            )
        },
        ivars=grid,
        meas_name="resume_progress",
        output_root=tmp_path,
        resume=True,
        chunk_size=1,
    )
    resumed.run()

    progress = FakeProgress.instances[-1]
    assert progress.total == 2
    assert progress.initial == 1
    assert progress.updates == 1
