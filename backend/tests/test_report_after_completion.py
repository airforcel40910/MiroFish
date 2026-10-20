from types import SimpleNamespace

from flask import Flask

from app.api import report as report_api
from app.services import simulation_runner as runner_module
from app.services.simulation_runner import (
    RunnerStatus,
    SimulationRunState,
    SimulationRunner,
)
from app.services.zep_graph_memory_updater import ZepGraphMemoryManager

SIM_ID = "sim-idle"


class IdleProcess:
    """A simulation process parked in wait-for-commands mode."""

    returncode = 1

    def __init__(self):
        self.exited = False
        self.terminated = False

    def poll(self):
        return self.returncode if self.exited else None


def _setup(monkeypatch, tmp_path):
    twitter_dir = tmp_path / SIM_ID / "twitter"
    twitter_dir.mkdir(parents=True)
    (twitter_dir / "actions.jsonl").write_text(
        '{"event_type":"simulation_end","total_rounds":1,"total_actions":0}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))

    state = SimulationRunState(
        simulation_id=SIM_ID,
        runner_status=RunnerStatus.RUNNING,
        twitter_running=True,
    )
    synced = []
    stopped = []
    updaters = {SIM_ID: object()}

    monkeypatch.setitem(SimulationRunner._run_states, SIM_ID, state)
    monkeypatch.setattr(
        SimulationRunner, "_save_run_state", classmethod(lambda _c, _s: None)
    )
    monkeypatch.setattr(
        SimulationRunner,
        "_sync_simulation_status",
        classmethod(lambda _c, _id, status, error=None: synced.append(status)),
    )
    monkeypatch.setattr(
        ZepGraphMemoryManager,
        "get_updater",
        classmethod(lambda _c, simulation_id: updaters.get(simulation_id)),
    )

    def stop_updater(_c, simulation_id):
        stopped.append(simulation_id)
        updaters.pop(simulation_id, None)

    monkeypatch.setattr(
        ZepGraphMemoryManager, "stop_updater", classmethod(stop_updater)
    )
    monkeypatch.setitem(SimulationRunner._graph_memory_enabled, SIM_ID, True)
    process = IdleProcess()
    monkeypatch.setitem(SimulationRunner._processes, SIM_ID, process)
    return state, process, synced, stopped, updaters


def test_completion_with_live_process_publishes_completed(monkeypatch, tmp_path):
    state, process, synced, stopped, updaters = _setup(monkeypatch, tmp_path)
    SimulationRunner._read_action_log(
        str(tmp_path / SIM_ID / "twitter" / "actions.jsonl"), 0, state, "twitter"
    )

    SimulationRunner._finalize_on_platform_completion(SIM_ID)

    assert state.runner_status == RunnerStatus.COMPLETED
    assert state.error is None
    assert synced[-1] == RunnerStatus.COMPLETED
    assert stopped == [SIM_ID]
    assert ZepGraphMemoryManager.get_updater(SIM_ID) is None
    assert SIM_ID not in SimulationRunner._graph_memory_enabled
    assert process.poll() is None  # interviews keep working

    # A second call is a no-op.
    SimulationRunner._finalize_on_platform_completion(SIM_ID)
    assert stopped == [SIM_ID]


def test_process_exit_after_completion_leaves_status_unchanged(
    monkeypatch, tmp_path
):
    state, process, synced, stopped, _updaters = _setup(monkeypatch, tmp_path)
    monkeypatch.setitem(
        SimulationRunner._monitor_threads, SIM_ID, SimpleNamespace()
    )
    observed = {}

    def fake_sleep(_seconds):
        observed["status"] = state.runner_status
        observed["alive"] = process.poll() is None
        observed["updater"] = ZepGraphMemoryManager.get_updater(SIM_ID)
        # Close the environment with a non-zero exit code.
        process.exited = True

    monkeypatch.setattr(runner_module.time, "sleep", fake_sleep)

    SimulationRunner._monitor_simulation(SIM_ID)

    assert observed == {
        "status": RunnerStatus.COMPLETED,
        "alive": True,
        "updater": None,
    }
    assert state.runner_status == RunnerStatus.COMPLETED
    assert state.error is None
    assert stopped == [SIM_ID]
    assert RunnerStatus.FAILED not in synced
    assert RunnerStatus.STOPPED not in synced


def test_stop_after_completion_keeps_completed(monkeypatch, tmp_path):
    state, process, _synced, stopped, _updaters = _setup(monkeypatch, tmp_path)
    SimulationRunner._read_action_log(
        str(tmp_path / SIM_ID / "twitter" / "actions.jsonl"), 0, state, "twitter"
    )
    SimulationRunner._finalize_on_platform_completion(SIM_ID)

    def terminate(_c, proc, _simulation_id, timeout=10):
        proc.exited = True

    monkeypatch.setattr(
        SimulationRunner, "_terminate_process", classmethod(terminate)
    )

    result = SimulationRunner.stop_simulation(SIM_ID)

    assert result.runner_status == RunnerStatus.COMPLETED
    assert process.poll() is not None
    assert stopped == [SIM_ID]


def test_report_endpoint_is_not_blocked_after_completion(monkeypatch, tmp_path):
    state, _process, _synced, _stopped, _updaters = _setup(monkeypatch, tmp_path)
    SimulationRunner._read_action_log(
        str(tmp_path / SIM_ID / "twitter" / "actions.jsonl"), 0, state, "twitter"
    )
    SimulationRunner._finalize_on_platform_completion(SIM_ID)

    simulation = SimpleNamespace(project_id="proj-1", graph_id="graph-1")
    monkeypatch.setattr(
        report_api,
        "SimulationManager",
        lambda: SimpleNamespace(get_simulation=lambda _id: simulation),
    )
    monkeypatch.setattr(
        report_api.ProjectManager,
        "get_project",
        classmethod(lambda _c, _project_id: None),
    )

    app = Flask(__name__)
    with app.test_request_context(
        "/api/report/generate", method="POST", json={"simulation_id": SIM_ID}
    ):
        result = report_api.generate_report()

    response, status = result if isinstance(result, tuple) else (result, result.status_code)
    # Past the run-status gate: the next failure is the (stubbed) missing project.
    assert status == 404, response.get_json()
