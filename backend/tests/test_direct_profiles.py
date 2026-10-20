import json

import pytest

from app import create_app
from app.config import Config
from app.services import simulation_manager as simulation_manager_module
from app.services.oasis_profile_generator import OasisAgentProfile, OasisProfileGenerator
from app.services.simulation_manager import (
    SimulationManager,
    SimulationState,
    SimulationStatus,
    build_direct_entities,
)
from app.services.zep_entity_reader import EntityNode, FilteredEntities


def _prospects(n):
    return [
        {"name": f"Prospect {i:03d}", "persona": f"Persona text {i}.", "age": 30 + i % 20,
         "gender": "male", "country": "UK", "profession": "Founder", "entity_type": "Prospect"}
        for i in range(n)
    ]


def _graph_entity(uuid, name, entity_type="Person"):
    return EntityNode(uuid=uuid, name=name, labels=["Entity", entity_type], summary=f"{name} summary", attributes={})


class _Reader:
    def __init__(self, entities):
        self.entities = entities

    def __call__(self):
        return self

    def filter_defined_entities(self, **kwargs):
        return FilteredEntities(
            entities=list(self.entities),
            entity_types={e.get_entity_type() for e in self.entities},
            total_count=len(self.entities),
            filtered_count=len(self.entities),
        )


class _ConfigGenerator:
    calls = []

    def generate_config(self, **kwargs):
        _ConfigGenerator.calls.append(kwargs)

        class Params:
            generation_reasoning = "stub"

            def to_json(self):
                return json.dumps({"agents": [e.name for e in kwargs["entities"]]})

        return Params()


def _prepare(tmp_path, monkeypatch, graph_entities, direct_profiles, direct_only):
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(simulation_manager_module, "ZepEntityReader", _Reader(graph_entities))
    monkeypatch.setattr(simulation_manager_module, "SimulationConfigGenerator", _ConfigGenerator)
    monkeypatch.setattr(Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(Config, "ZEP_API_KEY", None)

    def fake_llm_profile(self, entity, user_id, use_llm=True):
        return OasisAgentProfile(user_id=user_id, user_name=f"u{user_id}", name=entity.name,
                                 bio="generated", persona="generated")

    monkeypatch.setattr(OasisProfileGenerator, "generate_profile_from_entity", fake_llm_profile)

    manager = SimulationManager()
    state = SimulationState(simulation_id="sim_direct", project_id="p", graph_id="g")
    manager._save_simulation_state(state)
    result = manager.prepare_simulation(
        simulation_id="sim_direct",
        simulation_requirement="req",
        document_text="doc",
        direct_profiles=direct_profiles,
        direct_profiles_only=direct_only,
    )
    profiles = json.loads((tmp_path / "sim_direct" / "reddit_profiles.json").read_text(encoding="utf-8"))
    return result, profiles


def test_direct_profiles_only_yields_exactly_one_agent_per_profile(tmp_path, monkeypatch):
    prospects = _prospects(100)
    graph = [_graph_entity("g1", "Calibre", "Company"), _graph_entity("g2", "unrelated person node")]

    state, profiles = _prepare(tmp_path, monkeypatch, graph, prospects, direct_only=True)

    assert state.status == SimulationStatus.READY
    assert state.profiles_count == 100
    assert [p["name"] for p in profiles] == [p["name"] for p in prospects]
    assert [p["user_id"] for p in profiles] == list(range(100))
    # verbatim: persona and fields are copied, not regenerated
    assert profiles[7]["persona"] == "Persona text 7."
    assert profiles[7]["age"] == prospects[7]["age"]
    assert (tmp_path / "sim_direct" / "direct_profiles.json").exists()


def test_direct_profiles_replace_same_name_graph_entities_and_keep_others(tmp_path, monkeypatch):
    prospects = _prospects(3)
    graph = [
        _graph_entity("g1", "prospect 001"),  # same person, extracted from the seed: must not duplicate
        _graph_entity("g2", "NHS GP"),
    ]

    state, profiles = _prepare(tmp_path, monkeypatch, graph, prospects, direct_only=False)

    names = [p["name"] for p in profiles]
    assert names == ["Prospect 000", "Prospect 001", "Prospect 002", "NHS GP"]
    assert profiles[1]["persona"] == "Persona text 1."
    assert profiles[3]["persona"] == "generated"
    assert state.entities_count == 4


@pytest.mark.parametrize("bad, message", [
    ([{"name": "A"}], "persona"),
    ([{"persona": "x"}], "name"),
    ([{"name": "A", "persona": "x"}, {"name": " a ", "persona": "y"}], "重复"),
])
def test_build_direct_entities_rejects_invalid_profiles(bad, message):
    with pytest.raises(ValueError, match=message):
        build_direct_entities(bad)


def test_prepare_api_rejects_invalid_direct_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(tmp_path))
    manager = SimulationManager()
    manager._save_simulation_state(SimulationState(simulation_id="sim_api", project_id="p", graph_id="g"))

    class Project:
        simulation_requirement = "req"

    from app.api import simulation as simulation_api
    monkeypatch.setattr(simulation_api.ProjectManager, "get_project", staticmethod(lambda pid: Project()))
    monkeypatch.setattr(simulation_api.ProjectManager, "get_extracted_text", staticmethod(lambda pid: "doc"))

    app = create_app()
    app.config.update(TESTING=True)
    response = app.test_client().post("/api/simulation/prepare", json={
        "simulation_id": "sim_api",
        "force_regenerate": True,
        "direct_profiles": [{"name": "A", "persona": "x"}, {"name": "A", "persona": "y"}],
    })

    assert response.status_code == 400
    assert response.json["success"] is False
