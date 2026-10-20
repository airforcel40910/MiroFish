import pytest

from app.services import zep_entity_reader as reader_module
from app.services.simulation_config_generator import (
    AgentActivityConfig,
    EventConfig,
    SimulationConfigGenerator,
)
from app.services.zep_entity_reader import ZepEntityReader, is_placeholder_entity_name


def _agent(agent_id, name, entity_type="Person"):
    return AgentActivityConfig(
        agent_id=agent_id,
        entity_uuid=f"uuid-{agent_id}",
        entity_name=name,
        entity_type=entity_type,
    )


def _assign(posts, agents):
    generator = object.__new__(SimulationConfigGenerator)
    config = EventConfig(initial_posts=posts)
    return generator._assign_initial_post_agents(config, agents).initial_posts


def test_initial_post_goes_to_the_named_persona_not_the_next_of_its_type():
    agents = [_agent(0, "Ben Okafor"), _agent(1, "Chris Doyle"), _agent(2, "Calibre", "Company")]
    posts = _assign(
        [{"content": "My business partner had a heart attack at 45.",
          "poster_name": "Chris Doyle", "poster_type": "Person"}],
        agents,
    )
    assert posts[0]["poster_agent_id"] == 1
    assert posts[0]["poster_name"] == "Chris Doyle"


def test_poster_name_match_ignores_case_and_spacing():
    agents = [_agent(0, "Ben Okafor"), _agent(1, "Mihai Popescu")]
    posts = _assign([{"content": "x", "poster_name": "  mihai   POPESCU ", "poster_type": "Person"}], agents)
    assert posts[0]["poster_agent_id"] == 1


def test_unknown_poster_name_falls_back_to_type_assignment():
    agents = [_agent(0, "Ben Okafor"), _agent(1, "Calibre", "Company")]
    posts = _assign([{"content": "x", "poster_name": "Nobody Known", "poster_type": "Company"}], agents)
    assert posts[0]["poster_agent_id"] == 1


@pytest.mark.parametrize("name", [
    "unidentified person", "Unidentified Person", "an unnamed man", "someone",
    "the individual", "Unknown", "", "  ",
])
def test_placeholder_names_are_detected(name):
    assert is_placeholder_entity_name(name)


@pytest.mark.parametrize("name", ["Mihai Popescu", "Calibre", "NHS GP", "accredited UK laboratory"])
def test_real_names_are_not_placeholders(name):
    assert not is_placeholder_entity_name(name)


def test_filter_skips_placeholder_and_duplicate_name_nodes(monkeypatch):
    nodes = [
        {"uuid": "a", "name": "Mihai Popescu", "labels": ["Entity", "Person"], "summary": "", "attributes": {}},
        {"uuid": "b", "name": "unidentified person", "labels": ["Entity", "Person"],
         "summary": "Mihai Popescu is a 38-year-old founder", "attributes": {}},
        {"uuid": "c", "name": "mihai popescu", "labels": ["Entity", "Person"], "summary": "", "attributes": {}},
        {"uuid": "d", "name": "Calibre", "labels": ["Entity", "Company"], "summary": "", "attributes": {}},
    ]
    reader = object.__new__(ZepEntityReader)
    monkeypatch.setattr(reader, "get_all_nodes", lambda graph_id: nodes, raising=False)
    monkeypatch.setattr(reader, "get_all_edges", lambda graph_id: [], raising=False)

    result = reader.filter_defined_entities("graph-1", enrich_with_edges=False)

    assert [e.uuid for e in result.entities] == ["a", "d"]
    assert result.filtered_count == 2
