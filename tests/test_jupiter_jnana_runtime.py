"""Offline checks against the installed Jnana implementation, without LLM calls."""

import importlib
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

JNANA = Path(__file__).resolve().parents[2] / "software/jnana-engine"


@pytest.fixture
def runtime(monkeypatch):
    if not (JNANA / "jnana/protognosis/core/coscientist.py").is_file():
        pytest.skip("Jupiter Jnana source is not installed")
    monkeypatch.syspath_prepend(str(JNANA))
    return importlib.import_module("jnana.protognosis.core.coscientist")


def test_actual_generation_scheduler_batches_the_exact_requested_count(runtime):
    scientist = runtime.CoScientist.__new__(runtime.CoScientist)
    scientist.logger = logging.getLogger("fixture")
    scientist.memory = SimpleNamespace(metadata={"research_goal": "synthetic fixture"})
    scientist.research_goal = "synthetic fixture"
    tasks = []
    scientist.supervisor = SimpleNamespace(add_task=tasks.append)
    scientist.generate_hypotheses(count=5, batch_size=2)
    assert len(tasks) == 3
    assert [task.params["batch_size"] for task in tasks] == [2, 2, 1]
    assert all(task.task_type == "generate_hypothesis" for task in tasks)


def test_actual_generation_rejects_zero_batch_before_scheduling(runtime):
    scientist = runtime.CoScientist.__new__(runtime.CoScientist)
    scientist.logger = logging.getLogger("fixture")
    scientist.memory = SimpleNamespace(metadata={"research_goal": "synthetic fixture"})
    tasks = []
    scientist.supervisor = SimpleNamespace(add_task=tasks.append)
    with pytest.raises(ValueError, match="batch_size"):
        scientist.generate_hypotheses(count=2, batch_size=0)
    assert tasks == []


def test_a_stalled_tournament_respects_the_deadline_used_by_loam(runtime, monkeypatch, tmp_path):
    periodic = importlib.import_module("jnana.protognosis.periodic_tournament")
    config = importlib.import_module("jnana.protognosis.core.multi_llm_config")

    class NoMatches:
        def __init__(self, **kwargs):
            self.memory = SimpleNamespace(tournament_state={"matches": []}, save=lambda: None)

        def start(self):
            pass

        def stop(self):
            pass

        def get_all_hypotheses(self):
            return [{"hypothesis_id": "a"}, {"hypothesis_id": "b"}]

        def run_tournament(self, **kwargs):
            pass

        def get_statistics(self):
            return {}

        def get_top_hypotheses(self, count):
            return []

    monkeypatch.setattr(periodic, "CoScientist", NoMatches)
    result = periodic.run_incremental_tournament(
        state_file=tmp_path / "state.json",
        llm_config=config.LLMConfig(provider="openai"),
        match_count=1,
        max_seconds=0.001,
    )
    assert result["status"] == "incomplete"
    assert result["stop_reason"] == "deadline_reached"
    assert result["matches_completed"] == 0
