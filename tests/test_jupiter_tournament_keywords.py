"""User tournament options must reach the hook unchanged after the rebase."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / ".loam/adapters/hermes/plugin/loam_config.py"
SPEC = importlib.util.spec_from_file_location("jupiter_tournament_config", PATH)
config = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = config
SPEC.loader.exec_module(config)


def write_profile(tmp_path, **options):
    path = tmp_path / "loam.yaml"
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "active_profile": "research",
                "profiles": {
                    "research": {
                        "workflow_id": "research-v1",
                        "state_dir": str(tmp_path / "state"),
                        "hypothesis_tournament": {
                            "enabled": True,
                            "jnana_root": "/software/jnana-engine",
                            "state_file": str(tmp_path / "jnana.json"),
                            "reports_dir": str(tmp_path / "reports"),
                            "inference_endpoint": "http://example.test/v1",
                            "model": "test-model",
                            **options,
                        },
                    }
                },
            }
        )
    )
    return path


def test_swiss_parallel_matches_and_generation_batch_size_reach_hook(tmp_path, monkeypatch):
    path = write_profile(tmp_path, scheme="swiss", rounds=3, workers=2, batch_size=4)
    monkeypatch.setenv("LOAM_CONFIG_PATH", str(path))
    tournament = config.payload_values(config.load_profile())["loam_hypothesis_tournament"]
    assert {key: tournament[key] for key in ("scheme", "rounds", "workers", "batch_size")} == {
        "scheme": "swiss",
        "rounds": 3,
        "workers": 2,
        "batch_size": 4,
    }


def test_existing_profiles_keep_random_single_worker_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("LOAM_CONFIG_PATH", str(write_profile(tmp_path)))
    tournament = config.payload_values(config.load_profile())["loam_hypothesis_tournament"]
    assert {key: tournament[key] for key in ("scheme", "rounds", "workers", "batch_size")} == {
        "scheme": "random",
        "rounds": 1,
        "workers": 1,
        "batch_size": 1,
    }


def test_unknown_pairing_scheme_fails_before_runtime_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("LOAM_CONFIG_PATH", str(write_profile(tmp_path, scheme="swis")))
    with pytest.raises(ValueError, match="scheme"):
        config.load_profile()
