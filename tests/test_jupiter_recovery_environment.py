"""Recovery must retain the campaign's env file and two-hour test allocation."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sourcing_jupiter_defaults_preserves_campaign_recovery_settings(tmp_path):
    software = tmp_path / "software"
    binary = software / "globus-compute/.venv/bin/python"
    binary.parent.mkdir(parents=True)
    binary.symlink_to(sys.executable)
    profile = tmp_path / "campaign/hpc_env.sh"
    env = dict(
        os.environ,
        LOAM_SOFTWARE_ROOT=str(software),
        HPC_ENV_FILE=str(profile),
        HPC_GLOBUS_ENDPOINT_ID="fixture-endpoint",
        HPC_WALLTIME="02:00:00",
    )
    env.pop("HPC_MAX_WALLTIME", None)
    probe = (
        'import os,json; print(json.dumps({"env_file":os.environ["HPC_ENV_FILE"],'
        '"cap":os.environ["HPC_MAX_WALLTIME"],'
        '"allocation":json.loads(os.environ["HPC_ENDPOINT_CONFIG_JSON"])["walltime"]}))'
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; "$HPC_VENV/bin/python" -c "$2"',
            "jupiter-env-test",
            str(ROOT / "skills/hpc-deployment/set_env_jupiter.sh"),
            probe,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    values = json.loads(result.stdout)
    assert values == {"env_file": str(profile), "cap": "02:00:00", "allocation": "02:00:00"}
