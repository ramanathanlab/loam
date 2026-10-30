"""Submit payloads to the running hpc-deployment MCP server."""
from __future__ import annotations

import asyncio
import json

from fastmcp import Client

URL = "http://127.0.0.1:8000/mcp"

# The loam venv on flare is built with include-system-site-packages=false, so
# torch is not importable from it. The Aurora frameworks install is the one
# that carries torch; _run_subprocess appends bin/python to venv_path.
FRAMEWORKS = "/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0"

CHAI_VENV = "/tmp/ht1410/copper/flare/FRAME-IDP/agentic_campaign_09_26/binder_design/envs/chai-2026-venv"

# Simple end-to-end check: which node, which tile, which interpreter.
ENV_PAYLOAD = """
import json, os, socket, sys

print(json.dumps({
    "host": socket.gethostname(),
    "pid": os.getpid(),
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "ZE_AFFINITY_MASK": os.environ.get("ZE_AFFINITY_MASK"),
    "ZE_FLAT_DEVICE_HIERARCHY": os.environ.get("ZE_FLAT_DEVICE_HIERARCHY"),
    "cwd": os.getcwd(),
}, indent=2))
"""

# Forward pass of a small MLP on the agent's pinned tile.
MLP_PAYLOAD = """
import json, os, socket

import torch
from torch import nn

device = "xpu" if torch.xpu.is_available() else "cpu"
torch.manual_seed(0)

mlp = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
).to(device)

x = torch.randn(32, 128, device=device)
with torch.no_grad():
    y = mlp(x)

print(json.dumps({
    "host": socket.gethostname(),
    "ZE_AFFINITY_MASK": os.environ.get("ZE_AFFINITY_MASK"),
    "torch": torch.__version__,
    "device": device,
    "xpu_available": torch.xpu.is_available(),
    "xpu_count": torch.xpu.device_count(),
    "xpu_name": torch.xpu.get_device_name(0) if torch.xpu.is_available() else None,
    "output_shape": list(y.shape),
    "output_mean": float(y.mean()),
    "finite": bool(torch.isfinite(y).all()),
}, indent=2))
"""

INFERENCE_PAYLOAD="""
import logging
import shutil
from pathlib import Path

import numpy as np

from chai_lab.chai1 import run_inference

logging.basicConfig(level=logging.INFO)  # control verbosity

# We use fasta-like format for inputs.
# - each entity encodes protein, ligand, RNA or DNA
# - each entity is labeled with unique name;
# - ligands are encoded with SMILES; modified residues encoded like AAA(SEP)AAA

# Example given below, just modify it


example_fasta = \"\"\">protein|name=example-of-long-protein
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
>protein|name=example-of-short-protein
AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
>protein|name=example-peptide
GAAL
>ligand|name=example-ligand-as-smiles
CCCCCCCCCCCCCC(=O)O\"\"\".strip()

fasta_path = Path("/tmp/example.fasta")
fasta_path.write_text(example_fasta)

# Inference expects an empty directory; enforce this
output_dir = Path("/tmp/outputs")
if output_dir.exists():
    logging.warning(f"Removing old output directory: {output_dir}")
    shutil.rmtree(output_dir)
output_dir.mkdir(exist_ok=True)

candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    # 'default' setup
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
    device="xpu:0",
    use_esm_embeddings=True,
)

cif_paths = candidates.cif_paths
agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]


# Load pTM, ipTM, pLDDTs and clash scores for sample 2
scores = np.load(output_dir.joinpath("scores.model_idx_2.npz"))
"""


async def call(client: Client, label: str, payload: str, ntasks: int = 1, **kwargs) -> None:
    print(f"\n===== {label} =====")
    results = await asyncio.gather(*[client.call_tool(
        "run_code", {"payload": payload, "timeout": 600.0, **kwargs}
    ) for _ in range(ntasks)])
    for result in results:
        data = json.loads(result.content[0].text)
        print("RETURNCODE", data.get("returncode"))
        print("STDOUT\n" + data.get("stdout", ""))
        stderr = data.get("stderr", "")
        if stderr.strip():
            print("STDERR\n" + stderr[-3000:])


async def main() -> None:
    async with Client(URL, timeout=1800) as client:
        print("TOOLS", {t.name for t in await client.list_tools()})
        # await call(client, "env probe (agent venv)", ENV_PAYLOAD)
        # await call(client, "mlp forward (frameworks torch)", MLP_PAYLOAD,
        #            venv_path=FRAMEWORKS)
        await call(client, "chai inference", INFERENCE_PAYLOAD, ntasks=24,
                           venv_path=CHAI_VENV)


if __name__ == "__main__":
    asyncio.run(main())
