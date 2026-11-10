#!/usr/bin/env python3
"""Run representative Jupiter software tests sequentially through the live MCP pool.

Does not start/stop allocations, modify campaign state, or declare imports a pass.
Run with software/globus-compute/.venv/bin/python. Case logs contain public inputs
and scientific diagnostics, never environment dumps or service credentials.
"""

import argparse
import asyncio
import json
import shlex
import time
import urllib.request
from pathlib import Path
from fastmcp import Client

ENVIRONMENTS = {
    "fetch-literature": "fetch-literature",
    "rag-build-index": "rag-build-index",
    "rag-query": "rag-query",
    "openmm": "openmm",
    "molecular-simulations": "molecular-simulations",
    "proteinmpnn": "proteinmpnn",
    "rfd4-proteina": "RFD4-Proteina",
    "chai": "chai-lab",
    "boltz": "boltz",
    "protenix": "protenix",
    "uma": "fairchem",
    "esmfold2": "esmfold2",
    "genslm-esm": "genslm-esm",
    "pyrosetta": "pyrosetta",
    "reactra": "reactra",
}


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000/mcp")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", nargs="*")
    p.add_argument("--timeout", type=int, default=1800)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sw = repo.parent / "software"
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": time.time(),
        "execution": "sequential real workloads through campaign MCP pool; one active task per invocation",
        "cases": {},
    }
    # Reactra is intentionally out of scope for the Jupiter campaign audit.
    for name in a.only or [name for name in ENVIRONMENTS if name != "reactra"]:
        out = a.output / name
        out.mkdir(exist_ok=True)
        if (out / "result.json").exists():
            raise FileExistsError(f"Use a fresh audit output directory: {out}")
        py = sw / ENVIRONMENTS[name] / ".venv/bin/python"
        report["cases"][name] = {"status": "running", "started_at": time.time()}
        (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print("START", name, flush=True)
        args = [
            str(py),
            str(repo / "ports/jupiter/smoke/case.py"),
            "--case",
            name,
            "--out",
            str(out),
            "--software",
            str(sw),
        ]
        command = (
            "source "
            + shlex.quote(str(repo / "scripts/jupiter_env.sh"))
            + "; export CLUSTER=local WANDB_MODE=disabled OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1; "
            + shlex.join(args)
        )
        payload = "import subprocess,json,socket,os\nfrom pathlib import Path\nout=Path(" + repr(str(out)) + ")\n"
        payload += (
            'import signal\nwith (out/"process.log").open("w") as f:\n p=subprocess.Popen(["bash","-lc",'
            + repr(command)
            + "],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)\n try:rc=p.wait(timeout="
            + str(a.timeout)
            + ')\n except subprocess.TimeoutExpired:\n  os.killpg(p.pid,signal.SIGTERM)\n  try:p.wait(timeout=10)\n  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()\n  rc=-1\nprint(json.dumps({"returncode":rc,"host":socket.gethostname(),"job":os.environ.get("SLURM_JOB_ID"),"log":str(out/"process.log")}))\n'
        )
        try:
            async with Client(a.url, timeout=a.timeout + 180) as c:
                x = await c.call_tool("run_code", {"payload": payload, "workdir": str(out), "timeout": a.timeout + 60})
                data = x.data if isinstance(x.data, dict) else x.structured_content
            (out / "mcp-result.json").write_text(json.dumps(data, indent=2, default=str) + "\n")
            meta = json.loads(data["stdout"])
            result = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
            status = (
                "passed"
                if str(data.get("returncode")) == "0" and meta["returncode"] == 0 and result.get("status") == "passed"
                else "failed"
            )
            report["cases"][name].update(status=status, remote=meta, result=result)
            if status == "passed" and name == "rag-query":
                retrieval = json.loads((out / "retrieval.json").read_text())
                prompt = (
                    "Use only these synthetic smoke-test passages: "
                    + json.dumps(retrieval["passages"])
                    + "\nWhat target is identified? Reply with the target name."
                )
                req = urllib.request.Request(
                    "http://localhost:50856/v1/chat/completions",
                    data=json.dumps(
                        {"model": "argo:gpt-5.6-luna", "messages": [{"role": "user", "content": prompt}]}
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=120) as response:
                    answer = json.load(response)["choices"][0]["message"]["content"]
                assert "EGFR" in answer
                report["cases"][name]["llm_answer"] = answer
                report["cases"][name]["execution_note"] = (
                    "MCP retrieval followed by login-node Argo completion, matching Loam orchestration"
                )
        except Exception as exc:
            report["cases"][name].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        report["cases"][name]["finished_at"] = time.time()
        (a.output / "summary.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print("END", name, report["cases"][name]["status"], flush=True)
    report["finished_at"] = time.time()
    (a.output / "summary.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    lines = ["# Jupiter manual workflow audit", "", report["execution"], "", "| Case | Status |", "|---|---|"]
    lines.extend(f"| {name} | {case['status']} |" for name, case in report["cases"].items())
    lines.extend(["", "See each case's process.log, command.log, mcp-result.json and result.json for evidence."])
    (a.output / "summary.md").write_text("\n".join(lines) + "\n")
    if any(case["status"] != "passed" for case in report["cases"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
