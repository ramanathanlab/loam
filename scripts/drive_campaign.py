#!/usr/bin/env python3
"""Advance a campaign round by round without a person typing `continue`.

    ./drive-campaign --max-rounds 4 --max-hours 8

What this is, and is not
------------------------

A *scheduler for turns*, nothing more. Every decision about the science belongs
to the agents: the tournament selects, validators measure, independent reviewers
accept or refuse claims, the reasoner synthesizes. This loop decides only when a
turn is worth taking, and when to stop.

Why it drives a real terminal
-----------------------------

The obvious shape -- a loop of `hermes -z "continue"` -- does not work, and the
reason is structural rather than cosmetic. Measured 2026-11-01:

1. `loam_dispatch_round` fails outright under `-z` with *"delegate_task requires
   a parent agent context."* The plugin dispatches through `ctx.dispatch_tool`,
   which injects `parent_agent` from the plugin manager's `_cli_ref` -- and
   `_cli_ref` is set only on the interactive CLI path (`cli.py`), never by
   `hermes_cli/oneshot.py`.
2. Even with that fixed, a delegation from the TOP-LEVEL model is forced into
   the background (`run_agent._dispatch_delegate_task` passes
   `background=(not _is_subagent)`, and the registry path agrees). Children
   return handles immediately and their results re-enter the conversation later
   as messages. A one-shot turn would therefore exit while its validators were
   still running, and take them with it.

So an unattended campaign needs a session that stays alive across rounds, which
means a pseudo-terminal. That also makes run-identity pinning unnecessary: there
is exactly one launch, so one `LOAM_RUN_ID`, which this reads back from the run
manifest the hooks write.

Why it polls the phase instead of sleeping on a timer
----------------------------------------------------

Every coordinator turn re-runs the Jnana tournament before the orchestration
hook even looks at the phase (`pre-llm.py` calls `Tournament(payload).context()`
before `orchestration_context_for`). At the measured ~16-21 s/match that is
minutes of ranking per turn, so a timer-driven loop would spend most of an
allocation paying to be told the validators are still running.

It also waits for the session to go quiet before sending anything. Background
delegation means the agent often takes its own next turn when a child's result
arrives; a nudge sent into a live turn is at best wasted and at worst a second
dispatch.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import select
import shutil
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Phases where a turn does work, and phases where it would only pay for a
# tournament to be told to wait.
ACTIONABLE = ("idle", "validators_due", "reviewers_due", "reasoner_due")
WAITING = ("validators_running", "reviewers_running", "reasoner_running")
# Phases whose turn will dispatch work onto HPC, so the pool must be usable
# first. A reasoner or reviewer turn needs no GPU.
NEEDS_POOL = ("idle", "validators_due")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Session:
    """One campaign launch, running in a pseudo-terminal we can type into."""

    def __init__(self, launcher: Path, log_path: Path, env: dict[str, str]) -> None:
        self.master, slave = pty.openpty()
        # Give it a believable window. prompt_toolkit renders to whatever size
        # it finds, and an 0x0 terminal makes the transcript unreadable.
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 48, 200, 0, 0))
        self.log = log_path.open("ab")
        self.process = subprocess.Popen(
            [str(launcher)],
            stdin=slave, stdout=slave, stderr=slave,
            cwd=str(launcher.parent), env=env,
            # Its own process group, so stopping the drive stops the whole
            # campaign -- hermes, its children, and anything they spawned --
            # rather than orphaning them against the next launch.
            start_new_session=True,
        )
        os.close(slave)
        self.last_output = time.monotonic()
        self.bytes_seen = 0
        self._closed = False
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        while not self._closed:
            try:
                ready, _, _ = select.select([self.master], [], [], 0.5)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                data = os.read(self.master, 65536)
            except OSError:
                return
            if not data:
                return
            self.log.write(data)
            self.log.flush()
            self.bytes_seen += len(data)
            self.last_output = time.monotonic()

    def quiet_seconds(self) -> float:
        return time.monotonic() - self.last_output

    def alive(self) -> bool:
        return self.process.poll() is None

    def send(self, text: str) -> None:
        os.write(self.master, (text + "\r").encode("utf-8"))
        self.last_output = time.monotonic()

    def close(self) -> None:
        self._closed = True
        if self.alive():
            try:
                self.send("/exit")
                self.process.wait(timeout=30)
            except Exception:  # noqa: BLE001
                pass
        if self.alive():
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(self.process.pid), sig)
                    self.process.wait(timeout=15)
                except Exception:  # noqa: BLE001
                    continue
                break
        try:
            os.close(self.master)
        except OSError:
            pass
        self.log.close()


class Driver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.campaign = Path(args.campaign).resolve()
        self.repo = Path(os.environ.get("LOAM_REPO") or args.repo or self.campaign).resolve()
        self.state = self.campaign / "state"
        self.log_dir = self.state / "driver"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "driver.jsonl"
        self.stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.deadline = time.monotonic() + args.max_hours * 3600.0
        self.run_id = ""
        self.sends = 0
        self.pool_failures = 0
        self.unreadable = 0
        self.phase_repeats = 0
        self.last_phase = ""
        self.last_send = 0.0
        self.barren_rounds = 0
        self.session: Session | None = None
        self.previous_run_id = ""
        self._payload_cache: dict[str, Any] | None = None
        # Before the first import of either: the config loader and the hooks are
        # the campaign's own copies under LOAM_REPO, not installed packages.
        for path in (
            str(self.repo / ".loam" / "adapters" / "hermes" / "plugin"),
            str(self.repo / ".loam" / "hooks"),
        ):
            if path not in sys.path:
                sys.path.insert(0, path)

    # --- logging ------------------------------------------------------------

    def log(self, event: str, **fields: Any) -> None:
        record = {"event": event, "observed_at": _now(), "run_id": self.run_id, **fields}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"[driver] {event}: " + " ".join(f"{key}={value}" for key, value in fields.items()),
              file=sys.stderr, flush=True)

    def notify(self, message: str) -> None:
        """Push a line somewhere a person will see it, if one was configured.

        Unattended means nobody reads a negative result as it arrives. Failure
        to notify never stops the run.
        """

        command = os.environ.get("LOAM_NOTIFY_CMD", "").strip()
        if not command:
            return
        try:
            subprocess.run(f"{command} {json.dumps(message)}", shell=True, check=False, timeout=60)
        except Exception as exc:  # noqa: BLE001
            self.log("notify_failed", error=f"{type(exc).__name__}: {exc}")

    # --- campaign state -----------------------------------------------------

    def _payload(self) -> dict[str, Any]:
        """The hook payload a coordinator turn sees, minus the session id.

        Built from the same config the Hermes adapter loads, so the phase this
        reads is the phase the next turn will act on. `loam_run_id` is forced to
        the live launch's id, because `payload_values` reads it from this
        process's environment and the campaign's id was minted inside the pty.
        """

        if self._payload_cache is None:
            from loam_config import effective_profile, payload_values  # noqa: PLC0415

            profile = effective_profile()
            if profile is None:
                raise SystemExit("drive_campaign: no active Loam profile; check ~/.hermes/loam.yaml")
            if not profile.hypothesis_orchestration:
                raise SystemExit("drive_campaign: the active profile has no hypothesis orchestration to drive")
            self._payload_cache = {"repo_root": str(self.repo), **payload_values(profile)}
        return {**self._payload_cache, "loam_run_id": self.run_id, "loam_state_dir": str(self.state)}

    def manifest_run_id(self) -> str:
        """The live launch's run id, from the manifest the hooks write."""

        path = self.state / "run-manifest.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        identity = value.get("identity") if isinstance(value, dict) else None
        return str((identity or {}).get("run_id") or "")

    def phase(self) -> str:
        import hypothesis_orchestration as orchestration  # noqa: PLC0415

        payload = self._payload()
        return orchestration._round(payload, orchestration._rows(payload))["phase"]

    def _rows(self, filename: str) -> list[dict[str, Any]]:
        path = self.state / filename
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def progress(self) -> dict[str, Any]:
        """The numbers the stop conditions are about.

        `rounds` is reasoner syntheses finished by THIS run; `accepted` is the
        accepted-evidence ledger, which is the only measure of whether the
        campaign is producing findings rather than merely spending nodes.
        """

        rounds, briefs = 0, []
        for row in self._rows("hypothesis-orchestration.jsonl"):
            if row.get("event") != "reasoner_completed":
                continue
            if self.run_id and row.get("run_id") != self.run_id:
                continue
            rounds += 1
            briefs.append((row.get("summary") or "").strip())
        return {
            "rounds": rounds,
            "accepted": len(self._rows("evidence-ledger.jsonl")),
            "reviews": len(self._rows("hypothesis-reviews.jsonl")),
            "repeat_brief": len(briefs) >= 2 and briefs[-1] == briefs[-2] and bool(briefs[-1]),
        }

    # --- HPC ----------------------------------------------------------------

    def pool_ready(self) -> bool:
        """One real placement probe, and a restart if it fails.

        A listening port is NOT evidence of a usable pool: observed
        2026-10-31, the server held :8000 with no agents against an expiring
        block. The preflight's probe returns hostname, PBS job id and device pin
        from an actual `run_code`, which is the only check that tells the two
        apart -- `cluster pool ready` does not fire on a rebuild, so it cannot
        be used instead.
        """

        if self.args.no_probe:
            return True
        python = os.environ.get("LOAM_PREFLIGHT_PYTHON") or str(self.campaign / ".venv" / "bin" / "python")
        if not Path(python).is_file():
            self.log("probe_skipped", reason=f"no preflight interpreter at {python}")
            return True
        base = [
            python,
            str(self.repo / "scripts" / "campaign_preflight.py"),
            "--repo", str(self.repo),
            "--hermes-home", str(self.campaign / ".hermes" / "home"),
            "--timeout", str(self.args.probe_timeout),
        ]
        probe = subprocess.run([*base, "--no-auto-start"], capture_output=True, text=True)
        if probe.returncode == 0:
            self.pool_failures = 0
            return True
        # Failed: let the preflight start what is missing and wait for it. This
        # is the block-boundary path -- min_blocks=1 requests a replacement
        # within minutes and the pool rebuilds onto it unaided, but the server
        # may need restarting onto the new block.
        self.log("pool_unready", detail=(probe.stderr or probe.stdout).strip()[-400:])
        repair = subprocess.run([*base, "--wait-seconds", str(self.args.hpc_wait_seconds)],
                                capture_output=True, text=True)
        if repair.returncode == 0:
            self.log("pool_restored")
            self.pool_failures = 0
            return True
        self.pool_failures += 1
        self.log("pool_restore_failed", attempts=self.pool_failures,
                 detail=(repair.stderr or repair.stdout).strip()[-400:])
        return False

    # --- the loop -----------------------------------------------------------

    def launcher(self) -> Path:
        if self.args.launcher:
            return Path(self.args.launcher).resolve()
        for name in ("launch.sh", "run-campaign"):
            candidate = self.campaign / name
            if candidate.is_file():
                return candidate
        raise SystemExit(f"drive_campaign: no launcher in {self.campaign} (expected launch.sh or run-campaign)")

    def start(self) -> None:
        launcher = self.launcher()
        transcript = self.log_dir / f"session-{self.stamp}.log"
        environment = {**os.environ}
        # The launch owns its own state identity; nothing is pinned here.
        for key in ("LOAM_RUN_ID", "LOAM_STATE_DIR"):
            environment.pop(key, None)
        self.previous_run_id = self.manifest_run_id()
        self.log("launching", launcher=str(launcher), transcript=transcript.name,
                 previous_run_id=self.previous_run_id)
        self.session = Session(launcher, transcript, environment)

    def wait_for_run_id(self) -> bool:
        """Block until the hooks have written a run manifest for this launch.

        The launcher's own preflight may be waiting on a queued allocation, so
        this is generous: a `debug-scaling` block has taken 2-8 minutes to start
        and the gate waits up to 1500s for the server to listen.
        """

        deadline = time.monotonic() + self.args.startup_seconds
        while time.monotonic() < deadline:
            if self.session is None or not self.session.alive():
                return False
            run_id = self.manifest_run_id()
            # The manifest from the PREVIOUS launch is still on disk, and
            # reading it as this launch's id would scope every phase read to a
            # finished run -- so wait for an id this drive has not seen.
            if run_id and run_id != self.previous_run_id:
                self.run_id = run_id
                self.log("run_id", run_id=run_id)
                return True
            # Nothing writes a manifest until the first hook fires, which needs
            # the first turn -- so send the opening prompt once the session has
            # settled, then keep waiting for the manifest it produces.
            if not self.sends and self.session.quiet_seconds() >= self.args.idle_seconds:
                self.send(self.args.first_prompt, "startup")
            time.sleep(5)
        return bool(self.run_id)

    def send(self, text: str, phase: str) -> None:
        assert self.session is not None
        self.sends += 1
        self.last_send = time.monotonic()
        self.session.send(text)
        self.log("sent", turn=self.sends, phase=phase, text=text[:60])

    def stop_reason(self, before: dict[str, Any], after: dict[str, Any]) -> str:
        if after["rounds"] >= self.args.max_rounds:
            return f"reached max-rounds ({self.args.max_rounds})"
        if after["repeat_brief"]:
            return "the reasoner returned the same brief twice; the campaign is repeating itself"
        completed = after["rounds"] - before["rounds"]
        if completed and after["accepted"] == before["accepted"]:
            self.barren_rounds += 1
            if self.barren_rounds >= self.args.barren_rounds:
                return (
                    f"{self.barren_rounds} consecutive rounds accepted no new evidence; "
                    "a loop that produces no accepted findings only spends the allocation"
                )
        elif completed:
            self.barren_rounds = 0
        return ""

    def run(self) -> int:
        self.log("start", max_rounds=self.args.max_rounds, max_hours=self.args.max_hours,
                 max_turns=self.args.max_turns, campaign=str(self.campaign))
        self.start()
        if not self.wait_for_run_id():
            return self.finish("the launch never produced a run manifest; see the session transcript")
        progress = self.progress()
        try:
            while True:
                if self.session is None or not self.session.alive():
                    return self.finish("the campaign session exited")
                if time.monotonic() >= self.deadline:
                    return self.finish(f"reached max-hours ({self.args.max_hours})")
                if self.sends >= self.args.max_turns:
                    return self.finish(f"reached max-turns ({self.args.max_turns})")

                try:
                    phase = self.phase()
                    self.unreadable = 0
                except SystemExit:
                    raise
                except Exception as exc:  # noqa: BLE001
                    # A transient read (a half-written ledger line) is worth
                    # retrying; a broken import is not, and retrying it forever
                    # is how an unattended run looks busy while doing nothing.
                    self.unreadable += 1
                    self.log("phase_unreadable", attempts=self.unreadable, error=f"{type(exc).__name__}: {exc}")
                    if self.unreadable >= 3:
                        return self.finish(f"could not read the round phase: {type(exc).__name__}: {exc}")
                    time.sleep(self.args.poll_seconds)
                    continue

                after = self.progress()
                if after["rounds"] > progress["rounds"]:
                    self.log("round_completed", rounds=after["rounds"], accepted=after["accepted"],
                             reviews=after["reviews"])
                    self.notify(
                        f"Loam round {after['rounds']} finished: {after['accepted']} accepted evidence rows "
                        f"({after['accepted'] - progress['accepted']} new), {after['reviews']} reviews."
                    )
                    reason = self.stop_reason(progress, after)
                    progress = after
                    if reason:
                        return self.finish(reason)
                progress = after

                if phase in WAITING:
                    self.phase_repeats = 0
                    time.sleep(self.args.poll_seconds)
                    continue
                if phase not in ACTIONABLE:
                    return self.finish(f"unknown phase {phase!r}; refusing to guess what to send")

                # Background delegation means the agent often takes its own next
                # turn when a child's result lands. Only nudge a session that has
                # gone quiet, and never twice inside one turn's worth of time.
                if self.session.quiet_seconds() < self.args.idle_seconds:
                    time.sleep(self.args.poll_seconds)
                    continue
                if time.monotonic() - self.last_send < self.args.min_send_gap:
                    time.sleep(self.args.poll_seconds)
                    continue

                if phase == self.last_phase:
                    self.phase_repeats += 1
                    if self.phase_repeats >= self.args.stuck_turns:
                        return self.finish(
                            f"{self.phase_repeats} nudges left the phase at {phase!r}; "
                            "something is refusing to advance and more turns will not fix it"
                        )
                else:
                    self.phase_repeats = 0
                self.last_phase = phase

                if phase in NEEDS_POOL and not self.pool_ready():
                    if self.pool_failures >= self.args.pool_failures:
                        return self.finish(f"HPC pool unusable after {self.pool_failures} restart attempts")
                    time.sleep(self.args.poll_seconds)
                    continue

                self.send(self.args.prompt, phase)
                time.sleep(self.args.poll_seconds)
        except KeyboardInterrupt:
            return self.finish("interrupted")

    def finish(self, reason: str) -> int:
        progress = self.progress()
        self.log("stop", reason=reason, turns=self.sends, rounds=progress["rounds"],
                 accepted=progress["accepted"], reviews=progress["reviews"])
        self.notify(
            f"Loam driver stopped: {reason} (rounds={progress['rounds']}, accepted={progress['accepted']})"
        )
        if self.session is not None and not self.args.leave_running:
            self.session.close()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default=".", help="the campaign directory (default: cwd)")
    parser.add_argument("--repo", default="", help="the loam checkout; defaults to $LOAM_REPO")
    parser.add_argument("--launcher", default="", help="what to run in the pty (default: ./launch.sh, else ./run-campaign)")
    parser.add_argument("--max-rounds", type=int, default=4,
                        help="stop after this many completed reasoner syntheses (default 4)")
    parser.add_argument("--max-hours", type=float, default=8.0,
                        help="wall-clock budget for the whole drive (default 8)")
    parser.add_argument("--max-turns", type=int, default=24,
                        help="hard cap on nudges sent, whatever the phase says (default 24)")
    parser.add_argument("--barren-rounds", type=int, default=2,
                        help="stop after this many consecutive rounds that accept no new evidence (default 2)")
    parser.add_argument("--stuck-turns", type=int, default=3,
                        help="stop when this many nudges leave the phase unchanged (default 3)")
    parser.add_argument("--pool-failures", type=int, default=3,
                        help="stop after this many failed HPC restarts (default 3)")
    parser.add_argument("--poll-seconds", type=float, default=60.0,
                        help="how long to wait between phase reads (default 60)")
    parser.add_argument("--idle-seconds", type=float, default=180.0,
                        help="how long the session must be silent before a nudge (default 180)")
    parser.add_argument("--min-send-gap", type=float, default=600.0,
                        help="minimum seconds between two nudges (default 600)")
    parser.add_argument("--startup-seconds", type=float, default=2400.0,
                        help="how long to wait for the launch to write a run manifest (default 2400)")
    parser.add_argument("--probe-timeout", type=float, default=300.0,
                        help="seconds for one HPC placement probe (default 300)")
    parser.add_argument("--hpc-wait-seconds", type=int, default=1500,
                        help="seconds to wait for a restarted allocation to listen (default 1500)")
    parser.add_argument("--first-prompt", default="Begin the campaign.",
                        help="the opening turn (default 'Begin the campaign.')")
    parser.add_argument("--prompt", default="continue", help="each later turn (default 'continue')")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip the per-round pool probe. Diagnostics only: a turn dispatched onto a "
                             "dead pool loses its round's work")
    parser.add_argument("--leave-running", action="store_true",
                        help="leave the campaign session alive when the driver stops")
    args = parser.parse_args()
    if not shutil.which("hermes"):
        print("drive_campaign: hermes is not on PATH", file=sys.stderr)
    return Driver(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
