#!/usr/bin/env python3
"""Advance a campaign round by round without a person typing `continue`.

Run it through the launcher, which owns the campaign's environment:

    ./run-campaign --drive --max-rounds 6 --max-hours 8

What this is, and is not
------------------------

It is a *scheduler for turns*, nothing more. Every decision about the science
still belongs to the agents: the tournament selects, validators measure,
independent reviewers accept or refuse claims, the reasoner synthesizes. This
loop decides only *when a turn is worth taking*, and when to stop.

Why turns and not a timer
-------------------------

Every coordinator turn re-runs the Jnana tournament before the orchestration
hook even looks at the phase (`pre-llm.py` calls `Tournament(payload).context()`
before `orchestration_context_for`). At the measured ~16-21 s/match that is
minutes of ranking per turn, so a timer-driven loop would spend most of an
allocation being told "validators still running". This polls the phase instead
and spends a turn only when one will do something.

What a turn is here
-------------------

`hermes -z` ALWAYS starts a new session: measured 2026-11-01, neither
`-c <name>` nor `-r <id>` resumes a one-shot (both produced a fresh untitled
session, and the second turn had no memory of the first). So a driven campaign
has no conversation continuity between turns, and that is survivable only
because the campaign keeps its state on disk: the pre-LLM hook rebuilds the
coordinator's directive from the ledger every turn, and `loam_dispatch_round`
reads the plan from persisted state rather than from context.

The one thing a fresh session would break is run identity -- `run-campaign`
mints a new `LOAM_RUN_ID` per launch, and `_active_plan` is scoped by run, so a
new id per turn would orphan the round in flight. The launcher therefore pins
one id for the whole drive and this loop passes it to every turn.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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


class Driver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = Path(os.environ.get("LOAM_REPO") or ".").resolve()
        self.state = Path(os.environ["LOAM_STATE_DIR"])
        self.run_id = os.environ.get("LOAM_RUN_ID", "")
        self.log_dir = self.state / "driver"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "driver.jsonl"
        self.deadline = time.monotonic() + args.max_hours * 3600.0
        self.turns = 0
        self.pool_failures = 0
        self.phase_repeats = 0
        self.last_phase = ""
        self.unreadable = 0
        self._payload_cache: dict[str, Any] | None = None
        # Both before the first import of either: the config loader and the
        # hooks are the campaign's own copies in LOAM_REPO, not installed
        # packages, and a late path insert would make the first import fail.
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
        print(f"[driver] {event}: " + " ".join(f"{key}={value}" for key, value in fields.items()), file=sys.stderr)

    def notify(self, message: str) -> None:
        """Push a line somewhere a person will see it, if the campaign configured one.

        Unattended means nobody reads a negative result as it arrives, so the
        round summary is worth pushing. Failure to notify never stops the run.
        """

        command = os.environ.get("LOAM_NOTIFY_CMD", "").strip()
        if not command:
            return
        try:
            subprocess.run(f"{command} {json.dumps(message)}", shell=True, check=False, timeout=60)
        except Exception as exc:  # noqa: BLE001 - never let alerting end a campaign
            self.log("notify_failed", error=f"{type(exc).__name__}: {exc}")

    # --- campaign state -----------------------------------------------------

    def _payload(self) -> dict[str, Any]:
        """The hook payload a coordinator turn would see, minus the session.

        Built from the same config the Hermes adapter loads, so the phase this
        reads is the phase the next turn will act on.
        """

        if self._payload_cache is not None:
            return self._payload_cache
        from loam_config import effective_profile, payload_values  # noqa: PLC0415

        profile = effective_profile()
        if profile is None:
            raise SystemExit("drive_campaign: no active Loam profile; check ~/.hermes/loam.yaml")
        if not profile.hypothesis_orchestration:
            raise SystemExit("drive_campaign: the active profile has no hypothesis orchestration to drive")
        payload = {"repo_root": str(self.repo), **payload_values(profile)}
        if not payload.get("loam_run_id"):
            raise SystemExit("drive_campaign: LOAM_RUN_ID is unset; run through ./run-campaign --drive")
        self._payload_cache = payload
        return payload

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
        """The three numbers the stop conditions are about.

        `rounds` is reasoner syntheses finished by THIS run; `accepted` is the
        accepted-evidence ledger, which is the only measure of whether the
        campaign is producing findings rather than merely spending nodes;
        `briefs` is the latest synthesis, to catch a campaign repeating itself.
        """

        rounds, briefs = 0, []
        for row in self._rows("hypothesis-orchestration.jsonl"):
            if row.get("event") != "reasoner_completed":
                continue
            if self.run_id and row.get("run_id") not in ("", self.run_id):
                continue
            rounds += 1
            briefs.append((row.get("summary") or "").strip())
        return {
            "rounds": rounds,
            "accepted": len(self._rows("evidence-ledger.jsonl")),
            "reviews": len(self._rows("hypothesis-reviews.jsonl")),
            "brief": briefs[-1] if briefs else "",
            "repeat_brief": len(briefs) >= 2 and briefs[-1] == briefs[-2] and bool(briefs[-1]),
        }

    # --- HPC ----------------------------------------------------------------

    def pool_ready(self) -> bool:
        """One real placement probe, and a restart if it fails.

        A listening port is NOT evidence of a usable pool: observed
        2026-10-31, the server held :8000 with no agents against an expiring
        block. The preflight's probe returns hostname, PBS job id and device
        pin from an actual `run_code`, which is the only check that distinguishes
        the two -- and `cluster pool ready` does not fire on a rebuild, so it
        cannot be used instead.
        """

        if self.args.no_preflight:
            return True
        python = os.environ.get("LOAM_PREFLIGHT_PYTHON") or str(Path.home() / ".loam-venv-312" / "bin" / "python")
        base = [
            python,
            str(self.repo / "scripts" / "campaign_preflight.py"),
            "--repo", str(self.repo),
            "--hermes-home", os.environ.get("HERMES_HOME", ""),
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
        repair = subprocess.run(
            [*base, "--wait-seconds", str(self.args.hpc_wait_seconds)], capture_output=True, text=True
        )
        if repair.returncode == 0:
            self.log("pool_restored")
            self.pool_failures = 0
            return True
        self.pool_failures += 1
        self.log("pool_restore_failed", attempts=self.pool_failures,
                 detail=(repair.stderr or repair.stdout).strip()[-400:])
        return False

    # --- turns --------------------------------------------------------------

    def turn(self, phase: str) -> int:
        """Send one `continue` and keep its whole transcript on disk."""

        self.turns += 1
        path = self.log_dir / f"turn-{self.turns:03d}-{phase}.log"
        if self.args.dry_run:
            self.log("turn_skipped", turn=self.turns, phase=phase, reason="dry run")
            return 0
        environment = {**os.environ, "LOAM_RUN_ID": self.run_id}
        started = time.monotonic()
        with path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [self.args.hermes, "-z", self.args.prompt],
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        self.log("turn", turn=self.turns, phase=phase, exit=completed.returncode,
                 seconds=round(time.monotonic() - started, 1), log=path.name)
        return completed.returncode

    # --- the loop -----------------------------------------------------------

    def stop_reason(self, before: dict[str, Any], after: dict[str, Any]) -> str:
        if after["rounds"] >= self.args.max_rounds:
            return f"reached max-rounds ({self.args.max_rounds})"
        if time.monotonic() >= self.deadline:
            return f"reached max-hours ({self.args.max_hours})"
        if after["repeat_brief"]:
            return "the reasoner returned the same brief twice; the campaign is repeating itself"
        completed = after["rounds"] - before["rounds"]
        if completed and after["accepted"] == before["accepted"]:
            self.barren_rounds = getattr(self, "barren_rounds", 0) + 1
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
                 dry_run=self.args.dry_run, state=str(self.state))
        while True:
            if time.monotonic() >= self.deadline:
                return self.finish(f"reached max-hours ({self.args.max_hours})")
            try:
                phase = self.phase()
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001
                # A transient read (a half-written ledger line) is worth
                # retrying; a broken import or a missing state dir is not, and
                # retrying it forever is how an unattended run looks busy while
                # doing nothing.
                self.unreadable += 1
                self.log("phase_unreadable", attempts=self.unreadable, error=f"{type(exc).__name__}: {exc}")
                if self.unreadable >= 3:
                    return self.finish(f"could not read the round phase: {type(exc).__name__}: {exc}")
                time.sleep(self.args.poll_seconds)
                continue
            self.unreadable = 0

            self.phase_repeats = self.phase_repeats + 1 if phase == self.last_phase else 0
            self.last_phase = phase

            if phase in WAITING:
                # Children are in flight inside a turn that has not returned, or
                # a turn ended with work still recorded as running. Either way a
                # new turn would be refused by tool_policy, so wait.
                if self.phase_repeats and self.phase_repeats % 20 == 0:
                    self.log("waiting", phase=phase, polls=self.phase_repeats)
                time.sleep(self.args.poll_seconds)
                continue

            if phase not in ACTIONABLE:
                return self.finish(f"unknown phase {phase!r}; refusing to guess what to send")

            if self.phase_repeats >= self.args.stuck_turns:
                return self.finish(
                    f"{self.phase_repeats} turns left the phase at {phase!r}; "
                    "something is refusing to advance and more turns will not fix it"
                )

            if phase in NEEDS_POOL and not self.pool_ready():
                if self.pool_failures >= self.args.pool_failures:
                    return self.finish(f"HPC pool unusable after {self.pool_failures} restart attempts")
                time.sleep(self.args.poll_seconds)
                continue

            before = self.progress()
            code = self.turn(phase)
            after = self.progress()
            if code != 0:
                self.log("turn_failed", exit=code)
                if self.args.stop_on_turn_error:
                    return self.finish(f"a turn exited {code}")
            if after["rounds"] > before["rounds"]:
                message = (
                    f"Loam round {after['rounds']} finished: {after['accepted']} accepted evidence rows "
                    f"({after['accepted'] - before['accepted']} new), {after['reviews']} reviews."
                )
                self.log("round_completed", rounds=after["rounds"], accepted=after["accepted"])
                self.notify(message)
            reason = self.stop_reason(before, after)
            if reason:
                return self.finish(reason)

    def finish(self, reason: str) -> int:
        progress = self.progress()
        self.log("stop", reason=reason, turns=self.turns, rounds=progress["rounds"],
                 accepted=progress["accepted"])
        self.notify(f"Loam driver stopped: {reason} (rounds={progress['rounds']}, accepted={progress['accepted']})")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-rounds", type=int, default=4,
                        help="stop after this many completed reasoner syntheses (default 4)")
    parser.add_argument("--max-hours", type=float, default=8.0,
                        help="wall-clock budget for the whole drive (default 8)")
    parser.add_argument("--barren-rounds", type=int, default=2,
                        help="stop after this many consecutive rounds that accept no new evidence (default 2)")
    parser.add_argument("--stuck-turns", type=int, default=3,
                        help="stop when this many turns leave the phase unchanged (default 3)")
    parser.add_argument("--pool-failures", type=int, default=3,
                        help="stop after this many failed HPC restarts (default 3)")
    parser.add_argument("--poll-seconds", type=float, default=60.0,
                        help="how long to wait between phase reads while children run (default 60)")
    parser.add_argument("--probe-timeout", type=float, default=300.0,
                        help="seconds for one HPC placement probe (default 300)")
    parser.add_argument("--hpc-wait-seconds", type=int, default=900,
                        help="seconds to wait for a restarted allocation to listen (default 900)")
    parser.add_argument("--prompt", default="continue",
                        help="the turn text (default 'continue')")
    parser.add_argument("--hermes", default=shutil.which("hermes") or "hermes")
    parser.add_argument("--no-preflight", action="store_true",
                        help="skip the per-round pool probe. Diagnostics only: a turn dispatched onto a "
                             "dead pool loses its round's work")
    parser.add_argument("--dry-run", action="store_true",
                        help="poll and log the phase without sending any turn")
    parser.add_argument("--stop-on-turn-error", action="store_true",
                        help="stop when a turn exits non-zero, instead of polling the phase again")
    args = parser.parse_args()
    if "LOAM_STATE_DIR" not in os.environ:
        raise SystemExit("drive_campaign: LOAM_STATE_DIR is unset; run through ./run-campaign --drive")
    return Driver(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
