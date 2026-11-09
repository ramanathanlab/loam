# Automating round advancement

Status: **plan, not built.** Written 2026-10-31 after driving round one by hand.

## The problem

A Loam campaign advances one scheduler step per turn. The hook injects the next
action; you supply the turn. From `ONBOARDING.md`:

| Turn | Injected directive | What the parent does |
|---|---|---|
| 1 | tournament runs, then *N validators planned* | calls `loam_dispatch_round` |
| 2 | (validators still running → nothing) | waits |
| 3 | all validators complete → reasoner directive | synthesizes the brief |
| 4 | brief steers a fresh tournament | next round begins |

So a round is two to three turns, each of which someone has to type. For a
multi-day campaign on a long allocation that is the only manual step left.

## What does not work

### `--yolo`

`hermes --help`: *"Bypass all dangerous command approval prompts."* It suppresses
approval dialogs. It does not cause Hermes to take another turn. Neither does
`-z/--oneshot`, which already auto-bypasses approvals and then **exits after one
turn**. There is no "run until done" flag.

### Looping `run-campaign -z "continue"`

This is the obvious idea and it is **unsafe**. `run-campaign` unsets and re-mints
the run identity on every launch:

```bash
# run-campaign:88 -- "A launch owns its state identity. Never inherit these
#                     from a parent shell."
unset LOAM_STATE_DIR LOAM_RUN_ID ...
# run-campaign:184
export LOAM_RUN_ID="$(uuidgen | tr '[:upper:]' '[:lower:]')"
```

and `_active_plan()` in `.loam/hooks/hypothesis_orchestration.py:101` is scoped
to that run:

```python
return row.get('workflow_id') == workflow_id and (not run_id or row.get('run_id') == run_id)
# "Scoped to the run, so a plan an earlier run left unfinished cannot deadlock this one."
```

A relaunch therefore cannot see the in-flight round. `_round()` falls through to
`latest_report()`, finds the last tournament report, and returns
`validators_due` — so it **re-dispatches validators against a report that was
already validated**. Duplicate compute, duplicate LLM spend, and a provenance
record that claims two validations of one ranking.

Pinning `LOAM_RUN_ID` from the parent shell is deliberately prevented by the
`unset` above. Do not work around it without understanding why that line exists.

### A plain pipe or FIFO into `launch.sh`

Hermes requires a terminal. Feeding it from a non-TTY produces:

```
Warning: Input is not a terminal (fd=0).
… Shutting down… (finalizing session)
```

It starts, finds no TTY, and exits. Verified 2026-10-31 by running
`! ./launch.sh` from a Claude Code session.

## What should work

One session, one `run_id`, fed turns by a driver that knows when a turn is
worth taking.

### 1. Give Hermes a pseudo-terminal

`expect`, or the simpler `script -q /dev/null ./launch.sh`. The session stays
alive across rounds, so `run_id` and the ledger scoping stay consistent.

### 2. Drive on state, not on a timer

This is the part that matters. **Every turn re-runs the tournament** before the
orchestration hook looks at the phase, because `pre-llm.py` calls
`Tournament(payload).context()` (line 46) before
`orchestration_context_for(payload)` (line 47). At `matches_per_run: 10` that is
~3 minutes of serialized ranking per turn, measured ~16-21 s/match. A timer-based
loop would burn that repeatedly to be told "validators still running".

So poll the phase and only send a turn when it is actionable:

| Phase | Send `continue`? | Why |
|---|---|---|
| `validators_due` | yes | dispatch the round's validators |
| `validators_running` | **no** | the turn does nothing but pay for a tournament |
| `reasoner_due` | yes | synthesize the brief |
| `reasoner_running` | **no** | same as above |
| `idle` | yes | start the next round |

The phase is one call, the same one used to diagnose round one by hand:

```python
import loam_config as lc, hypothesis_orchestration as o
p = lc.load_profile()
payload = dict(lc.payload_values(p))
payload.update(workflow_id=p.workflow_id, session_id=..., parent_session_id=...)
state = o._round(payload, o._rows(payload))
state['phase']            # -> validators_due | validators_running | reasoner_due | ...
```

Note `_round()` takes `loam_run_id` from the payload, so the driver must pass the
**live session's** run id, not a fresh one, or it will read the same misleading
`validators_due` that a relaunch sees.

`_round()` is a private helper. Preferably ask Loam for a supported way to read
the phase rather than importing an underscore name; failing that, pin the import
and expect it to break on refactor.

### 3. Stop conditions

A driver that only ever sends `continue` will run until the allocation dies.
It should stop on:

- a configured round count, or a wall-clock budget;
- `parent-memory.jsonl` not growing across N rounds (the campaign has stopped
  producing accepted findings);
- the goal's own success criteria being met;
- repeated identical outcomes — e.g. round one returned five negative structural
  gates; a driver should not silently run twenty more rounds of that.

## Prerequisites

- **A long block.** `debug-scaling` caps walltime at 1 h, so an unattended run
  loses in-flight validator work every hour to `ManagerLost`, and
  `HPC_MIN_BLOCKS=1` then re-provisions cold. `capacity` allows 168 h at ≤16
  nodes, which fits the 10-node shape. Unattended on `debug-scaling` is not
  worth building.
- **A pool-health check per round.** `:8000` listening is *not* evidence of a
  usable pool — observed 2026-10-31, the server bound the port with
  `cluster pool ready: 0` against an expiring block. The driver should run the
  placement probe (`run_code` returning `hostname`, `pbs_jobid`, `device_pin`)
  before dispatching work, and wait for a replacement block to reach `R` before
  trusting the pool.
- **Alerting.** Unattended means nobody reads the negative results as they
  arrive. Worth pushing a summary per round.

## Resolved: run_id comes from the environment, so it can be pinned

`loam_config.py:685` is the whole story:

```python
run_id = _clean_text(os.environ.get("LOAM_RUN_ID"))
...
values["loam_run_id"] = run_id
```

`run_id` is read from the environment and nowhere else, and
`_round()` passes `payload['loam_run_id']` straight into `_active_plan()`. So
resuming a session does **not** preserve it by itself — `run-campaign:184` mints
a fresh one on every launch, which is why a naive relaunch loses the round.

But that also means it is **pinnable**, and the live value is recoverable:

```bash
python3 -c "import json;print(json.load(open('state/run-manifest.json'))['identity']['run_id'])"
# or from any ledger row's run_id field
```

This removes the need for a pty. `-z/--oneshot` is non-interactive by design, so
the driver can be a plain loop:

```bash
RUN_ID=$(…read from run-manifest.json…)
# only when the phase is actionable:
LOAM_RUN_ID="$RUN_ID" hermes -c "$SESSION" -z "continue"
```

The catch is that `run-campaign` deliberately unsets `LOAM_RUN_ID` before
re-minting it ("A launch owns its state identity"), so the driver must either
invoke `hermes` directly with the rest of `run-campaign`'s environment, or
`run-campaign` must gain an explicit opt-in (e.g. honour `LOAM_RUN_ID_PIN`).
Patching it is the smaller change, but that `unset` is a deliberate guard — take
it up with whoever wrote it rather than quietly removing it.

**Still untested:** whether `hermes -c <session> -z "continue"` actually resumes
the conversation state as well as the session name.

## The acceptance boundary is deliberately not automatable

This is the hard limit on how autonomous a campaign can be, so design around it
rather than against it.

`.loam/reconcile/README.md`: *"Reconciliation is the acceptance boundary. A
child's summary is untrusted... A candidate receives no finding, evidence,
memory, or credit until a reconciler is run explicitly against a review."*

`hypothesis_provenance.py` takes `--reviews`, described as **"Operator-authored
JSONL acceptance reviews"**. The README states the intent plainly:
*"acceptance is a human decision about a sourced claim rather than an automatic
equality check."* The earlier contract that *did* auto-verify by re-fetching the
source field by field (`protein_structure_metadata_v1`) was **removed on
2026-10-28**, along with its `project_parent_memory.py` projector.

So in a fully autonomous run:

| Artifact | Writer | Grows unattended? |
|---|---|---|
| `jnana-memory.json` (hypotheses, Elo) | tournament | yes |
| `tournament-reports/` | tournament | yes |
| `hypothesis-campaign.jsonl` | lifecycle hooks | yes |
| `candidate-results.jsonl` (`pending`) | lifecycle hooks | yes |
| `parent-memory.jsonl` | **the reasoner, by judgment** | yes, if the reasoner chooses to append |
| `hypothesis-memory.jsonl`, accepted evidence, credit | projection from a reconciled **review** | **no — requires an operator** |

Consequences for a driver:

- **Do not treat an empty `parent-memory.jsonl` as a malfunction**, and do not
  make the loop wait on accepted evidence. It will never arrive on its own.
- **Surface the pending-candidate backlog** as the thing needing human
  attention: `candidate-results.jsonl` rows with `candidate_status: pending`.
  Consider pausing the loop when that backlog crosses a threshold, since
  everything after it is unaccepted by definition.
- **Reconciliation is a batch operation between rounds**, not a per-turn step.
  Natural rhythm: run N rounds unattended, then review the accumulated
  candidates and reconcile once.
- If a campaign genuinely wants automatic verification, the README says that is
  "a new contract plus a source adapter, and it belongs to the campaign that
  needs it" — not a change to this boundary.

Observed 2026-10-31 after two completed rounds (10 validators): 4 pending
candidates, 0 accepted, `hypothesis-memory.jsonl` absent, `parent-memory.jsonl`
absent. All of that is consistent with the design, not a fault — except the
reasoner appending nothing, which is its own call to make.

## Proposal: an evidence-reviewer role, so acceptance can run unattended

The acceptance boundary protects **independence of the acceptor from the
producer**, not humanity as such. A claim carrying a resolvable `source_url` and
a stance is checkable by anyone who did not make it. So the boundary survives if
the reviewer is a *different* agent with access to primary artifacts — and the
repo already contains both halves of the pattern:

- `hypothesis-validator.md:120-125` requires `source_url` ("must resolve") and
  `stance` ∈ `supports` / `contradicts` / `context`.
- `reconcile/verify_workspace.py` already verifies autonomously on HPC: it
  re-runs the manifest program through `run_code`, re-hashes outputs, and logs
  anything changed, removed or added as a violation.
- `reconcile/README.md` explicitly allows automatic verification as "a new
  contract plus a source adapter... it belongs to the campaign that needs it".

**No source adapters are needed for this.** That phrasing was written for the
retired `protein_structure_metadata_v1`, which compared *named fields of a
specific record type* and so needed a per-source adapter — and which was retired
precisely because that made it campaign-specific, not because auto-verification
was unsound. `hypothesis_provenance_v1` is deliberately domain-neutral: "a
hypothesis key, a statement, and evidence carrying a claim, a resolvable
`source_url`, and a `supports` / `contradicts` / `context` stance. That shape
fits any field." Verifying it requires only *fetch the URL and judge whether the
content supports the claim* — which needs no schema knowledge of the source and
so stays generic. A reviewer role is therefore framework-level, not campaign
-level, which is the bar this project is building to.

What is missing is a role. `agent_registry` knows only `code-generator`,
`hypothesis-validator` and `reasoner` (`.loam/agents/*.md`), so there is nobody
to dispatch for review.

### Shape

1. New role `evidence-reviewer` in `.loam/agents/evidence-reviewer.md`.
2. New phase between `validators_due` and `reasoner_due`: when validators have
   completed and `candidate-results.jsonl` has `pending` rows, dispatch
   reviewers — one per candidate, or batched by source.
3. Each reviewer receives the **candidate claim, its `source_url`, and the
   artifact paths** — deliberately *not* the validator's prose summary, which is
   the untrusted object.
4. Reviewers emit the operator-review JSONL that `hypothesis_provenance.py`
   already expects via `--reviews`.
5. The hook then runs `hypothesis_provenance.py` plus
   `project_hypothesis_memory.py` / `project_hypothesis_steering.py`.

Nothing downstream changes: the reconciler's interface is unchanged, and the
review file is simply authored by an independent agent instead of a person.

### Tier the claims, because "re-fetch" only verifies some of them

| Tier | Example | How a reviewer verifies it | Automatable |
|---|---|---|---|
| Retrieval | `P00533` sequence length, `1NQL` resolution | re-fetch `source_url`, compare | **yes** |
| Computed | patch SASA, cross-conformer Cα RMSD, paralog divergence | confirm the artifact exists, its hash matches the manifest, and the claimed number is actually *in* the artifact | **yes** — this is exactly the campaign's own rule ("inspect an actual output file before reporting any aggregate number from it") |
| Judgment | "this epitope is worth pursuing" | not mechanically checkable | **no** — stays with the reasoner, and with you |

Tier 2 is the important one: it does not require recomputing the science, only
proving that the reported number faithfully came from the artifact. That catches
the failure the boundary exists for — a summary asserting something its own
output files do not say.

### Unverifiable metrics still have to reconcile into better/worse

Most of what a computational campaign produces cannot be verified against a
source at all. An ipTM, an MM-PBSA ΔG, a predicted RMSD — the number *is* the
computation's output; there is no authoritative record to re-fetch. Those must
still reconcile into a comparative judgement, or the acceptance boundary throws
away most of the campaign's evidence.

Separate two things the word "verify" is doing:

| Mode | What is established | Applies to |
|---|---|---|
| `source_confirmed` | an independent record supports the claim | retrieval claims |
| `artifact_attested` | the claimed number genuinely came from a named artifact, whose hash is recorded, produced by a declared tool + version + inputs | computed metrics |
| `unverifiable` | neither | judgement, or a number with no artifact |

`artifact_attested` is not weaker bookkeeping — it establishes the thing the
boundary exists for: that a summary's number is not invented. Truth of the
metric is a separate question and often unanswerable.

**For comparison, what matters is method identity, not truth.** A metric is
comparable across candidates when it was produced the same way. This project
already states that principle, in the campaign profile's own skills comment:
naming exactly one folding tool is deliberate, because "left to pick among
chai/boltz/esmfold2/protenix, 150 validators will not all choose the same one
and their results stop being comparable."

So an unverifiable metric should reconcile as comparable **only** when it
carries a method fingerprint — tool, version, parameters, input hashes — and
only *within* a cohort sharing that fingerprint. Cross-fingerprint comparison
should be refused or flagged, never silently averaged.

Two routes into the existing steering projector, in increasing order of change:

1. **Threshold to a stance.** Validators already predeclare gates (round one's
   reports did: "≤3.0 Å threshold", "relative SASA ≥0.20"). Let the reviewer
   apply the predeclared threshold and emit `supports` / `contradicts`.
   Magnitude is used at threshold time and not stored. Needs **no contract
   change** and works with the projector as written.
2. **Carry the measurement.** Extend the evidence item with an optional, still
   domain-neutral block: `metric_name`, `value`, `units`,
   `direction: higher_better | lower_better`, `method_fingerprint`. Steering can
   then rank *within* a fingerprint cohort instead of only counting stances.
   More powerful, and the thing to be careful with — a number with a unit and a
   direction is generic, but a vocabulary of named metrics is how this slid into
   campaign-specificity last time.

Either way, `unverifiable` evidence should land in the **uncertainty** term
(already weighted 0.15) rather than silently carrying the same force as a
source-confirmed claim, and an attested metric should never be promoted to an
absolute claim — only a relative one.

### Guardrails, or this becomes rubber-stamping

- **The reviewer must never be the producing validator**, and must not see its
  summary. An agent handed a claim to confirm tends to confirm it.
- **Prompt it adversarially**: its job is to find the claim unsupported, and
  "unverifiable" must be a first-class verdict distinct from "rejected".
- **Keep a human sample.** Audit a random fraction of auto-accepted claims; if
  the reviewer's accept rate is ~100%, it is not reviewing.
- **Record the reviewer's identity and model in the review row**, so an
  accepted finding's provenance shows who accepted it and on what basis.
- **Do not let the reasoner dispatch reviewers for evidence it is synthesizing**
  — that reintroduces exactly the conflict of interest being removed. Dispatch
  belongs to the orchestration hook, structurally parallel to validators.

### Status

Not built. It changes shared code: a new agent definition, a new phase in
`hypothesis_orchestration.py`, and wiring in `canonical_delegation_args`.

The 2026-10-28 removal is **not** an argument against it. That contract and its
projector were generated for earlier, narrower campaigns and were too
campaign-specific for a generic framework, which is why they went. The reviewer
above is the opposite: it leans on the domain-neutral provenance contract and
needs no per-source knowledge, so it belongs in the framework rather than in any
one campaign.

The one thing that must stay operator-owned is **calibration**: the sampling
audit of auto-accepted claims, and the accept-rate check. Those are cheap, they
are not per-round, and they are what keeps an autonomous acceptance boundary
honest rather than decorative.

## What to do, in order (grounded in 2026-10-31 round 1-2 output)

Measured, not assumed: 10 validators produced **4** candidate contracts holding
**16** evidence candidates, with stances `context: 15, supports: 1,
contradicts: 0`. Every claim is source bookkeeping — *"PDB 1NQL supplied the
tested inactive EGFR ectodomain structure"* — while the findings themselves
(11.71 Å cross-conformer RMSD against a predeclared ≤3.0 Å gate; local
divergence 0.459 against a 0.875 cetuximab-like control) appear only in the
prose summaries and on-disk artifacts.

So the reviewer is **not** the first thing to build. In order:

### 1. Make validators emit findings, not just provenance

The contract shape is fine and already domain-neutral; it is being used as a
bibliography. `.loam/agents/hypothesis-validator.md` needs to require that each
**predeclared gate outcome** becomes an evidence candidate with:

- the stance that follows from the gate: a failed gate is `contradicts`, not
  `context`;
- a `source_url` pointing at the **artifact** that carries the number (the
  campaign already writes `results.json` with a recorded SHA256), not only at
  the upstream database record;
- the threshold and the measured value in the claim text, so the claim is
  checkable against that artifact.

Without this, nothing downstream can work: steering counts stances, so a
refutation that never becomes `contradicts` cannot lower anything's standing.
This is also the cheap route from the unverifiable-metrics section — magnitude
is used at threshold time and need not be stored, so **no contract change**.

### 2. Find out why 6 of 10 validators emitted no contract

4 of 10 is the ceiling on everything else. Compare the six silent validators'
summaries against the emission instructions: whether they were told clearly,
whether the extractor rejected a near-miss shape, or whether a stage that failed
early (one hit `RuntimeError: No mapped EGFR candidate chains`) simply had
nothing to emit. `build_candidate_records` in `.loam/hooks/ledger.py` is where
shape-validity is decided.

### 3. Restore a pool that lets steering matter

With 6 hypotheses and `top_k: 5`, selection takes 5 of 6 regardless of Elo or
evidence. The 376-hypothesis pool is intact at
`state/archive/jnana-memory.bigpool-20261031T121107.json`. Until the pool is
larger than `top_k` by a real margin, no amount of correct evidence changes what
gets validated next round.

### 4. Then the evidence-reviewer role

Only once there are substantive claims to review. Build it as described above,
starting with tier-1 retrieval claims and tier-2 artifact attestation.

### 5. Then the driver

Phase-driven turn advancement. It is the smallest piece and the least
interesting; it only pays off once 1-4 make an unattended round produce
something worth having.

## Open questions

- Is there an upstream-supported way to advance a round without re-minting the
  run identity? A `--rounds N` or a documented driver would make all of the
  above unnecessary — worth asking before building.
- **Why is `parent-memory.jsonl` still absent after two completed rounds?**
  Ten validators ran to completion and the reasoner synthesized a brief, but no
  accepted finding has been promoted. Automating more rounds is premature until
  this is understood: a loop that produces no accepted findings just spends an
  allocation quietly. Check whether promotion requires a positive result (round
  one was five negative structural gates) or is simply not firing.
- Should the driver skip the tournament on turns where only the phase matters?
  `matches_per_run: 0` makes `run_incremental_tournament` return the existing
  ranking without ranking anything (`if match_count:` guards the whole block),
  which would make a "just advance the round" turn nearly free. Untested as a
  per-turn override.
