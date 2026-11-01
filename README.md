# Loam

Loam turns an agent harness into a hypothesis-driven scientific campaign runner.
It is a set of lifecycle hooks and four agent definitions: a **reasoner** that
synthesizes each round into the next tournament's input, **hypothesis
validators** that each own one ranked hypothesis and design the experiment to
falsify it, **code-generators** that write and run the code, and
**evidence-reviewers** that check each claim against its source or its recorded
artifact before anything is accepted. The hypothesis tournament itself is
[Jnana](https://github.com/ramanathanlab/jnana); the harness is Hermes.

A campaign can run its rounds unattended — `./drive-campaign` advances them and
accepted evidence steers the next one — but nothing is accepted on an agent's
word. See [How a round closes](#how-a-round-closes).

It is the spiritual successor of StructBioReasoner, with the execution details
condensed into skills.

**Loam is domain-agnostic and ships no campaign content.** There is no goal file
and no rule naming a particular measurement in this repo. What the campaign is
for, and which experiment decides a hypothesis, are yours to write.

## Division of responsibility

| You own | Loam owns |
| --- | --- |
| The execution layer: a Globus Compute endpoint on the cluster, and the HPC MCP server in front of it | Deciding what to run on it, and enforcing who may call it |
| Inference endpoints for the harness and the tournament | Which tier gets which model, and what each is told |
| Configuration: the Hermes config and the Loam profile | Validating it, and refusing to launch on a bad one |
| Campaign content: the goal, and any rules specific to your campaign | The rules invariant across every campaign |
| Launching | Orchestration, provenance, and the fail-closed preflight gate |
| Calibration: auditing what review accepts, and how often | Independence of the acceptor from the producer, and the weight an accepted claim carries |

Nothing here reaches out and provisions compute for you. Loam assumes the
execution layer is already live and will abort the launch if it is not.

## Setup, in order

> **Starting a new campaign, or handing one to a collaborator?** Copy
> [`examples/campaign-starter/`](examples/campaign-starter/) — a runnable
> skeleton of the profile, a goal, two scoped rules, and a checklist of what no
> script can supply. It covers steps 4 and 5 below; the rest of this list is the
> machine setup it assumes.
>
> ```bash
> cp -r examples/campaign-starter ~/campaigns/my-campaign
> ```

Each step is detailed in [ONBOARDING.md](ONBOARDING.md); this is the shape.

**1. Stand up the execution layer.** A Globus Compute endpoint running on the
HPC login node (templates in `globus_compute_endpoint_configs/` for Aurora and
Polaris), then the HPC MCP server from the `hpc-deployment` skill
(`skills/hpc-deployment/scripts/mcp_server.py`), which submits the PBS
allocation and exposes `run_code`, `run_batch` and `inspect_filesystem` as MCP
tools. Do not trust it until `smoke_test_client.py` passes. Shut it down with
SIGINT, never `kill -9` — a clean exit is what releases the allocation.

**2. Bring up inference.** One OpenAI-compatible base for the harness and its
subagents, one for the tournament judge. The preflight starts the ALCF Minerva
proxy if nothing is listening on its port (`ALCF_PROXY=0` to opt out); the SSH
tunnel is yours to bring up.

**3. Configure Hermes.** In the Hermes home's `config.yaml`: enable the
`loam-hooks` plugin and point `mcp_servers.hpc` at the server from step 1. See
`examples/hermes-config.yaml.example`.

**4. Write the Loam profile.** Copy the starter kit's `loam.yaml` (or
`examples/loam.yaml.example`, which annotates every key more fully) to
`~/.hermes/loam.yaml` and fill in every `<PLACEHOLDER>`. This is where the
tournament settings, the `execution:` contract (including the `setup:` recipe
handed verbatim to every code-generator child) and the `campaign:` block live.
It is deliberately outside the repo: it holds your paths and endpoints.

**5. Write your campaign.** In a directory you own, named by `campaign.dir` —
the starter kit is already this shape, with a template for each file:

```
<campaign.dir>/campaigns/<goal>.md   # what the campaign is for
<campaign.dir>/rules/<name>.md       # rules specific to this campaign
```

```yaml
    campaign:
      dir: /home/you/campaigns/mine
      goal: my-campaign                # <dir>/campaigns/my-campaign.md
      rules: [my-protocol]             # <dir>/rules/my-protocol.md
```

The goal states the objective, success criteria, constraints, and the decision
it informs; it reaches all three tiers. A rule is policy injected into the tiers
its `scope:` names — often a few lines is enough, saying which measurement
decides a hypothesis, or which hypotheses and tool categories are out of
bounds. A campaign with no rules of its own still runs governed by the invariant
ones. See [`.loam/rules/README.md`](.loam/rules/README.md) for the layers, the
front matter, and what a rule has to answer.

Everything past the front matter in these files is injected into agent context
verbatim, and a rule arrives labelled authoritative over conflicting
instructions. So keep them short, and leave no editorial notes in them — an HTML
comment is not stripped, and is read as policy.

**6. Launch.**

```bash
mkdir -p ~/campaigns/my-campaign
cp <loam-repo>/run-campaign <loam-repo>/drive-campaign ~/campaigns/my-campaign/
cd ~/campaigns/my-campaign && ./run-campaign
```

The directory name is the campaign name, so copying the directory starts a
separate campaign. `run-campaign` gives each campaign its own isolated Hermes
home and writes every artifact inside it, then runs a fail-closed preflight —
one real completion per model role and one returned `run_code` probe — before
`exec hermes`.

Overrides, for a scratch or diagnostic launch:

```bash
LOAM_SKIP_PREFLIGHT=1 ./run-campaign    # no readiness gate, and no proxy or
                                        # allocation auto-start; never a real run
ALCF_PROXY=0 ./run-campaign             # ranking model served some other way
LOAM_REPO=/path/to/loam ./run-campaign  # required when the script was copied out
```

`run-campaign`'s header documents every variable it reads, including which ones
it deliberately discards — the tournament cadence is a profile setting
(`pre_llm_every`), not an environment override.

**7. Drive it.** A launch leaves you at an interactive prompt with nothing
started: the pre-LLM hook fires on a turn, so the campaign begins when you take
one. `Begin the campaign.` is enough — the goal, the rules, and the round's
scheduler directive are all injected, so the message itself carries no content.
Thereafter one scheduler step happens per turn, so `continue` advances the
round.

To run it unattended, copy `drive-campaign` alongside `run-campaign` and use it
**instead of** the launcher — it starts the campaign itself:

```bash
./drive-campaign --max-rounds 4 --max-hours 8
```

It polls the round phase and types `continue` only when a turn will do
something, because every coordinator turn re-runs the tournament before the
orchestration hook looks at the phase, and a timer-driven loop pays minutes of
ranking to be told the validators are still running. It runs the launcher in a
pseudo-terminal rather than looping `hermes -z`, which cannot drive a campaign
at all: the dispatch tool needs a parent-agent context oneshot never creates,
and a top-level delegation is backgrounded, so the turn would exit while its
validators ran. `--help` lists the budgets and stop conditions — a round or
wall-clock budget, a cap on nudges, a phase that will not advance, an endpoint
it cannot restore, a repeated reasoner brief, and consecutive rounds that accept
no new evidence. Set `LOAM_NOTIFY_CMD` to a push command if nobody will be
reading the terminal.

See [ONBOARDING.md](ONBOARDING.md) Step 6 for the per-turn breakdown and what to
watch in `state/`.

## How a round closes

A round is a phase machine the orchestration hook reads out of the campaign's
own ledger, not a script. Each turn it works out where the round stands and
injects the next directive; `loam_dispatch_round` then launches that work from
persisted state, so no agent has to transcribe a plan it cannot see in full.

| Phase | What the turn does |
| --- | --- |
| `idle` | the tournament ranks, and the round's hypotheses are selected |
| `validators_due` | one hypothesis-validator per selected hypothesis |
| `validators_running` | nothing — a turn here only pays for a tournament |
| `reviewers_due` | one evidence-reviewer per pending claim set *(opt-in)* |
| `reviewers_running` | nothing |
| `reasoner_due` | one reasoner, synthesizing the brief that steers the next tournament |

### Nothing is accepted on an agent's word

A validator's report is untrusted model output. Its findings travel as a
`hypothesis_provenance_v1` candidate — a claim, a resolvable `source_url`, and a
`supports` / `contradicts` / `context` stance — which lands `pending` and
carries no weight at all until a **review** accepts it.

What that boundary protects is **independence of the acceptor from the
producer**, not humanity as such: a claim with a resolvable source and a named
artifact is checkable by anyone who did not make it. So a review has two
possible authors, and the reconciler cannot tell them apart:

- **an operator**, writing review rows by hand;
- **an `evidence-reviewer` child**, dispatched per candidate by the hook when
  the profile opts in.

The reviewer gets the claims, their sources, and the recorded output manifest of
the hypothesis's completed runs — deliberately **not** the validator's prose,
which is the object under review. It returns a verdict per claim and nothing
else: it cannot execute code, so it cannot recompute the number it is checking
and call the agreement verification, and it cannot weight its own acceptance.
The weights come from the verification mode it declares:

| Mode | What it establishes |
| --- | --- |
| `source_confirmed` | an independent record was read and says what the claim says |
| `artifact_attested` | the number provably came from an artifact of a declared run whose hash Loam recorded — the failure the boundary exists to catch. It says nothing about whether the metric is *true* |
| `unverifiable` | neither. A first-class verdict, not a rejection: most of what a computational campaign measures has no authoritative record to re-fetch |

Accepted evidence is then reconciled into two projections — hypothesis state and
a steering snapshot — which reach the next round two ways: the reasoner's brief
shapes what gets **invented**, and the steering snapshot re-orders what gets
**dispatched**, so a hypothesis whose own evidence contradicts it goes last and
is the first thing a round truncated by a walltime boundary fails to reach.
Nothing is dropped: a refuted claim is not a refuted hypothesis.

### The part that stays yours

A reviewer that accepts everything looks, from inside the campaign, exactly like
a campaign whose claims all happen to be good. No structure inside the loop can
tell those apart, so the check sits outside it:

```bash
python3 scripts/review_audit.py --state <campaign>/state --sample 5
```

Read-only. It prints the verdict split, how each acceptance was verified, the
per-reviewer accept rate — flagging any reviewer that accepted everything it saw
— acceptances carrying no note of what was checked, and a sample of accepted
claims to read in full. It is not per-round work, and it is what keeps an
autonomous acceptance boundary honest rather than decorative.

Full detail, including the review contract and the reconciler's inputs, is in
[`.loam/reconcile/README.md`](.loam/reconcile/README.md).

## Validate without a cluster first

Run the arithmetic campaign in [CAMPAIGN_TEST.md](CAMPAIGN_TEST.md) with
`execution.mode: local`. It exercises the whole
hook → validator → code-generator → reasoner → next-round path deterministically
and is the fastest way to confirm the wiring before spending an allocation.

The test suite needs no harness, MCP endpoint, or inference endpoint:

```bash
python -m pytest tests/           # use the interpreter with `pip install -e .`
```

PyYAML in particular is required — without it `load_profile` returns `None` and
the profile-parsing tests fail.

## Optional features and their toggles

These capabilities are off (or external) by default, and each is turned on where
the thing it governs already lives — there is no single global switch. The exact
keys and the reasoning are in [ONBOARDING.md](ONBOARDING.md) Step 4c; the
annotated `workspace:` block is in
[`examples/loam.yaml.example`](examples/loam.yaml.example).

| Feature | Turn on | Turn off (default) |
| --- | --- | --- |
| **Hypothesis workspace** — a Pando-style folder per hypothesis (see `.loam/README.md`, "Hypothesis workspace") | `workspace.enabled: true` in the profile; needs `hypothesis_orchestration.enabled: true` | `workspace.enabled: false`, or omit the block |
| **Provenance tracing** — hash-chained decision log, per-run traces, reasoner→validator→code-generator dispatch edges | comes **with** the workspace; no separate switch. `workspace.trace_full: true` keeps full tool-result bodies | disable the workspace. `trace_full: false` strips already-hashed skill reads from traces |
| **Weights & Biases** — experiment tracking from a campaign's own code | not a Loam key: `wandb login` once, then export `WANDB_MODE=online` (or `wandb.init(mode=...)`) in `execution.setup:` so the generators' code picks it up | `WANDB_MODE=disabled`, and the default when no code calls `wandb` |
| **Independent evidence review** — an `evidence-reviewer` per pending claim set, then automatic reconciliation (see [How a round closes](#how-a-round-closes)) | `hypothesis_orchestration.evidence_review: true` | omit it. Candidates then stay `pending` unless a person writes the review JSONL, so nothing a validator measures reaches steering |
| **Review backlog catch-up** — widens review past the active round to every pending candidate, five per turn | `hypothesis_orchestration.evidence_review_backlog: true`, for a campaign that ran before review existed | omit it. Leave it off once the pile is drained, or a long campaign keeps re-opening old rounds |

Provenance tracing has one piece that is always on and not configurable: every
run manifest records the goal and rule files' SHA-256 digest by name, so a
result is attributable to the policy that produced it even with the workspace
off.

## Run enzyme reactions with Reactra

The [Reactra skill](skills/qm/reactra/SKILL.md) covers YAML-driven enzyme-reaction
simulations, including packaged PMD and chorismate-mutase examples, HPC execution,
recovery, and interpreting crossings versus barriers. Make the skill available in
the campaign's Hermes skill library and add `reactra` to `campaign.skills` in your
Loam profile so code-generators consult it. The
[execution guide](skills/qm/reactra/references/loam-execution.md) explains environment
setup and resource ownership for direct and cluster execution.

## Where things live

| Path | What |
| --- | --- |
| `.loam/manifest.yaml` | The harness-neutral hook contract |
| `.loam/hooks/` | Hook implementations; `hypothesis_orchestration.py` is the orchestrator |
| `.loam/agents/` | The four agent definitions |
| `.loam/reconcile/` | The acceptance boundary: the reconciler, the projectors, and their README |
| `.loam/rules/` | Rules invariant across every campaign, and their README |
| `.loam/adapters/hermes/` | The Hermes plugin that invokes the hooks |
| `.loam/README.md` | Lifecycle-hook behavior, and global Hermes installation |
| `skills/` | Domain procedures the campaign's code draws on |
| `scripts/campaign_preflight.py` | The fail-closed readiness gate |
| `drive-campaign`, `scripts/drive_campaign.py` | Unattended round advancement |
| `scripts/review_audit.py` | What review has been accepting, for calibration |
| `examples/campaign-starter/` | A new campaign's skeleton: profile, goal, rules, checklist |
| `examples/*.example` | Fully annotated config templates to copy outside the repo |
| `scripts/install_skills.py` | Registers `skills/` in a Hermes library (run-campaign does this) |
| `~/.hermes/loam.yaml` | Your profile (never tracked here) |
