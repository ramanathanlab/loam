# Loam

Loam turns an agent harness into a hypothesis-driven scientific campaign runner.
It is a set of lifecycle hooks and three agent definitions: a **reasoner** that
synthesizes each round into the next tournament's input, **hypothesis
validators** that each own one ranked hypothesis and design the experiment to
falsify it, and **code-generators** that write and run the code. The hypothesis
tournament itself is [Jnana](https://github.com/ramanathanlab/jnana); the
harness is Hermes.

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
subagents, one for the tournament judge. `run-hermes` will start the ALCF
Minerva proxy if nothing is listening on its port; the rest is yours.

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
cp <loam-repo>/run-campaign ~/campaigns/my-campaign/
cd ~/campaigns/my-campaign && ./run-campaign
```

The directory name is the campaign name, so copying the directory starts a
separate campaign. `run-campaign` gives each campaign its own isolated Hermes
home and writes every artifact inside it, then runs a fail-closed preflight —
one real completion per model role and one returned `run_code` probe — before
`exec hermes`.

To run from the repo root instead, without a per-campaign directory:

```bash
./run-hermes                          # pre-LLM hook every turn
LOAM_PRE_LLM_EVERY=3 ./run-hermes     # ... every three turns
LOAM_STATE_DIR=/var/tmp/loam/$ID ./run-hermes   # isolate this run's artifacts
```

**7. Drive it.** A launch leaves you at an interactive prompt with nothing
started: the pre-LLM hook fires on a turn, so the campaign begins when you take
one. `Begin the campaign.` is enough — the goal, the rules, and the round's
scheduler directive are all injected, so the message itself carries no content.
Unattended, pass a one-shot prompt through to Hermes:

```bash
./run-campaign -z "Begin the campaign."
```

Thereafter one scheduler step happens per turn (tournament → validators →
reasoner brief → next tournament), so `continue` advances the round. See
[ONBOARDING.md](ONBOARDING.md) Step 6 for the per-turn breakdown and what to
watch in `state/`.

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
| `.loam/agents/` | The three agent definitions |
| `.loam/rules/` | Rules invariant across every campaign, and their README |
| `.loam/adapters/hermes/` | The Hermes plugin that invokes the hooks |
| `.loam/README.md` | Lifecycle-hook behavior, and global Hermes installation |
| `skills/` | Domain procedures the campaign's code draws on |
| `scripts/campaign_preflight.py` | The fail-closed readiness gate |
| `examples/campaign-starter/` | A new campaign's skeleton: profile, goal, rules, checklist |
| `examples/*.example` | Fully annotated config templates to copy outside the repo |
| `scripts/install_skills.py` | Registers `skills/` in a Hermes library (run-campaign does this) |
| `~/.hermes/loam.yaml` | Your profile (never tracked here) |
