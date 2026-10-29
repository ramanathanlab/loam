# Campaign starter kit

Copy this directory, fill in every `<PLACEHOLDER>`, and you have a runnable
campaign. Nothing here is read from this location — the files are templates for
three different destinations, listed below.

```
cp -r <loam-repo>/examples/campaign-starter ~/campaigns/my-campaign
cd ~/campaigns/my-campaign
```

The **directory name is the campaign name**, so rename it to something you will
recognize in a ledger six weeks from now.

## Where each file goes

| This file | Goes to | Why there |
|---|---|---|
| `loam.yaml` | `~/.hermes/loam.yaml` | One active profile per machine. Not in the campaign dir. |
| `campaigns/<goal>.md` | stays here | `campaign.dir` points at this directory. |
| `rules/<name>.md` | stays here | Same. |
| — | `<loam-repo>/.hermes/.loam-home/config.yaml` | Copy `examples/hermes-config.yaml.example`. Shared across campaigns. |
| — | this directory | Copy `<loam-repo>/run-campaign` here. |

So: `loam.yaml` moves to `~/.hermes/`, the `campaigns/` and `rules/` directories
stay put, and you add `run-campaign` beside them.

## Checklist

Each item is something no script can do for you.

1. **`~/.hermes/loam.yaml`** — every `<PLACEHOLDER>` in `loam.yaml`, especially
   `campaign.dir` (the absolute path of *this* directory), `jnana_root`, and the
   `execution.setup` recipe.
2. **`campaigns/<goal>.md`** — your objective. Rename the file and update
   `campaign.goal` to match. A named-but-missing goal raises at startup and lists
   every directory it searched.
3. **`rules/<name>.md`** — optional. Keep the ones you want, delete the rest, and
   list the survivors in `campaign.rules`. A campaign with no rules of its own
   still runs under the invariant rules the repo ships.
4. **Hermes config** — copy `examples/hermes-config.yaml.example` into the shared
   home as `config.yaml`. Fill in `model:`, `custom_providers:`, and the
   `delegation:` block — that last one is the only place child agents get a
   model, so a campaign with it missing has validators and code-generators
   running on nothing.
5. **Credentials** — `hermes` login writes `auth.json`. Never commit it.
6. **HPC env** — edit `skills/hpc-deployment/set_env_akv.sh`: your
   `HPC_GLOBUS_ENDPOINT_ID` (the Globus Compute **manager UUID**),
   `HPC_BASE_DIR`, `HPC_CAMPAIGN_ID`, `HPC_INSPECT_ROOTS`, and the venv paths
   inside `worker_init`. If you let the preflight auto-start the allocation,
   export `HPC_ENV_FILE` pointing at the file you actually edited — its default
   is `set_env.sh`, which is somebody else's endpoint.
7. **A Loam venv** at `~/.loam-venv-312`, or export `LOAM_PREFLIGHT_PYTHON`.

## Writing the goal and the rules

Everything in these files past the `---` front matter is **injected verbatim into
agent context**, and the rules arrive labelled as authoritative over any
conflicting instruction. Two consequences:

- **Do not leave editorial notes in them**, including HTML comments — `<!-- -->`
  is not stripped, so a leftover note is read as campaign policy. That is why the
  templates carry no comments and this guidance lives here instead.
- **Keep them short.** Both are injected on every turn. A tight page beats five;
  a long one crowds out the round's actual instructions.

**The goal** reaches all three tiers, so write it for an agent with no other
context about why it is working. Rename the file and set `campaign.goal` to the
new basename.

**A rule** declares the tiers it reaches in `scope:` — `validator`,
`code-generator`, `reasoner`, a comma-separated combination, or `all`:

| Template | `scope:` | Why |
|---|---|---|
| `example-protocol` | `all` | The validator picks the experiment, the code-generator runs it, the reasoner weighs it — all three need one definition of success. |
| `example-constraint` | `validator,reasoner` | Only these two choose what to do next. A code-generator does not select hypotheses, so search-space policy is noise in its context. |

A protocol rule's job is to name the decisive measurement and its metrics, so
that three validators cannot each invent a different definition of success for
the same campaign. A constraint rule's job is what is out of bounds. Write the
bodies imperatively — hedged prose reads as policy too.

`name:` must match the filename, `scope:` is required (there is no default, so a
rule cannot silently reach a tier it was not written for), and omitting
`applies:` means the rule loads only when `campaign.rules` names it. Delete a
template you do not want and drop its name from `campaign.rules`; an unnamed file
left in `rules/` is simply never loaded.

## Before you launch

Three things must be answering, or the fail-closed preflight aborts:

| Port | What | Who starts it |
|---|---|---|
| `:51900` | Argo tunnel — reasoner and every subagent | you |
| `:51911` | ALCF proxy — the tournament's ranking judge | `run-hermes` does; **`run-campaign` does not** |
| `:8000/mcp` | HPC code execution | you (see the `hpc-deployment` skill). If nothing is listening, the preflight starts a bounded 1-node debug allocation and waits. |

Prove the wiring without a cluster first: run the arithmetic campaign in
`CAMPAIGN_TEST.md` with `execution.mode: local`. It exercises the whole
hook → validator → code-generator → reasoner → next-round path and costs no
allocation.

## Launching

```bash
./run-campaign
```

This installs the skills into your Hermes library, builds an isolated Hermes home
for this campaign, runs the preflight into `state/preflight.json`, and starts
Hermes. **It does not start the campaign.** You are left at a prompt; the first
turn is what begins round one:

```
Begin the campaign.
```

Do not restate the objective — the goal file, the rules, and the round's
directive are all injected for you, and a retyped copy can contradict the file
without the provenance record showing it. For an unattended run:

```bash
./run-campaign -z "Begin the campaign."
```

Afterwards, one scheduler step happens per turn — tournament → validators →
reasoner brief → next tournament — so `continue` advances a round. Watch
`state/hypothesis-campaign.jsonl` for the orchestration ledger,
`state/tournament-reports/` for each round's ranking, and
`state/parent-memory.jsonl` for accepted findings.

See `ONBOARDING.md` in the repo for the full version of any step here.

## Validating your copy before you burn an allocation

The loader rejects some mistakes and silently accepts others. Check both:

```bash
# 1. The profile parses and the goal/rules resolve. Raises if a name is missing,
#    listing every directory searched.
LOAM_CONFIG_PATH=~/.hermes/loam.yaml python3 -c "
import sys; sys.path.insert(0, '<loam-repo>/.loam/adapters/hermes/plugin')
import loam_config as c; p = c.load_profile()
print(p.campaign.goal, list(p.campaign.rules), list(p.campaign.skills))"

# 2. Every skill you named actually resolves. A name that does not is NOT
#    caught above -- it fails inside a code-generator mid-round.
hermes skills list | grep -E 'chai|proteinmpnn'
```

`campaign.skills` entries must be bare names: a leftover `<PLACEHOLDER>` there
raises, while a well-formed name for a skill that is not installed does not.
