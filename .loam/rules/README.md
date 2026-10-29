# Campaign rules

Policy text injected into a reasoning tier's context by `campaign_context()` in
`.loam/hooks/hypothesis_orchestration.py`, in the tiers each rule's `scope:`
front matter lists (`validator`, `code-generator`, `reasoner`, or `all`).

**This directory holds only rules invariant across every campaign.** Loam ships
no campaign content — no goal, and no rule naming a particular measurement.
Those come from a directory the user owns, named by `campaign.dir` in the
profile, because a framework carrying one field's protocol is a framework about
that field.

```yaml
    campaign:
      dir: /home/you/campaigns/mine        # your files, not in this repo
      goal: my-campaign                    # <dir>/campaigns/my-campaign.md
      rules: [my-protocol]                 # <dir>/rules/my-protocol.md
```

Names are resolved in `<dir>` first, then in this repo's `.loam/`, so a campaign
can shadow a shipped rule with its own text. `dir` is optional: a campaign kept
beside the code still works without it.

## What loads when

A rule declaring `applies: always` reaches every campaign, named or not — it
loads even from a profile with no `campaign:` block. Everything else loads only
when `campaign.rules` names it, which is the default for a file that omits
`applies:`.

A profile should not have to restate that an experiment must finish before it is
reported, so the invariant layers are always on; everything that differs between
campaigns is configuration. The safe default runs the other way, though: a
half-written rule in a directory must not reach a campaign it was not written
for, because the wrong decisive experiment is worse than a missing one.

| Layer | Holds | Lives in | `applies:` |
| --- | --- | --- | --- |
| Integrity | What makes a datum trustworthy at all | this repo | `always` |
| Discipline | How an experiment is run and reported | this repo | `always` |
| Shape | What a *kind* of campaign has to get right | either | named |
| Protocol | The decisive measurement and its metrics | `campaign.dir` | named |
| Constraint | What is out of bounds for this campaign | `campaign.dir` | named |

Base rules are injected before named ones, so a named rule is the most specific
text a child reads. A profile that names an invariant rule anyway keeps working —
the loader deduplicates rather than sending it twice.

The always-on layers must hold for a campaign in **any** field, testing any
hypothesis. That is stricter than domain-neutral: it also rules out assuming the
campaign's *shape*. "Paste what was designed, retrieve what was published" is not
invariant, because a campaign testing an existing claim designs nothing — so it
sits in `design-provenance.md`, which any design campaign names and a
hypothesis-testing campaign does not. A rule that says what the experiment *is*
never belongs here at all.

So: no tool, no metric, no molecule class, no campaign shape in an always-on
rule. If removing the example would leave the rule saying nothing, it was not
invariant.

## Starting a campaign

A new campaign needs **no new always-on rule**, and nothing committed to this
repo. The invariant layers already apply. In your own `campaign.dir`, write:

- `campaigns/<goal>.md` — what the campaign is for, its success criteria,
  constraints, and the decision it informs.
- `rules/<name>.md` — only what is specific to this campaign.

Often one constraint rule is all a campaign needs: a few lines naming hypotheses
not worth exploring, tool categories to avoid, or a budget ceiling. Give it
`scope: all` (or the one tier it binds) and name it in the profile. A campaign
with no rules of its own still runs governed by the invariant layers.

Add a protocol rule when the domain has a standard decisive measurement. It has
to answer four things, because they are what the discipline layer deliberately
leaves open:

1. What the decisive measurement is, concretely enough to write code for.
2. Which numbers it reports, and which one is the headline.
3. Where the procedure comes from — the `ENVIRONMENT SETUP` recipe, or a named
   skill once the code-generator can read skills (see `todo.md`).
4. What the measurement does **not** establish, so a child does not overclaim.

## Front matter

```
name:        must match the file name
scope:       validator | code-generator | reasoner | all   (required)
applies:     always | when-named                           (default when-named)
enforced_by: which pre-tool.py clause backs this rule, if any
```

## Enforcement

Injected text enforces nothing on its own. A rule that must hold needs a veto in
`.loam/hooks/pre-tool.py`; `enforced_by:` records which clause backs a rule, so
"stated" stays distinguishable from "enforced".
