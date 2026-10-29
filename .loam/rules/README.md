# Campaign rules

Policy text injected into a reasoning tier's context by `campaign_context()` in
`.loam/hooks/hypothesis_orchestration.py`, in the tiers each rule's `scope:`
front matter lists (`validator`, `code-generator`, `reasoner`, or `all`).

A rule declaring `applies: always` reaches every campaign, named or not — it
loads even from a profile with no `campaign:` block. Everything else loads only
when the active profile's `campaign.rules` names it, which is the default for a
file that omits `applies:`.

Which way a rule goes follows from the layer it sits in. A profile should not
have to restate that an experiment must finish before it is reported, so the
invariant layers are always on; everything that differs between campaigns is
configuration. The safe default runs the other way, though: a half-written rule
left in this directory must not reach a campaign it was not written for, because
the wrong decisive experiment is worse than a missing one.

Loam itself is domain-neutral — the loader reads markdown and never inspects it.
The rules are where a campaign's science lives, so they are layered by how far
they generalize. Keep each file in one layer; a rule that mixes layers cannot be
reused by the next campaign without being rewritten.

| Layer | Holds | Changes when | `applies:` | Files |
| --- | --- | --- | --- | --- |
| Integrity | What makes a datum trustworthy at all | ~never | `always` | `evidence-integrity.md` |
| Discipline | How an experiment is run and reported | rarely | `always` | `experiment-discipline.md` |
| Shape | What this kind of campaign has to get right | per campaign shape | named | `design-provenance.md` |
| Protocol | The decisive measurement and its metrics | per domain | named | `binder-cofold.md` |
| Constraint | What is out of bounds for this campaign | per campaign | named | — |

Base rules are injected before the named ones, in name order, so a named rule is
the most specific text a child reads. A profile that names an invariant rule
anyway keeps working — the loader deduplicates rather than sending it twice.

The always-on layers must hold for a campaign in **any** field, testing any
hypothesis. That is stricter than being domain-neutral: it also rules out
assuming the campaign's *shape*. "Paste what was designed, retrieve what was
published" is not invariant, because a campaign testing an existing claim
designs nothing — so it sits in `design-provenance.md` and a design campaign
names it. A rule that says what the experiment *is* never belongs in an
always-on file.

A constraint rule is the lightest thing to write and often all a new campaign
needs: a few lines naming hypotheses not worth exploring, tool categories to
avoid, or a budget ceiling. It needs no protocol and no new concepts — give it
`scope: all` (or the one tier it binds) and name it in the profile.

Integrity and discipline rules must name no tool, no metric, no molecule class,
and no campaign shape. If removing the example would leave the rule saying
nothing, the rule was not invariant.

## Starting a campaign in a new field

A campaign exploring something unrelated to the current ones needs **no new
always-on rule**. The invariant layers already apply; write only what is
specific to it, which is usually a constraint rule and, if there is a standard
decisive measurement, a protocol rule. A campaign with neither still runs
governed.

## Writing a protocol rule

One per domain, named for the measurement rather than the campaign, so two
campaigns in the same domain share it. A protocol rule has to answer four
things, because they are what the discipline layer deliberately leaves open:

1. What the decisive measurement is, concretely enough to write code for.
2. Which numbers it reports, and which one is the headline.
3. Where the procedure comes from — the `ENVIRONMENT SETUP` recipe, or a named
   skill once the code-generator can read skills (see `todo.md`).
4. What the measurement does **not** establish, so a child does not overclaim.

Near-term scope is binder design and enzyme engineering. Binder design has its
rule. **Enzyme engineering does not yet** — writing it needs someone who owns
that campaign to settle what the per-round decisive check actually is (activity
proxy? pocket geometry on a holo fold? an MD stability run?), and inventing one
here would read as authoritative when it is a guess.

## Enforcement

Injected text enforces nothing on its own. A rule that must hold needs a veto in
`.loam/hooks/pre-tool.py`; `enforced_by:` front matter records which clause backs
a rule, so "stated" stays distinguishable from "enforced".
