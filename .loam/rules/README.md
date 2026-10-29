# Campaign rules

Policy text injected into a reasoning tier's context by `campaign_context()` in
`.loam/hooks/hypothesis_orchestration.py`. A rule reaches a child only when the
active profile's `campaign.rules` names it, and only in the tiers its `scope:`
front matter lists (`validator`, `code-generator`, `reasoner`, or `all`).

Loam itself is domain-neutral — the loader reads markdown and never inspects it.
The rules are where a campaign's science lives, so they are layered by how far
they generalize. Keep each file in one layer; a rule that mixes layers cannot be
reused by the next campaign without being rewritten.

| Layer | Holds | Changes when | Files |
| --- | --- | --- | --- |
| Integrity | What makes a datum trustworthy at all | ~never | `design-integrity.md` |
| Discipline | How an experiment is run and reported | rarely | `experiment-discipline.md` |
| Assay | The decisive measurement and its metrics | per domain | `binder-cofold.md` |

Integrity and discipline rules must not name a tool or a metric, and must stay
true if the campaign switched domains tomorrow. They may use a domain example to
make a principle concrete — a named tool or number is what moves a rule down to
the assay layer, not the illustration.

## Writing an assay rule

One per domain, named for the measurement rather than the campaign, so two
campaigns in the same domain share it. An assay rule has to answer four things,
because they are what the discipline layer deliberately leaves open:

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
