# loam

Hypothesis-driven computational science campaigns: hypotheses are generated and ranked, validated by running real computations, independently reviewed, and the accepted results steer the next round.

## Memory

**Lesson**:
A reusable account of something that went wrong in a round, and why, kept so that later rounds and later campaigns can avoid repeating it. A lesson is a lead to check, never evidence: a claim may not cite one as its source.
_Avoid_: memory, finding, fact, observation, insight

**Procedural lesson**:
A lesson about carrying out the work: a code-generator run that was blocked or crashed, or a validation that was blocked, together with what blocked it.
_Avoid_: tool lesson, failure log

**Review lesson**:
A lesson about the standard of evidence: a claim the reviewer rejected or could not verify, together with what the reviewer checked.
_Avoid_: rejection, critique

**Lesson memory**:
The store of lessons shared by every campaign, which the orchestrator alone writes to after each round and recalls from into validator and code-generator prompts.
_Avoid_: bank, knowledge base, agent memory

## Deployment

**System**:
An HPC machine as every user sees it (Aurora, Polaris, Jupiter): its scheduler, node shape, accelerator type, and how work is launched on it.
_Avoid_: machine, cluster, platform, site

**Site profile**:
One project's or user's setup on a System: allocation account, filesystem roots, software environments, caches, model-weight locations, and inference endpoints.
_Avoid_: site, site config, user config, environment
