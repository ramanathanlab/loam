# Lesson memory is advice, never evidence: only the orchestrator writes it, and no agent has a memory tool

loam's results are trusted because every accepted claim traces to a run this hypothesis owns, checked by an independent reviewer and reconciled deterministically. Lesson memory must not open a side door around that. So lessons are leads, never evidence: a claim may not cite one, and prompts say so. Only the orchestrator writes lessons, from a finished round's record after it is on disk, and only the orchestrator recalls them, injecting them into validator and code-generator prompts. No agent has a memory tool, and the reviewer and reasoner never see lessons.

## Considered Options

- **A `recall_memory` tool for agents.** Rejected for v1: agents would choose when and what to look up, which loam otherwise never lets them do outside their own hypothesis's workspace. Injection keeps reads deterministic and the toolsets unchanged.
- **Agents retaining their own notes.** Rejected: memory would fill with an agent's speculation rather than what a round actually established.
- **Lessons for the reviewer.** Rejected: the reviewer's verdicts must stay independent of what earlier work concluded.
- **Storing accepted findings, not just lessons.** Deferred: a scientific finding from one target can mislead a campaign on another, while a lesson about tools or standards of evidence carries over safely.

## Consequences

- A lesson can only help an agent design or carry out work it then does itself; anything it relies on must be re-derived in a run it owns.
- Adding a memory tool or findings later is a change to this decision, not an extension of it.
