# The embedding model is fixed per memory instance, and checked before the server starts

Lesson memory stores vectors whose size is set by the first embedding model used, and Hindsight (1.2.3) silently ALTERs the database's embedding column to match whatever model it is started with. Started with a different model, it would resize the column holding months of lessons from the old one. So loam records which embedding model each loam-owned database instance was created with (only once the server first starts healthy, so a typo'd model cannot lock it) and refuses to start memory when the config names a different one, checking *before* the server process launches. Switching models means a new `memory.instance`, which starts with an empty bank.

## Considered Options

- **Document it and trust the operator.** Rejected: the failure is silent and destroys accumulated lessons, the one thing this memory exists to keep.
- **Re-embed on change.** Rejected for v1: it needs every lesson's original text and a migration path Hindsight doesn't offer; a new instance is the cheap, safe switch.

## Consequences

- With `memory.url`, loam does not own the database, so the guard does not apply; keeping the model consistent is the operator's job.
- `text-embedding-3-large` (3072 dimensions) is out regardless: pgvector cannot HNSW-index vectors over 2000 dimensions.
