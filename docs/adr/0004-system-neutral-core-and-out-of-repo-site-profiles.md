# Core and skills are System-neutral; Site profiles live outside the repo

loam runs on more than one System (Aurora with PBS and Intel XPUs; Jupiter with Slurm and CUDA GH200s; Polaris), and different people run it on each System with different allocations and filesystem roots. So everything that differs between Systems (scheduler, node shape, accelerator-pinning variable, launcher) is written once per System in `loam.systems` and resolved into explicit deployment environment variables; nothing else in `loam/`, nor the MCP server or cluster agent, names a scheduler or vendor or branches on which System it is. Everything that differs between people on the same System (allocation account, filesystem and software roots, caches, model weights, inference endpoints) is a Site profile kept outside the repo and named from the campaign config. The repo ships only redacted example profiles. Each skill keeps one System-neutral `SKILL.md`, plus one reference file per System that the agent picks by `HPC_SYSTEM`, and refers to paths only through Site profile variables such as `$LOAM_SOFTWARE_ROOT`.

## Considered Options

- **Retarget the repo to whichever System its current user runs on** (what the `moeen` branch did for Jupiter, and what `develop` had done for Aurora). Rejected: each retarget deletes the other System's recipes and swaps one person's paths for another's.
- **Per-System branches.** Rejected: features such as history, rate limits and synthesis would have to be ported to every branch.
- **Site settings committed in the repo, one file per user.** Rejected: allocation names and personal paths end up in a shared repository, and every new user means a PR.

## Consequences

- Adding a System means writing one System implementation plus one reference file per skill. Skills without a reference file for a System are unsupported there, not silently wrong.
- Code that uses the self-hosted Nemotron service is an optional backend installed through a `jupiter` extra, not a core dependency.
