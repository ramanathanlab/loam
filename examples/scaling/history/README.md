# Historical measured recipes

`20i/` retains the original launch script, operational setting extract and hashes of
the best qualified 20-node campaign (job 2979745, 51.06% whole-parent sampled GPU
busy). `256c/` retains the corresponding best observed 256-node campaign (job
2999784, 40.51%, three committed rounds, unresolved claim-provenance findings).
The original launch scripts are inert `.txt` records and reference the old
package name, frozen runtimes and site paths; do not submit them into old workspaces.
The original research-bearing configuration is not published; `settings.yaml`
extracts its operational settings and records the original configuration hash.

`../run-20i.sh` and `../run-256c.sh` stage fresh jobs using those admission/model
settings with the current package. The current MCP/cache/packaging fixes are
not a new GPU benchmark. `results.json` includes all 13 owned recipes/costs and
the 11 cases with numerical science outputs, including failed/partial cases.
`native-runtime-versions.json` records the observed ARM64/CUDA inference
installation; versions alone do not guarantee compatible wheels on another site.
`stage_single_allocation_20261119.py.txt` preserves the old staging source before
private-run dependencies were removed.

Full raw campaign/report directories remain external data. They are not copied
into this Git branch. See the [recipe book](../../../docs/scaling-recipe-book.md)
for definitions, uncertainty, timing changes and cross-machine guidance.
