# Registered batch-tool workflow

Use this pattern when an HPC MCP service exposes a registry plus per-row batch runner:

1. Use `run_code` to write a small JSON-input script to a shared work path. The script receives the path to one input JSON object as `sys.argv[1]`, writes outputs such as `result.json` in its run directory, and exits nonzero on errors.
2. Register it with `register_tool`, declaring typed `params`, a minimal representative `example`, and an accurate statement of output files. Registration performs (or can perform) a smoke test; record the returned tool version and `tested` state.
3. Write one JSON object per line to an input `.jsonl`; include a stable `id` for each row for traceability.
4. Start `run_batch(tool=<registered name>, inputs_path=<absolute JSONL path>)`. Preserve the returned `batch_id`.
5. Poll `batch_status(batch_id=...)` until `complete: true`; report total/succeeded/failed/unfinished and the results directory. Do not infer completion merely from batch submission.
6. Optionally use `list_tools` afterward to confirm the registered version, use count, and `last_ok_batch`.

Example script shape:

```python
import json, sys
row = json.load(open(sys.argv[1]))
json.dump({"y": row["x"] ** 2}, open("result.json", "w"))
```

The normal `run_batch` contract copies the inputs and script at creation, so later edits do not affect an already-created batch.
