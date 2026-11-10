#!/usr/bin/env python3
"""Package each manual smoke recipe inside its own Hermes-readable skill."""

import ast
import hashlib
import textwrap
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
source = repo / "ports/jupiter/smoke/case.py"
text = source.read_text()
lines = text.splitlines(keepends=True)
tree = ast.parse(text)
branch = next(n for n in tree.body if isinstance(n, ast.If))
prefix = "".join(lines[: branch.lineno - 1])
footer = "".join(lines[branch.end_lineno :])
cases = {}
while isinstance(branch, ast.If):
    test = branch.test
    assert isinstance(test, ast.Compare)
    value = ast.literal_eval(test.comparators[0])
    names = [value] if isinstance(value, str) else value
    body = textwrap.dedent("".join(lines[branch.body[0].lineno - 1 : branch.body[-1].end_lineno]))
    for name in names:
        cases[name] = body
    branch = branch.orelse[0]

count = 0
for main in sorted((repo / "skills").rglob("*-jupiter/SKILL.md")):
    name = main.parent.name.removesuffix("-jupiter")
    recipe = prefix.replace('p.add_argument("--case", required=True)', f'p.add_argument("--case", default={name!r})')
    recipe += cases[name] + footer
    recipe = (
        f"# Generated from ports/jupiter/smoke/case.py; SHA256 {hashlib.sha256(source.read_bytes()).hexdigest()}\n"
        + recipe
    )
    (main.parent / "references/smoke-workload.py").write_text(recipe)
    link = (
        f"\n[Exact smoke recipe](references/smoke-workload.py): "
        f'`skill_view(name="{main.parent.name}", file_path="references/smoke-workload.py")`. '
        "This is a self-contained snapshot of this tool's manual recipe, with public-input provenance and output assertions. "
        "Run it through MCP with this tool's own Python, `--out <fresh-workdir> --software <software-root>`; "
        "the suite additionally checks the login-side Argo answer for rag-query.\n"
    )
    if 'file_path="references/smoke-workload.py"' not in main.read_text():
        main.write_text(main.read_text() + link)
    count += 1
print(f"Packaged {count} self-contained smoke recipes")
