#!/usr/bin/env python3
"""Register this checkout's `skills/` tree in a Hermes skill library.

`campaign.skills` names are resolved by Hermes against the skill library of the
home it is running in, not against this repository. A clone therefore ships the
skills without them being reachable: a code-generator told to `skill_view` on
`chai` gets nothing, and because the name is only shape-checked when the profile
loads, that failure lands in a leaf agent mid-round rather than at startup.

This installs them. The default target is the shared home inside the checkout,
but `--hermes-home` (or `$HERMES_HOME`) points it at whichever home a user
actually launches with, which is the case a fresh clone has to cover.

Symlinks are the default so that one tree stays the source of truth and a
`git pull` updates the library with it. `--copy` is for a home that cannot
reference the checkout -- a different filesystem, or a home that has to stay
self-contained after the checkout moves.

Re-running is safe: a link this script owns is replaced, and anything else in
the way is reported and left alone unless `--force` says otherwise.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# A skill is a directory holding SKILL.md; a category is a directory of those.
# The repo has both shapes at its top level (`qm/uma/SKILL.md` and
# `hpc-deployment/SKILL.md`), and Hermes accepts either, so each top-level entry
# is installed under its own name whichever it is.
SKILL_FILE = "SKILL.md"


def _is_installable(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith("."):
        return False
    if (path / SKILL_FILE).is_file():
        return True
    return any(child.joinpath(SKILL_FILE).is_file() for child in path.iterdir() if child.is_dir())


def _default_home(repo: Path) -> Path:
    from_env = os.environ.get("HERMES_HOME", "").strip()
    return Path(from_env).expanduser() if from_env else repo / ".hermes" / ".loam-home"


def _owned_by_repo(link: Path, source_root: Path) -> bool:
    """True when `link` is a symlink already pointing inside our skills tree."""

    if not link.is_symlink():
        return False
    try:
        return source_root.resolve() in link.resolve().parents
    except OSError:
        return False


def _link_target(entry: Path, library: Path, repo: Path) -> str:
    """Relative when the library lives inside the checkout, absolute otherwise.

    A relative link is what keeps the in-repo shared home working after a clone
    or a move. It is wrong for any other home: the two paths may share no useful
    prefix, and on macOS a home under /tmp resolves through /private/tmp, so the
    computed hops land nowhere. An external home gets an absolute target, which
    is accurate for exactly as long as the checkout stays put -- re-run this
    script after moving it, or use --copy.
    """

    library_real = library.resolve()
    if repo == library_real or repo in library_real.parents:
        return os.path.relpath(entry.resolve(), library_real)
    return str(entry.resolve())


def install(repo: Path, home: Path, copy: bool, force: bool) -> int:
    source_root = repo / "skills"
    if not source_root.is_dir():
        print(f"install_skills: no skills tree at {source_root}", file=sys.stderr)
        return 1

    library = home / "skills"
    library.mkdir(parents=True, exist_ok=True)

    entries = sorted(p for p in source_root.iterdir() if _is_installable(p))
    if not entries:
        print(f"install_skills: {source_root} holds no skills", file=sys.stderr)
        return 1

    installed, skipped, blocked = [], [], []
    for entry in entries:
        target = library / entry.name
        if target.exists() or target.is_symlink():
            if _owned_by_repo(target, source_root) or force:
                if target.is_symlink() or target.is_file():
                    target.unlink()
                else:
                    shutil.rmtree(target)
            else:
                blocked.append(entry.name)
                continue
        if copy:
            shutil.copytree(entry, target)
        else:
            target.symlink_to(_link_target(entry, library, repo))
        installed.append(entry.name)

    for name in installed:
        resolved = (library / name)
        if not (resolved / SKILL_FILE).is_file() and not any(
            child.joinpath(SKILL_FILE).is_file() for child in resolved.iterdir() if child.is_dir()
        ):
            print(f"install_skills: {name} installed but holds no {SKILL_FILE}", file=sys.stderr)
            return 1

    verb = "copied" if copy else "linked"
    print(f"install_skills: {verb} {len(installed)} entries into {library}")
    for name in installed:
        print(f"  {name}")
    if skipped:
        print(f"  unchanged: {', '.join(skipped)}")
    if blocked:
        print(
            "install_skills: left in place (not ours; pass --force to replace): "
            + ", ".join(blocked),
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parent.parent),
        help="the Loam checkout holding skills/ (default: this script's checkout)",
    )
    parser.add_argument(
        "--hermes-home",
        default=None,
        help="the Hermes home to install into (default: $HERMES_HOME, else the checkout's shared home)",
    )
    parser.add_argument("--copy", action="store_true", help="copy instead of symlinking")
    parser.add_argument(
        "--force", action="store_true", help="replace an entry this script does not own"
    )
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    home = Path(args.hermes_home).expanduser() if args.hermes_home else _default_home(repo)
    return install(repo, home, args.copy, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
