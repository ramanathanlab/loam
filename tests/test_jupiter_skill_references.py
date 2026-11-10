"""Jupiter references must be packaged within the skill Hermes resolves."""
from pathlib import Path
import re

import pytest

SKILLS = sorted((Path(__file__).resolve().parents[1] / "skills").rglob("*-jupiter/SKILL.md"))


@pytest.mark.parametrize("main", SKILLS, ids=lambda p: p.parent.name)
def test_jupiter_skill_reference_paths_are_self_contained(main):
    root = main.parent.resolve()
    text = main.read_text()
    advertised = re.findall(r'file_path="([^"]+)"', text)
    assert advertised, f"{root.name} must advertise explicit skill_view references"
    for relative in advertised:
        assert ".." not in Path(relative).parts
        target = (root / relative).resolve()
        assert target.is_relative_to(root) and target.is_file(), relative

    documents = [main, *sorted((root / "references").rglob("*.md"))]
    for document in documents:
        for link in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if re.match(r"^[a-z]+:", link) or link.startswith("#"):
                continue
            target = (document.parent / link.split("#", 1)[0]).resolve()
            assert target.is_relative_to(root), (document, link)
            assert target.is_file(), (document, link)
