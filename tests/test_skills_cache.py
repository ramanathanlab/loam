import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def module(name, folder):
    spec = importlib.util.spec_from_file_location(name, HERE / 'fixtures/skills_before_cache.py' if folder == 'baseline' else HERE.parent / 'loam/skills.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


old = module('old_skills', 'baseline')
new = module('new_skills', 'candidate')


@pytest.fixture(autouse=True)
def empty_cache():
    new._cached_frontmatter.cache_clear()
    new._cached_catalog.cache_clear()


@pytest.mark.parametrize('text', [
    'plain skill body', '---\nname: test\ndescription: hello\n---\nBody',
    '---\nname: test\ndescription: [a, b]\n---\nBody',
    '---\nname: [bad YAML\n---', '---\n---\nBody',
    '\ufeff---\nname: bom\n---', '---\n[a, b]\n---',
    '---\nname: "Unicode αβ"\ndescription: |\n  line1\n  line2\n---\n',
])
def test_parser_results_equal(tmp_path, text):
    path = tmp_path / 'SKILL.md'
    path.write_text(text)
    assert new._frontmatter(path) == old._frontmatter(path)
    assert new._frontmatter(path) == old._frontmatter(path)


def test_warm_catalog_avoids_content_reads(tmp_path, monkeypatch):
    path = tmp_path / 'SKILL.md'
    path.write_text('---\nname: sample\ndescription: metadata\n---\nBody')
    skills = {'sample': tmp_path}
    expected = old.catalog(skills)
    assert new.catalog(skills) == expected
    count = 0
    read_text = Path.read_text

    def tracked(self, *args, **kwargs):
        nonlocal count
        count += 1
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', tracked)
    assert all(new.catalog(skills) == expected for _ in range(16))
    assert count == 0
    assert old.catalog(skills) == expected
    assert count == 1


def test_edit_invalidates(tmp_path):
    path = tmp_path / 'SKILL.md'
    path.write_text('---\ndescription: old\n---')
    assert new._frontmatter(path)['description'] == 'old'
    path.write_text('---\ndescription: updated\n---')
    assert new._frontmatter(path) == old._frontmatter(path)
    assert new._frontmatter(path)['description'] == 'updated'


def test_atomic_replacement_with_same_mtime_and_size_invalidates(tmp_path):
    path = tmp_path / 'SKILL.md'
    path.write_text('---\ndescription: old\n---')
    before = path.stat()
    assert new._frontmatter(path)['description'] == 'old'
    replacement = tmp_path / 'replacement'
    replacement.write_text('---\ndescription: new\n---')
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(path)
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert new._frontmatter(path)['description'] == 'new'


def test_symlink_retarget_invalidates(tmp_path):
    a, b, link = [tmp_path / name for name in ['a', 'b', 'SKILL.md']]
    a.write_text('---\ndescription: first\n---')
    b.write_text('---\ndescription: second\n---')
    link.symlink_to(a)
    assert new._frontmatter(link)['description'] == 'first'
    link.unlink()
    link.symlink_to(b)
    assert new._frontmatter(link)['description'] == 'second'


def test_deletion_is_not_hidden_and_recreation_is_fresh(tmp_path):
    path = tmp_path / 'SKILL.md'
    path.write_text('---\ndescription: old\n---')
    new._frontmatter(path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        new._frontmatter(path)
    path.write_text('---\ndescription: recreated\n---')
    assert new._frontmatter(path)['description'] == 'recreated'


def test_mutable_metadata_and_yaml_aliases_remain_independent(tmp_path):
    path = tmp_path / 'SKILL.md'
    path.write_text('---\na: &shared [original]\nb: *shared\n---')
    first = new._frontmatter(path)
    assert first['a'] is first['b']
    first['a'].append('caller mutation')
    second = new._frontmatter(path)
    assert second == old._frontmatter(path)
    assert second['a'] is second['b']
    assert second['a'] is not first['a']


def test_concurrent_warm_callers_do_not_share_mutable_results(tmp_path):
    path = tmp_path / 'SKILL.md'
    path.write_text('---\nvalues: [original]\n---')
    new._frontmatter(path)

    def use(index):
        result = new._frontmatter(path)
        result['values'].append(index)
        return result

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(use, range(96)))
    assert [v['values'] for v in results] == [['original', i] for i in range(96)]
    assert new._frontmatter(path) == {'values': ['original']}


def test_cache_is_bounded_with_many_skill_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(new, '_file_version', lambda p: (0, 0, 1, 2, 3))
    monkeypatch.setattr(new, '_read_frontmatter', lambda p: {'name': p.name})
    for index in range(600):
        assert new._frontmatter(tmp_path / str(index)) == {'name': str(index)}
    assert new._cached_frontmatter.cache_info().currsize == 512


def test_relative_paths_are_scoped_to_current_directory(tmp_path, monkeypatch):
    for directory, label in [('one', 'first'), ('two', 'second')]:
        folder = tmp_path / directory
        folder.mkdir()
        (folder / 'SKILL.md').write_text('---\ndescription: ' + label + '\n---')
        monkeypatch.chdir(folder)
        assert new._frontmatter(Path('SKILL.md'))['description'] == label


def test_continuous_edits_fall_back_without_unbounded_retry(tmp_path, monkeypatch):
    path = tmp_path / 'SKILL.md'
    path.write_text('---\ndescription: actual\n---')
    versions = iter((i,) for i in range(6))
    monkeypatch.setattr(new, '_file_version', lambda p: next(versions))
    assert new._frontmatter(path) == old._frontmatter(path)


def test_public_reader_safety_and_fresh_content_are_unchanged(tmp_path):
    skill = tmp_path / 'skill'
    skill.mkdir()
    path = skill / 'SKILL.md'
    path.write_text('---\nname: sample\ndescription: first\n---\nBody')
    outside = tmp_path / 'outside.txt'
    outside.write_text('outside')
    (skill / 'link').symlink_to(outside)
    skills = {'sample': skill}
    assert new.discover([tmp_path]) == old.discover([tmp_path])
    assert new.catalog(skills) == old.catalog(skills)
    for name, file in [('sample', None), ('sample', '../outside.txt'), ('sample', 'link'), ('missing', None)]:
        assert new.read(skills, name, file) == old.read(skills, name, file)
    path.write_text('---\nname: sample\ndescription: second\n---\nUpdated')
    assert new.read(skills, 'sample') == old.read(skills, 'sample')
    assert new.catalog(skills) == old.catalog(skills)


def test_catalog_cache_preserves_mapping_names_and_order(tmp_path):
    skill = tmp_path / 'skill'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('---\ndescription: same file\n---')
    for mapping in [{'one': skill, 'two': skill}, {'two': skill, 'one': skill}, {'renamed': skill}]:
        assert new.catalog(mapping) == old.catalog(mapping)


def test_catalog_cache_still_checks_for_deleted_files(tmp_path):
    (tmp_path / 'SKILL.md').write_text('---\ndescription: exists\n---')
    assert new.catalog({'a': tmp_path}) == old.catalog({'a': tmp_path})
    (tmp_path / 'SKILL.md').unlink()
    with pytest.raises(FileNotFoundError):
        new.catalog({'a': tmp_path})
