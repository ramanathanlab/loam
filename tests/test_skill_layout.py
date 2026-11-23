"""Check every skill: System-neutral instructions and references for each supported System."""

from pathlib import Path

import pytest

from loam import skills

SKILLS = Path(__file__).resolve().parent.parent / 'skills'


@pytest.mark.parametrize('skill', sorted(path.parent.relative_to(SKILLS)
                                        for path in SKILLS.rglob('SKILL.md')), ids=str)
def test_every_skill_passes_the_layout_check(skill):
    assert skills.layout_problems(SKILLS / skill) == []


def make_skill(root: Path, body: str = 'Read references/<HPC_SYSTEM>.md. Unsupported on any other System.\n',
               systems=('aurora',), references=('aurora',)) -> Path:
    """A minimal skill directory: SKILL.md claiming `systems`, plus a reference file per `references`."""
    skill = root / 'demo'
    (skill / 'references').mkdir(parents=True)
    claim = f'systems: [{", ".join(systems)}]\n' if systems is not None else ''
    (skill / 'SKILL.md').write_text(f'---\nname: demo\ndescription: A demo.\n{claim}---\n\n{body}')
    for system in references:
        (skill / 'references' / f'{system}.md').write_text(f'# demo on {system}\n')
    return skill


def test_a_well_formed_skill_passes(tmp_path):
    assert skills.layout_problems(make_skill(tmp_path)) == []


@pytest.mark.parametrize('path', ['/flare/PROJ/me/software/chai', '/lus/flare/projects/PROJ',
                                  '/e/project1/PROJ/me', '/Users/someone/tools', '/opt/aurora/26.26.0/lib'])
def test_system_specific_or_personal_path_in_skill_md_fails(tmp_path, path):
    problems = skills.layout_problems(make_skill(tmp_path, body=f'Run `{path}/bin/tool`. Unsupported elsewhere.\n'))
    assert any(path.split('/')[1] in p for p in problems), problems


@pytest.mark.parametrize('path', ['/flare/PROJ/me/software/chai', '/lus/flare/projects/PROJ',
                                  '/e/project1/PROJ/me', '/Users/someone/tools'])
def test_personal_path_in_a_bundled_file_fails(tmp_path, path):
    skill = make_skill(tmp_path)
    (skill / 'scripts' / 'aurora').mkdir(parents=True)
    (skill / 'scripts' / 'aurora' / 'run.sh').write_text(f'REPO={path}\n')

    problems = skills.layout_problems(skill)

    assert any('scripts/aurora/run.sh' in p and path.split('/')[1] in p for p in problems), problems


def test_compiled_bytecode_beside_a_script_is_not_scanned(tmp_path):
    skill = make_skill(tmp_path)
    (skill / 'scripts' / '__pycache__').mkdir(parents=True)
    (skill / 'scripts' / '__pycache__' / 'run.cpython-312.pyc').write_bytes(b'\x00/flare/PROJ/me\x00')

    assert skills.layout_problems(skill) == []


def test_a_system_path_in_a_reference_file_passes(tmp_path):
    skill = make_skill(tmp_path)
    (skill / 'references' / 'aurora.md').write_text('export LD_LIBRARY_PATH=/opt/aurora/26.26.0/lib\n')

    assert skills.layout_problems(skill) == []


@pytest.mark.parametrize('line', ['#PBS -A MYPROJ', '#SBATCH --account=myproj', '#SBATCH -A myproj',
                                  'sbatch --account myproj job.sh', 'qsub -A MYPROJ job.pbs', 'account: MYPROJ',
                                  'Submit with `-A MYPROJ`.', '-A MYPROJ', '  account: "MYPROJ"'])
def test_hard_coded_account_in_skill_md_fails(tmp_path, line):
    problems = skills.layout_problems(make_skill(tmp_path, body=f'```\n{line}\n```\nUnsupported elsewhere.\n'))
    assert any('account' in p for p in problems), problems


@pytest.mark.parametrize('line', ['grep -A 3 ERROR run.log', 'ls -A out', 'Take into account: memory grows with L.'])
def test_unrelated_dash_a_or_the_word_account_passes(tmp_path, line):
    body = f'```\n{line}\n```\nUnsupported elsewhere.\n'
    assert skills.layout_problems(make_skill(tmp_path, body=body)) == []


def test_account_taken_from_a_variable_passes(tmp_path):
    body = '```\nqsub -A "$ACCOUNT" job.pbs\n#SBATCH --account=${SLURM_ACCOUNT}\n```\nUnsupported elsewhere.\n'
    assert skills.layout_problems(make_skill(tmp_path, body=body)) == []


def test_a_system_independent_skill_needs_no_reference_files(tmp_path):
    skill = make_skill(tmp_path, body='Make the HTTP calls yourself. This skill does not depend on the System.\n',
                       systems=('all',), references=())

    assert skills.layout_problems(skill) == []


def test_a_single_claim_may_be_written_without_brackets(tmp_path):
    skill = make_skill(tmp_path, systems=('all',), references=())
    (skill / 'SKILL.md').write_text('---\nname: demo\ndescription: A demo.\nsystems: all\n---\n\nAny System.\n')

    assert skills.layout_problems(skill) == []


def test_a_system_independent_skill_may_not_hold_a_personal_path(tmp_path):
    skill = make_skill(tmp_path, body='Run `/flare/PROJ/me/software/tool`. No System needed.\n',
                       systems=('all',), references=())

    assert any('/flare/' in problem for problem in skills.layout_problems(skill))


def test_a_system_independent_skill_with_a_reference_file_fails(tmp_path):
    problems = skills.layout_problems(make_skill(tmp_path, systems=('all',), references=('aurora',)))

    assert any('all' in p and 'aurora' in p for p in problems), problems


def test_naming_all_beside_a_system_fails(tmp_path):
    problems = skills.layout_problems(make_skill(tmp_path, systems=('all', 'aurora'), references=('aurora',)))

    assert any('all' in p for p in problems), problems


def test_claiming_a_system_without_its_reference_file_fails(tmp_path):
    problems = skills.layout_problems(make_skill(tmp_path, systems=('aurora', 'jupiter'), references=('aurora',)))
    assert any('jupiter' in p and 'references/jupiter.md' in p for p in problems), problems


def test_naming_an_undefined_system_fails(tmp_path):
    problems = skills.layout_problems(make_skill(tmp_path, systems=('frontier',), references=('frontier',)))
    assert any('frontier' in p and 'not a defined System' in p for p in problems), problems


def test_a_reference_file_for_a_system_the_skill_does_not_claim_fails(tmp_path):
    problems = skills.layout_problems(make_skill(tmp_path, systems=('aurora',), references=('aurora', 'jupiter')))
    assert any('jupiter' in p for p in problems), problems


def test_a_skill_that_claims_no_system_fails(tmp_path):
    problems = skills.layout_problems(make_skill(tmp_path, systems=None))
    assert any('systems' in p for p in problems), problems


def test_a_skill_unsupported_on_some_system_must_say_so(tmp_path):
    problems = skills.layout_problems(make_skill(tmp_path, body='Read references/<HPC_SYSTEM>.md.\n'))
    assert any('unsupported' in p for p in problems), problems
