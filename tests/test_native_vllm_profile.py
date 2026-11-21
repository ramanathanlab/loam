"""Serving arguments are a numerical/model contract, independent of the scheduler."""
import json
from pathlib import Path

import pytest

pytest.importorskip('aiohttp', reason='needs the nemotron extra')

from loam.nemotron import server_arguments  # noqa: E402


def profile():
    return dict(model='nemotron-3.5-super-vl', checkpoint='nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16',
                checkpoint_revision='f' * 40, model_dir='/weights/pinned', command=['/installed/bin/vllm'],
                max_model_len=131072, max_num_seqs=16)


def test_four_gpu_arguments_match_measured_native_profile():
    assert server_arguments(profile(), 12345) == [
        '/installed/bin/vllm', 'serve', '/weights/pinned', '--host', '127.0.0.1', '--port', '12345',
        '--served-model-name', 'nemotron-3.5-super-vl', '--tensor-parallel-size', '4',
        '--gpu-memory-utilization', '0.85', '--max-model-len', '131072', '--max-num-seqs', '16',
        '--language-model-only', '--trust-remote-code', '--enable-auto-tool-choice',
        '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'nemotron_v3']


def test_other_tensor_layout_changes_only_the_explicit_device_count():
    base = server_arguments(profile(), 12345)
    result = server_arguments({**profile(), 'tensor_parallel_size': 8}, 12345)
    base[base.index('--tensor-parallel-size') + 1] = '8'
    assert result == base


def test_served_model_name_comes_from_the_profile():
    arguments = server_arguments({**profile(), 'model': 'other-name'}, 1)
    assert arguments[arguments.index('--served-model-name') + 1] == 'other-name'


@pytest.mark.parametrize('key,value', [('tensor_parallel_size', 0), ('tensor_parallel_size', True),
                                       ('max_num_seqs', 0), ('max_model_len', -1), ('gpu_memory_utilization', 1.0)])
def test_invalid_serving_budgets_rejected(key, value):
    with pytest.raises(ValueError):
        server_arguments({**profile(), key: value}, 12345)


def test_the_example_profile_is_the_measured_one():
    path = Path(__file__).parents[1] / 'examples/nemotron/serving-profile.json'
    arguments = server_arguments(json.loads(path.read_text()), 1)
    assert arguments[arguments.index('--max-num-seqs') + 1] == '16'
