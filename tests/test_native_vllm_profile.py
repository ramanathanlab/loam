"""Serving arguments are a numerical/model contract, independent of Slurm."""
from jupiter_nemotron import server_arguments
import pytest


def request():
    return dict(command=['/installed/bin/vllm'],model_dir='/weights/pinned',
                max_model_len=131072,max_num_seqs=16)


def test_four_gpu_arguments_match_measured_native_profile():
    assert server_arguments(request(),12345)==[
        '/installed/bin/vllm','serve','/weights/pinned','--host','127.0.0.1','--port','12345',
        '--served-model-name','nemotron-3.5-super-vl','--tensor-parallel-size','4',
        '--gpu-memory-utilization','0.85','--max-model-len','131072','--max-num-seqs','16',
        '--language-model-only','--trust-remote-code','--enable-auto-tool-choice',
        '--tool-call-parser','qwen3_coder','--reasoning-parser','nemotron_v3']


def test_other_tensor_layout_changes_only_the_explicit_device_count():
    base=server_arguments(request(),12345)
    result=server_arguments({**request(),'tensor_parallel_size':8},12345)
    base[base.index('--tensor-parallel-size')+1]='8'
    assert result==base


@pytest.mark.parametrize('key,value',[('tensor_parallel_size',0),('tensor_parallel_size',True),
    ('max_num_seqs',0),('max_model_len',-1),('gpu_memory_utilization',1.0)])
def test_invalid_serving_budgets_rejected(key,value):
    with pytest.raises(ValueError):server_arguments({**request(),key:value},12345)
