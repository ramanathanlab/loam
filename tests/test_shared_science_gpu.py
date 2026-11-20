"""Bounded sharing keeps physical GPU identity and campaign resume ownership."""
import importlib
import json
from pathlib import Path

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'skills/hpc-deployment/scripts'


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module('hpc_agent')


def test_el_slots_are_distinct_from_physical_devices(tmp_path,monkeypatch):
    agent=module(monkeypatch)
    shared=agent.HpcClusterAgent(str(tmp_path),'checkpoint',12,'jupiter',science_workers_per_gpu=2)
    assert shared.system_config.ngpus==8
    assert shared.system_config.gpus==list(range(8))
    assert shared.launcher_config.gpu_selector=='LOAM_EL_GPU_SLOTS'
    assert '--gpus-per-node=4' in shared.launcher_config.mpi_config.extra_launcher_flags
    default=agent.HpcClusterAgent(str(tmp_path),'checkpoint',12,'jupiter')
    assert default.system_config.ngpus==4
    assert default.launcher_config.gpu_selector=='CUDA_VISIBLE_DEVICES'


@pytest.mark.parametrize('slot',range(8))
def test_real_row_child_inherits_one_physical_gpu(tmp_path,monkeypatch,slot):
    agent=module(monkeypatch)
    monkeypatch.setenv('LOAM_EL_GPU_SLOTS',str(slot))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2,3')
    script=tmp_path/'probe.py'
    script.write_text('import os,json\nfrom pathlib import Path\nPath("result.json").write_text(json.dumps({"gpu":os.environ["CUDA_VISIBLE_DEVICES"]}))\n')
    folder=tmp_path/'row'
    status=agent._run_one(str(script),{'seed':1},str(folder),10,expects=['result.json'])
    assert status['returncode']==0
    assert status['gpu']==str(slot%4)
    assert status['logical_gpu_slot']==str(slot)
    assert json.loads((folder/'result.json').read_text())['gpu']==str(slot%4)


def test_batch_resume_cannot_change_sharing_policy(tmp_path,monkeypatch):
    agent=module(monkeypatch)
    monkeypatch.setattr(agent,'_MIN_BATCH_ROWS',1)
    monkeypatch.setattr(agent,'_MIN_BATCH_DISTINCT',1)
    script=tmp_path/'work.py';script.write_text('pass\n')
    inputs=tmp_path/'inputs.jsonl';inputs.write_text('{"id":"one","seed":1}\n')
    args=(str(script),str(inputs),None,{'hypothesis_id':'probe'},900,1,None,str(tmp_path),[str(tmp_path)])
    entry,_,_=agent._prepare_batch(*args,science_workers_per_gpu=2)
    assert entry['science_workers_per_gpu']==2
    resume=(None,None,entry['batch_id'],{'hypothesis_id':'probe'},900,1,None,str(tmp_path),[str(tmp_path)])
    agent._prepare_batch(*resume,science_workers_per_gpu=2)
    with pytest.raises(ValueError,match='different GPU sharing policy'):
        agent._prepare_batch(*resume,science_workers_per_gpu=1)
    with pytest.raises(ValueError,match='one-device scientific rows'):
        agent._prepare_batch(str(script),str(inputs),None,{'hypothesis_id':'probe'},900,2,None,str(tmp_path),[str(tmp_path)],science_workers_per_gpu=2)


def test_staging_selects_isolated_runtime_and_physical_node_count(tmp_path,monkeypatch):
    from loam.config import load
    monkeypatch.syspath_prepend(str(SCRIPTS.parents[2]/'scripts'))
    import stage_single_allocation as staging
    from check_single_allocation import check
    job=staging.stage(tmp_path,20,8,'test-sharing','campaign','06:00:00','',24,top_k=96,
        initial_hypotheses=192,new_per_round=48,tournament_workers=16,science_workers_per_gpu=2)
    cfg=load(job/'campaign/campaign.yaml');proof=check(job)
    assert cfg.execution.hpc.science_workers_per_gpu==2
    assert cfg.execution.hpc.environment()['HPC_SCIENCE_WORKERS_PER_GPU']=='2'
    assert cfg.execution.hpc.server_script==job/'infra/control-runtime/skills/hpc-deployment/scripts/mcp_server.py'
    assert proof['science_GPU_slots']==48


def test_zero_gpu_metadata_clears_inherited_shared_slot(tmp_path,monkeypatch):
    import asyncio
    import concurrent.futures
    from types import SimpleNamespace
    agent=module(monkeypatch)
    monkeypatch.setattr(agent,'_MIN_BATCH_ROWS',1)
    monkeypatch.setattr(agent,'_MIN_BATCH_DISTINCT',1)
    script=tmp_path/'work.py';script.write_text('pass\n')
    inputs=tmp_path/'inputs.jsonl';inputs.write_text('{"id":"one","seed":1}\n')
    tasks=[]
    class Client:
        def submit_batch(self,submitted):
            tasks.extend(submitted)
            return [concurrent.futures.Future() for _ in submitted]
    client=SimpleNamespace(base_dir=str(tmp_path),inspect_roots=[str(tmp_path)],
        science_workers_per_gpu=2,_client=Client(),_batches={})
    asyncio.run(agent.HpcClientAgent.run_batch(client,script_path=str(script),inputs_path=str(inputs),gpus=0))
    assert tasks[0].get_resource_requirements().resources[0].gpu_count==0
    assert tasks[0].env=={'LOAM_EL_GPU_SLOTS':'','CUDA_VISIBLE_DEVICES':''}


def test_eight_gpu_adapter_maps_slots_to_the_configured_device_count(tmp_path,monkeypatch):
    agent=module(monkeypatch)
    monkeypatch.setenv('HPC_GPUS_PER_NODE','8')
    monkeypatch.setenv('HPC_CPUS_PER_NODE','96')
    monkeypatch.setenv('HPC_SCIENCE_WORKERS_PER_GPU','2')
    cluster=agent.HpcClusterAgent(str(tmp_path),'checkpoint',2,'jupiter',science_workers_per_gpu=2)
    assert cluster.system_config.ngpus==16 and cluster.system_config.ncpus==96
    assert '--gpus-per-node=8' in cluster.launcher_config.mpi_config.extra_launcher_flags
    monkeypatch.setenv('LOAM_EL_GPU_SLOTS','10')
    script=tmp_path/'probe.py'
    script.write_text('import os;open("result.txt","w").write(os.environ["CUDA_VISIBLE_DEVICES"])')
    result=agent._run_one(str(script),{},str(tmp_path/'row'),10,expects=['result.txt'])
    assert result['returncode']==0
    assert (tmp_path/'row/result.txt').read_text()=='2'
