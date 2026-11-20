#!/usr/bin/env bash
# Start a newly staged trial, reusing the established internal Jupiter services.
set -eo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/scripts/jupiter_env.sh"
source "$LOAM_SOFTWARE_ROOT/redis/env.sh"
export ACADEMY_EXCHANGE_TYPE=redis
exec "$repo/.venv/bin/python" - "$repo" "$1" <<'PY'
import json,subprocess,sys,time
from pathlib import Path
repo=Path(sys.argv[1]);trial=Path(sys.argv[2]).resolve()
assert trial.is_relative_to(repo/'runs')
assert (trial/'campaign.yaml').is_file()
for name in ('server-launch.json','supervisor-launch.json','controller-watch.json'):
    if (trial/name).exists():raise RuntimeError('Prior launch receipt exists; inspect it before restarting')
progress=json.loads((trial/'scale-progress.json').read_text());root=Path(progress['inference_pool_experiment']);owned=json.loads((root/'experiment.json').read_text())['owned_job_ids']
queue=subprocess.run(['squeue','-h','-j',','.join(owned),'-o','%i %D'],capture_output=True,text=True,check=True).stdout
reserved=sum(int(line.split()[1]) for line in queue.splitlines())
assert reserved+progress['scientific_allocation_nodes']<=progress['target_total_nodes']<=20,'Prior owned allocations must fully leave Slurm before the next science allocation'
for script,args,log,receipt in [
    ('hold_science_server.py',[trial/'science-server.yaml'],'science-server.log','server-launch.json'),
    ('watch_inference_balance_trial.py',[trial],'supervisor.log','supervisor-launch.json')]:
    with (trial/log).open('ab',buffering=0) as output:
        p=subprocess.Popen([str(repo/'.venv/bin/python'),'-u',str(repo/'scripts'/script),*map(str,args)],cwd=repo,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    data={'pid':p.pid,'started_at':time.time(),'script':script,'exchange_type':'redis'}
    (trial/receipt).write_text(json.dumps(data)+'\n');print(json.dumps(data))
PY
