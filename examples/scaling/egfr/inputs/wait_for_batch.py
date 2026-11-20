"""Read-only bounded wait for actual terminal batch rows; no fixed idle delay."""
import json,time
from pathlib import Path

def wait_batch(batch_dir,max_seconds=600,interval=15):
    root=Path(batch_dir).resolve(strict=True);metadata=json.loads((root/'batch.json').read_text());total=int(metadata['total']);assert total>0
    started=time.monotonic();interval=max(1,min(30,float(interval)));budget=max(0,min(600,float(max_seconds)))
    while True:
        finished=failed=0
        for folder in root.glob('run_*'):
            if not (folder/'done').exists():continue
            try:status=json.loads((folder/'status.json').read_text())
            except (OSError,ValueError):continue
            finished+=1;failed+=status.get('returncode')!=0 or bool(status.get('missing_outputs'))
        assert finished<=total,'More terminal row folders than the batch declares'
        elapsed=time.monotonic()-started
        if finished==total or elapsed>=budget:
            return {'batch_id':metadata['batch_id'],'dir':str(root),'total':total,'terminal':finished,'failed':failed,'unfinished':total-finished,'complete':finished==total,'wait_seconds':elapsed,'next_action':'Call MCP batch_status to attest completion and inspect actual outputs.'}
        time.sleep(min(interval,max(0,budget-elapsed)))
if __name__=='__main__':
    import sys
    print(json.dumps(wait_batch(sys.argv[1])))
