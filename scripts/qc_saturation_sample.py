#!/usr/bin/env python3
"""Independently check the earliest terminal full-context example per condition.

This is a bounded runtime diagnostic, not a replacement for the full postrun
audit or any campaign/scale-up gate. It never changes scientific artifacts.
"""
import argparse
import json
import re
import time
from pathlib import Path

from audit_scale20_campaign import (read,sha,input_protocol_issues,verify_model,
                                   check_minimized_geometry,amber_partition)


def check(campaign,output):
    canonical=read(campaign/'inputs/intact-egfr/P00533.json')['sequence']['value'][24:646]
    options={}
    for path in campaign.glob('batches/*/run_*/followup-summary.json'):
        folder=path.parent
        if not (folder/'done').exists() or not (folder/'status.json').exists():continue
        status=read(folder/'status.json')
        if status.get('returncode') or status.get('missing_outputs'):continue
        summary=read(path);arm=summary['condition']
        if arm not in ('dual','one_region','ridge'):continue
        lease=read(folder/'lease.json')
        end=status.get('process_finished_at') or lease['started_at']+status['seconds']
        if arm not in options or end<options[arm][0]:options[arm]=(end,path)
    checks=[];errors=[];verified=[]
    for arm,(_,path) in sorted(options.items()):
        try:
            folder=path.parent;summary=read(path);issues=input_protocol_issues(campaign,folder,'scale20_intact_egfr_followup')
            assert not issues,issues
            assert summary['target_residues']==622 and summary['target_disulfides']==25
            energy=read(folder/'mmgbsa/report.json')
            assert energy['mmgbsa_completed'] and energy['minimization_completed']
            topology=amber_partition(folder/'mmgbsa/complex.prmtop',622)
            assert topology['residues']==682 and not topology['target_binder_covalent_bonds']
            assert topology['disulfide_residue_pairs']==sorted(sorted(pair) for pair in energy['disulfides_from_uniprot'])
            assert len(topology['disulfide_residue_pairs'])==25
            actual=float(re.search(r'^DELTA TOTAL\s+([-+\d.eE]+)',(folder/'mmgbsa/FINAL_RESULTS_MMPBSA.dat').read_text(),re.M)[1])
            assert abs(actual-summary['static_MMGBSA_kcal_mol'])<1e-5
            models=read(folder/'validation-report.json')['predictions'];assert len(models)==5
            checked=[]
            for item in models:
                coordinate=Path(item['cif']).resolve(strict=True)
                assert coordinate.is_relative_to(folder.resolve())
                result=verify_model(coordinate,canonical,summary['binder_sequence'])
                checked.append(result);verified.append(result)
            assert abs(max(m['scores']['iptm'] for m in checked)-summary['intact_iptm_best'])<1e-6
            assert sum(m['dual_region_contact'] for m in checked)==summary['intact_models_with_dual_region_contact']
            check_minimized_geometry(summary,canonical)
            checks.append({'condition':arm,'summary':str(path),'summary_sha256':sha(path),
                'target_residues':622,'binder_residues':60,'target_disulfides_independently_counted':25,
                'target_binder_covalent_bonds':topology['target_binder_covalent_bonds'],
                'full_context_coordinate_files_verified':5,'full_context_best_iptm':summary['intact_iptm_best'],
                'poses_contacting_both_regions':summary['intact_models_with_dual_region_contact'],
                'minimized_geometry_independently_checked':True,'static_energy_matches_Amber_output':True})
        except Exception as exc:
            errors.append({'condition':arm,'summary':str(path),'type':type(exc).__name__,'error':str(exc)})
    result={'checked_at':time.time(),'campaign':str(campaign),'passed':bool(checks) and not errors,
        'scope':'Earliest terminal full-context example per available condition, not every output and not proof that campaign or saturation gates passed. Computational confidence/geometry/energy only; no experimental binding measurement.',
        'conditions_available':sorted(options),'checks':checks,'errors':errors,
        'verified_coordinate_files':len(verified),'coordinate_evidence':verified,
        'verifier_sha256':sha(Path(__file__)),'independent_numeric_helpers_sha256':sha(Path(__file__).with_name('audit_scale20_campaign.py'))}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='coordinate_evidence'},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('campaign',type=lambda p:Path(p).resolve());parser.add_argument('output',type=lambda p:Path(p).resolve());args=parser.parse_args();check(args.campaign,args.output)
