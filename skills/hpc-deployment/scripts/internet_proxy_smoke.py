"""Verify inherited proxy settings with real compute-node database requests."""
import hashlib
import json
import os
import socket
import subprocess
import urllib.request
from urllib.parse import urlsplit

import requests


def main():
    proxy = os.environ.get('https_proxy', '')
    assert proxy, 'Worker did not inherit https_proxy'
    assert urllib.request.proxy_bypass('127.0.0.1'), 'Local MCP/Argo must bypass proxy'
    report = {'hostname': socket.gethostname(), 'job': os.environ.get('SLURM_JOB_ID'),
              'proxy_host': urlsplit(proxy).hostname, 'proxy_port': urlsplit(proxy).port}
    url = 'https://rest.uniprot.org/uniprotkb/P00533.json'
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=8):
            report['direct_internet'] = 'available'
    except Exception as exc:
        report['direct_internet'] = type(exc).__name__
    with urllib.request.urlopen(url, timeout=30) as response:
        target = json.load(response)
        report['urllib_status'] = response.status
    assert target['primaryAccession'] == 'P00533'
    sequence = target['sequence']['value']
    report.update(accession='P00533', sequence_length=len(sequence),
                  sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest())
    response = requests.get('https://data.rcsb.org/rest/v1/core/entry/1IVO', timeout=30)
    response.raise_for_status()
    assert response.json()['rcsb_id'] == '1IVO'
    report['requests_rcsb_status'] = response.status_code
    result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--max-time', '30',
                             'https://rest.uniprot.org/uniprotkb/P00533.fasta'],
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, 'curl fetch failed'
    assert result.stdout.startswith('>sp|P00533|')
    assert ''.join(result.stdout.splitlines()[1:]) == sequence
    report.update(curl_uniprot_ok=True, local_services_bypass=True, status='passed')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
