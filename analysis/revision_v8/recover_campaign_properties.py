#!/usr/bin/env python3
"""Read historical IBM job properties; never submit or modify a hardware job.

Uses the configured IBM Runtime account. Raw properties are exclusive-create
archives, retaining each calibration parameter's timestamp and unit. No account
configuration or credential is written to these output files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
STEMS = ('e01_20261008', 'e02_20261014', 'e03_20261014')


def date(value):
    return datetime.fromisoformat(str(value).replace('Z', '+00:00'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', default=None, help='Optional saved Runtime account name.')
    parser.add_argument('--outdir', type=Path, default=HERE/'calibration_archives')
    args = parser.parse_args()
    from qiskit_ibm_runtime import QiskitRuntimeService

    args.outdir.mkdir(parents=True, exist_ok=True)
    try:
        service = (QiskitRuntimeService(name=args.account) if args.account
                   else QiskitRuntimeService())
    except Exception as exc:
        # SDK exception text can include account details; report only its type.
        print('Runtime account initialization failed:', type(exc).__name__)
        raise SystemExit(2)
    failed = False
    for stem in STEMS:
        base = PAPER/'ibm_variance_campaign/results'
        plan = json.loads((base/(stem+'_plan.json')).read_text())
        counts = json.loads((base/(stem+'_counts.json')).read_text())
        destination = args.outdir/(stem+'_properties.json')
        if destination.exists():
            previous = json.loads(destination.read_text())
            if previous['job_id'] != plan['job_id']:
                raise ValueError('Existing archive belongs to a different job.')
            print(stem, 'already archived; preserved')
            continue
        try:
            job = service.job(plan['job_id'])
            props = job.properties(refresh=True)
            if props is None:
                print(stem, 'historical properties unavailable')
                failed = True
                continue
            backend = job.backend()
            name = backend.name() if callable(backend.name) else backend.name
            if name != plan['backend'] or name != counts['backend']:
                raise ValueError('Job backend differs from local archives.')
            if date(props.last_update_date) != date(counts['calibration_date']):
                print(stem, 'calibration timestamp differs; no archive accepted')
                failed = True
                continue
            # Read-only metrics retain execution timing as a consistency aid.
            metrics = job.metrics()
            payload = dict(
                schema_version=1, epoch=stem, job_id=plan['job_id'], backend=name,
                calibration_date=str(props.last_update_date),
                expected_calibration_date=counts['calibration_date'],
                fetched_utc=datetime.now(timezone.utc).isoformat(),
                sdk_version=version('qiskit-ibm-runtime'),
                source='RuntimeJobV2.properties(refresh=True)',
                execution_timestamps=metrics.get('timestamps'),
                backend_properties=props.to_dict())
            with destination.open('x') as handle:
                json.dump(payload, handle, indent=2, default=str)
                handle.write('\n')
            checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
            destination.with_suffix('.sha256').write_text(checksum+'  '+destination.name+'\n')
            print(stem, 'archived', len(props.qubits), 'qubits,', len(props.gates),
                  'gate records; date matches')
        except Exception as exc:
            print(stem, 'retrieval failed:', type(exc).__name__)
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
