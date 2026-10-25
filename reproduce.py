"""Rebuild numerical checks, summaries, and figures from saved synthetic data.

No model download, network request, API call, or GPU use occurs here.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*arguments, quiet=False):
    result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT,
                            text=True, capture_output=quiet, check=True)
    if quiet:
        return result.stdout


def main():
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    archive = ROOT / 'evidence.zip'
    if digest(archive) != provenance['archive_sha256']:
        raise RuntimeError('Evidence archive checksum does not match provenance.json')
    for relative, expected in provenance['analysis_source_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Analysis source changed: {relative}')
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != set(provenance['archive_members_sha256']):
            raise RuntimeError('Archive member list does not match provenance.json')
        for item in z.infolist():
            destination = (data / item.filename).resolve()
            if not destination.is_relative_to(data.resolve()) or item.is_dir():
                raise RuntimeError(f'Unexpected archive path: {item.filename}')
            payload = z.read(item)
            if hashlib.sha256(payload).hexdigest() != provenance['archive_members_sha256'][item.filename]:
                raise RuntimeError(f'Archive member checksum failed: {item.filename}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
    outputs = ROOT / 'outputs'
    outputs.mkdir(exist_ok=True)
    print('Checksums verified. Testing the market evaluator...', flush=True)
    test_log = run(ROOT / 'src/test_market.py', quiet=True)
    (outputs / 'market_test.log').write_text(test_log)
    summaries = {}
    for name in ['clef', 'qwen']:
        print(f'Recomputing {name} results from raw records...', flush=True)
        log = run(ROOT / 'src/analyze.py', '--run', data / name / 'final', quiet=True)
        (outputs / f'{name}_audit.log').write_text(log)
        final = data / name / 'final'
        validation = json.loads((final / 'validation.json').read_text())
        if not validation['integrity_pass'] or not validation['complete']:
            raise RuntimeError(f'{name} did not pass the full data audit')
        summary = json.loads((final / 'summary.json').read_text())
        original = provenance['models'][name]['original_primary_contrast']
        actual = summary['primary_live_minus_replay_G']
        for field in ['estimate', 'replay_hierarchical_donor_ci95']:
            import numpy as np
            if not np.allclose(actual[field], original[field], rtol=1e-10, atol=1e-12):
                raise RuntimeError(f'{name}: primary result changed: {field}')
        for filename in ['summary.json', 'validation.json']:
            shutil.copyfile(final / filename, outputs / f'{name}_{filename}')
        summaries[name] = {'complete': True, 'record_count':sum(x['actual_records'] for x in validation['stages'].values()),
                           'model_evaluations':validation['record_sources']['model'],
                           'primary_live_minus_replay_G':actual}
    print('Rebuilding the shared-horizon comparison and all five figures...', flush=True)
    run(ROOT / 'src/compare.py', '--clef', data / 'clef/final', '--qwen', data / 'qwen/final',
        '--out', outputs / 'comparison.json', quiet=True)
    run(ROOT / 'src/plot_paper.py', quiet=True)
    expected_figures = ['outcomes', 'history', 'interventions', 'diagnostics', 'controls']
    for name in expected_figures:
        if not (outputs / 'figures' / f'{name}.pdf').is_file():
            raise RuntimeError(f'Figure missing: {name}')
    report = {'status':'complete','models':summaries,'figure_pdfs':[f'outputs/figures/{n}.pdf' for n in expected_figures],
              'note':'Saved-data reproduction only; no new model inference.'}
    (outputs / 'reproduction_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
