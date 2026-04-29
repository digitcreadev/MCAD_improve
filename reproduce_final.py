from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable or 'python'


def run(cmd: List[str], cwd: Path | None = None) -> None:
    shown = ' '.join(cmd)
    print(f'[MCAD/REPRO] $ {shown}')
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='One-command reproducibility runner for the MCAD prototype.')
    p.add_argument('--profile', choices=['smoke', 'full'], default='smoke', help='smoke = fast validation run, full = article-grade campaign')
    p.add_argument('--run-root', type=str, default='', help='Optional run root under which results-dir is created')
    p.add_argument('--results-dir', type=str, default='', help='Output directory. Defaults to repro_outputs/<profile>_<timestamp>')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--benchmark-repeats', type=int, default=0, help='Override automatic repeat count')
    p.add_argument('--skip-tests', action='store_true')
    return p.parse_args()


def make_results_dir(profile: str, raw: str, run_root: str = '') -> Path:
    if raw.strip():
        out = Path(raw).resolve()
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = Path(run_root).resolve() if run_root.strip() else (ROOT / 'repro_outputs')
        out = base / f'{profile}_{stamp}'
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_manifest(path: Path, args: argparse.Namespace, repeats: int) -> None:
    manifest = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'profile': args.profile,
        'seed': args.seed,
        'benchmark_repeats': repeats,
        'python_executable': PYTHON,
        'python_version': sys.version,
        'platform': platform.platform(),
        'cwd': str(ROOT),
        'configs': [
            'backend/harness/scenarios.yaml',
            'backend/harness/scenarios_adventureworks.yaml',
            'backend/config/objectives.yaml',
        ],
        'main_commands': [
            'python -m py_compile <backend python files>',
            'python -m unittest discover -s backend/tests -v',
            'python reproduce_final.py --profile smoke|full',
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')


def main() -> None:
    args = parse_args()
    out_dir = make_results_dir(args.profile, args.results_dir, args.run_root)
    repeats = args.benchmark_repeats or (5 if args.profile == 'smoke' else 25)

    # 1) Compile critical backend files.
    py_files = []
    for base in ['backend/mcad', 'backend/ckg', 'backend/harness', 'backend/app', 'backend/routers']:
        for p in sorted((ROOT / base).rglob('*.py')):
            py_files.append(str(p.relative_to(ROOT)))
    run([PYTHON, '-m', 'py_compile', *py_files])

    # 2) Run unit tests.
    if not args.skip_tests:
        run([PYTHON, '-m', 'unittest', 'discover', '-s', 'backend/tests', '-v'])

    # 3) Run the full MCAD pipeline in scenarios mode.
    full_results = out_dir / 'full_pipeline'
    cmd = [
        PYTHON,
        'backend/harness/run_full_pipeline.py',
        '--mode', 'scenarios',
        '--results-dir', str(full_results),
        '--config', 'backend/harness/scenarios.yaml',
        '--stop', '5',
        '--with-policy-benchmark',
        '--with-multidataset-benchmark',
        '--with-human-validation-dry-run',
        '--benchmark-repeats', str(repeats),
        '--seed', str(args.seed),
    ]
    run(cmd)

    # 4) Consolidate runtime metrics for p50/p95/p99 reporting.
    run([PYTHON, 'backend/harness/summarize_runtime_latency.py', '--results-dir', str(full_results)])

    # 5) Copy a lightweight entrypoint note.
    quickstart = out_dir / 'HOW_TO_RERUN.txt'
    quickstart.write_text(
        '\n'.join([
            'MCAD reproducibility quickstart',
            '',
            f'1) python reproduce_final.py --profile smoke --results-dir "{out_dir}"',
            '2) Main outputs:',
            f'   - {full_results}',
            f'   - {full_results / "runtime_latency"}',
            '3) For a heavier campaign: python reproduce_final.py --profile full',
        ]),
        encoding='utf-8'
    )

    write_manifest(out_dir / 'repro_manifest.json', args, repeats)
    print(f'[MCAD/REPRO] done -> {out_dir}')


if __name__ == '__main__':
    main()
