"""Dispatch a declared model worker to an explicit environment, without a shell.

Installs and downloads are separate explicit setup operations. This launcher
only runs the bounded worker selected by the saved execution configuration.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(os.environ.get('EDP_WORKSPACE', Path.cwd())).expanduser().resolve()


def wsl_path(value: str) -> str:
    text = value.replace('\\', '/')
    if len(text) >= 3 and text[1:3] == ':/':
        return '/mnt/'+text[0].lower()+text[2:]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', choices=['wsl', 'native'], required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--distribution', default='Ubuntu-22.04')
    parser.add_argument('--worker', choices=['depth_worker', 'm5_worker'], required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    worker_args = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if not worker_args or any(arg in {'-c', '--eval'} for arg in worker_args):
        parser.error('Supply explicit worker arguments; inline evaluation is unsupported')
    module = 'pipeline.'+args.worker
    if args.runtime == 'wsl':
        if os.name != 'nt':
            parser.error('The WSL dispatcher is launched from the Windows host')
        if not args.python.startswith('/'):
            parser.error('WSL Python must be an absolute Linux path')
        command = ['wsl.exe', '--distribution', args.distribution, '--cd', wsl_path(str(ROOT)),
                   '--exec', args.python, '-u', '-B', '-m', module, *[wsl_path(arg) for arg in worker_args]]
    else:
        python = Path(args.python)
        if not python.is_absolute() or not python.is_file():
            parser.error('Native Python must be an existing absolute interpreter path')
        command = [str(python), '-u', '-B', '-m', module, *worker_args]
    print(f'Runtime={args.runtime}; worker={module}; child approval effect=none', flush=True)
    result = subprocess.run(command, cwd=ROOT, shell=False, check=False)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
