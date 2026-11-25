#!/usr/bin/env python3
"""Regenerate the geographic block figure with terminology matching its scope."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def corrected_source(body):
    return (body.replace('PEF base (drone site)', 'candidate site (PEF position)')
            .replace('waypoint (synthetic sensor)', 'synthetic target')
            .replace('(a) bases, waypoints and blocks', '(a) sites, targets and blocks'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=HERE/'generated')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = ROOT/'scripts/make_block_figure.py'
    sys.path.insert(0, str(path.parent))
    namespace = dict(__file__=str(path), __name__='corrected_block_figure')
    exec(compile(corrected_source(path.read_text()), str(path), 'exec'), namespace)
    namespace['FIGURES_DIR'] = args.out_dir
    namespace['main']()


if __name__ == '__main__':
    main()
