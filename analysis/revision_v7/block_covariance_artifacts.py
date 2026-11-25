"""Figures, tables and manuscript numbers from the exploratory block profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {"defaults": "defaults", "matched": "XY4 + twirling", "E01": "E01", "E02": "E02", "E03": "E03"}
COLORS = {"defaults": "#4477AA", "matched": "#CC6677", "E01": "#228833", "E02": "#AA3377", "E03": "#4477AA"}


def interval(summary, index=None):
    value, bounds = summary['estimate'], summary['ci95']
    if index is not None:
        value, bounds = value[index], [b[index] for b in bounds]
    return f"${value:.3f}$ $[{bounds[0]:.3f},{bounds[1]:.3f}]$"


def write(result, output):
    output.mkdir(parents=True, exist_ok=True)
    contexts, ref = result['contexts'], result['reference']
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05), sharey=True)
    for ax, group, title in zip(axes, (contexts[:2], contexts[2:]),
                               ('(a) Historical executions', '(b) Fixed-circuit jobs')):
        offsets = np.linspace(-.16, .16, len(group))
        for offset, context in zip(offsets, group):
            stats, key = context['statistics'], context['context']
            values, ci = np.asarray(stats['retention']['estimate']), np.asarray(stats['retention']['ci95'])
            ax.errorbar(np.arange(6)+offset, values, yerr=[values-ci[0], ci[1]-values],
                        fmt='o', ms=3, lw=.8, capsize=2, color=COLORS[key], label=LABELS[key])
            ax.axhline(stats['global_retention']['estimate'], color=COLORS[key], ls='--', lw=.7, alpha=.8)
        ax.set(title=title, xticks=np.arange(6), xticklabels=ref['block_sizes'],
               xlabel='variables in logical block')
        ax.grid(axis='y', color='#dddddd', lw=.5)
        ax.set_ylim(-.05, 1.02)
        ax.legend(loc='upper center', ncol=1, frameon=False, fontsize=7)
    axes[0].set_ylabel(r'block weighted-covariance ratio $C_\alpha$')
    fig.text(.5, .015, 'Bars: pointwise conditional 95% intervals; dashed lines: global ratios',
             ha='center', fontsize=7)
    fig.tight_layout(rect=(0, .065, 1, 1), w_pad=1.7)
    fig.savefig(output/'fig_block_covariance.pdf', bbox_inches='tight')
    plt.close(fig)

    lines = [r'\begin{table*}[t]',
             r'\caption{Exploratory block weighted-covariance profiles. Brackets are pointwise conditional 95\% percentile intervals from 10,000 joint full-bitstring bootstrap draws per pub. Job entries average the four fixed round statistics. The seven singleton blocks have zero covariance contribution and no normalized ratio. Matched denotes the historical XY4-and-twirling execution; placements and dates differ from defaults.}',
             r'\label{tab:sm_block_profiles}', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{ruledtabular}', r'\begin{tabular}{rrrccccc}',
             r'Size & $\Delta_\alpha^{\rm id}$ & $w_\alpha$ (\%) & Defaults & Matched & E01 & E02 & E03 \\', r'\hline']
    for j, size in enumerate(ref['block_sizes']):
        cells = [str(size), f"{ref['delta'][j]:.4f}", f"{100*ref['weights'][j]:.2f}"]
        cells += [interval(c['statistics']['retention'], j) for c in contexts]
        lines.append(' & '.join(cells)+r' \\')
    lines.append(' & '.join(['Global', f"{ref['global_delta']:.4f}", '100']+
                           [interval(c['statistics']['global_retention']) for c in contexts])+r' \\')
    lines += [r'\end{tabular}', r'\end{ruledtabular}', r'\end{table*}', '', r'\begin{table*}[t]',
              r'\caption{Block-minus-global contrasts $G_\alpha=C_\alpha-C_{\rm global}$. Brackets are approximate simultaneous conditional 95\% bands for the single exploratory family of all 30 entries, using the standardized bootstrap maximum. An asterisk indicates exclusion of zero under the counting model, without a causal or population interpretation.}',
              r'\label{tab:sm_block_contrasts}', r'\scriptsize', r'\begin{ruledtabular}',
              r'\begin{tabular}{rccccc}', r'Size & Defaults & Matched & E01 & E02 & E03 \\', r'\hline']
    for j, size in enumerate(ref['block_sizes']):
        cells = [str(size)]
        for c in contexts:
            g = c['statistics']['contrast']['estimate'][j]
            lo, hi = c['contrast_simultaneous_ci95'][j]
            star = '*' if c['contrast_resolved'][j] else ''
            cells.append(f'${g:.3f}$ $[{lo:.3f},{hi:.3f}]^{{{star}}}$')
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\end{tabular}', r'\end{ruledtabular}', r'\end{table*}']
    (output/'block_covariance_table.tex').write_text('\n'.join(lines)+'\n')
    macros = {'BlockLargeWeight': f"{100*sum(w for s,w in zip(ref['block_sizes'],ref['weights']) if s in (9,13)):.1f}",
              'BlockFiveWeight': f"{100*ref['weights'][ref['block_sizes'].index(5)]:.1f}",
              'BlockMaxQuantile': f"{result['simultaneous']['quantile95']:.4f}"}
    index = ref['block_sizes'].index(5)
    for c in contexts:
        key = {'defaults': 'Defaults', 'matched': 'Matched', 'E01': 'EOne', 'E02': 'ETwo', 'E03': 'EThree'}[c['context']]
        macros['BlockFive'+key] = f"{c['statistics']['retention']['estimate'][index]:.3f}"
        lo, hi = c['contrast_simultaneous_ci95'][index]
        macros['BlockFive'+key+'Low'] = f'{lo:.3f}'
        macros['BlockFive'+key+'High'] = f'{hi:.3f}'
    (output/'block_covariance_numbers.tex').write_text('\n'.join(
        rf'\newcommand{{\{key}}}{{{value}}}' for key,value in macros.items())+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=HERE/'generated/block_covariance.json')
    parser.add_argument('--out-dir', type=Path, default=HERE/'generated')
    args = parser.parse_args()
    write(json.loads(args.input.read_text()), args.out_dir)


if __name__ == '__main__':
    main()
