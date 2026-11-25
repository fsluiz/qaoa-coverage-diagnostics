"""V7 ensemble figure: representation size and fixed-cell paired bootstrap.

The 24 design cells (four register sizes, six radii) remain fixed. Each draw
resamples the 25 paired instances within each cell, recomputes its Spearman
coefficient, and averages the six coefficients at each register size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RADII = (0.9, 1.35, 1.8, 2.4, 3.0, 3.9)
SIZES = (20, 40, 80, 160)


def spearman_rows(x, y):
    x, y = rankdata(x, axis=-1), rankdata(y, axis=-1)
    x, y = x - x.mean(axis=-1, keepdims=True), y - y.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((x*x).sum(axis=-1) * (y*y).sum(axis=-1))
    return np.divide((x*y).sum(axis=-1), denominator,
                     out=np.full_like(denominator, np.nan), where=denominator > 0)


def paired_bootstrap(x, ys, rng, n_boot):
    indices = rng.integers(0, len(x), size=(n_boot, len(x)))
    draws = np.column_stack([spearman_rows(x[indices], y[indices]) for y in ys])
    # Constant rank vectors make Spearman undefined; redraw and record any such event.
    rejected = 0
    for _ in range(100):
        bad = ~np.isfinite(draws).all(axis=1)
        if not bad.any():
            return draws, rejected
        rejected += int(bad.sum())
        indices = rng.integers(0, len(x), size=(int(bad.sum()), len(x)))
        draws[bad] = np.column_stack([spearman_rows(x[indices], y[indices]) for y in ys])
    raise RuntimeError("Degenerate bootstrap ranks; inspect cell rather than imputing zero.")


def analyze(n_boot=4000, seed=20261016):
    block_rows = json.loads((PROJECT/'results/block_decomposition.json').read_text())["ensemble"]
    cover_rows = json.loads((PROJECT/'results/ensemble_squeeze.json').read_text())["rows"]
    key = lambda r: (r['n_b'], r['R_over_delta'], r['seed'])
    blocks, covers = {key(r): r for r in block_rows}, {key(r): r for r in cover_rows}
    if len(blocks) != 600 or blocks.keys() != covers.keys():
        raise ValueError("Expected 600 uniquely matched instances.")
    if any(r['ilp_status'] != 0 for r in cover_rows):
        raise ValueError("An ensemble optimum is not certified.")
    rng, cells, sizes = np.random.default_rng(seed), [], []
    for nb in SIZES:
        cell_draws, cell_points = [], []
        for radius in RADII:
            keys = sorted(k for k in blocks if k[0] == nb and abs(k[1]-radius) < 1e-8)
            if len(keys) != 25:
                raise ValueError((nb, radius, len(keys)))
            edges = np.array([blocks[k]['n_ZZ'] for k in keys], dtype=float)
            dimensions = np.array([blocks[k]['log2_enum'] for k in keys])
            forced = np.array([covers[k]['phi'] for k in keys])
            x = np.log10(np.maximum(edges, 1))
            points = [float(spearman_rows(x, y)) for y in [dimensions, forced]]
            draws, rejected = paired_bootstrap(x, [dimensions, forced], rng, n_boot)
            cell_points.append(points)
            cell_draws.append(draws)
            cells.append(dict(n_b=nb, R_over_delta=radius, n=25,
                              mean_nzz=float(edges.mean()),
                              mean_largest_block=float(np.mean([blocks[k]['largest_block'] for k in keys])),
                              dimension_spearman=points[0], phi_spearman=points[1],
                              degenerate_replicates_redrawn=rejected))
        point = np.mean(cell_points, axis=0)
        distribution = np.mean(cell_draws, axis=0)
        interval = np.percentile(distribution, [2.5, 97.5], axis=0)
        sizes.append(dict(n_b=nb, n_cells=6, n_instances=150,
                          dimension=dict(estimate=float(point[0]), ci95=interval[:,0].tolist()),
                          phi=dict(estimate=float(point[1]), ci95=interval[:,1].tolist())))
    pooled = float(spearman_rows(
        np.array([np.log10(max(blocks[k]['n_ZZ'], 1)) for k in sorted(blocks)]),
        np.array([blocks[k]['log2_enum'] for k in sorted(blocks)])))
    output = dict(schema_version=1, bootstrap_replicates=n_boot, bootstrap_seed=seed,
                  bootstrap_unit="paired instances within each fixed (n_B, R/delta) cell",
                  estimand="equal-weight mean of six cell Spearman coefficients at each n_B",
                  intervals="pointwise percentile intervals; not simultaneous inference",
                  pooled_spearman=pooled, cells=cells, by_register_size=sizes)
    return output, blocks


def plot(output, blocks, path):
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42})
    colors = dict(zip(SIZES, ['#88CCEE','#44AA99','#CC6677','#332288']))
    markers = dict(zip(RADII, ['o','s','^','D','v','P']))
    fig, axes = plt.subplots(1,2,figsize=(7.4,3.05))
    for nb in SIZES:
        for radius in RADII:
            rows = [r for k,r in blocks.items() if k[0] == nb and abs(k[1]-radius)<1e-8]
            axes[0].scatter([r['n_ZZ'] for r in rows], [r['log2_enum'] for r in rows],
                            s=12,alpha=.65,color=colors[nb],marker=markers[radius],edgecolor='none')
    axes[0].set(xscale='log',xlabel=r'$n_{ZZ}$ (logical interactions)',
                ylabel=r'$\log_2\!\left(\sum_i 2^{|V_i|}\right)$',
                title='(a) Block-state representation size')
    axes[0].legend(handles=[Patch(color=c,label=rf'$n_B={n}$') for n,c in colors.items()]+
                   [Line2D([],[],ls='',marker=m,color='#777777',label=rf'$R/\delta={r}$')
                    for r,m in markers.items()],ncol=2,fontsize=6,frameon=False,loc='upper left')
    for key,label,color,fmt in [('dimension','block-state dimension','#332288','-o'),
                                ('phi',r'forced fraction $\phi$','#CC6677','--s')]:
        rows = [r[key] for r in output['by_register_size']]
        points = np.array([r['estimate'] for r in rows]); limits=np.array([r['ci95'] for r in rows])
        axes[1].errorbar(SIZES,points,yerr=[points-limits[:,0],limits[:,1]-points],
                         fmt=fmt,color=color,label=label,capsize=3,markersize=4)
    axes[1].axhline(0,color='#777777',lw=.8)
    axes[1].set(xscale='log',xlabel=r'register size $n_B$',
                ylabel=r'mean within-cell Spearman $\rho$',
                title=r'(b) Fixed $n_B$ and $R$ within each cell')
    axes[1].set_xticks(SIZES, [str(n) for n in SIZES]); axes[1].xaxis.set_minor_formatter(NullFormatter())
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc='lower center',
               bbox_to_anchor=(.5, -.03), ncol=2)
    fig.tight_layout(rect=(0, .11, 1, 1), w_pad=1.5)
    fig.savefig(path,bbox_inches='tight'); plt.close(fig)


def write_table(output, path):
    lines = [r'\begin{table}[t]',
             r'\caption{Mean of the six within-cell Spearman coefficients at each register size. Brackets are pointwise 95\% percentile intervals from 4000 paired-instance bootstrap draws within each fixed cell; radii are not resampled. Each cell contains 25 instances.}',
             r'\label{tab:sm_ensemble_correlations}',r'\begin{ruledtabular}',
             r'\begin{tabular}{ccc}',r'$n_B$ & Block-state dimension & Forced fraction $\phi$ \\',r'\hline']
    for row in output['by_register_size']:
        values=[]
        for key in ['dimension','phi']:
            r=row[key]; values.append(f"${r['estimate']:.2f}$ $[{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]$")
        lines.append(f"{row['n_b']} & "+' & '.join(values)+r' \\')
    lines += [r'\end{tabular}',r'\end{ruledtabular}',r'\end{table}']
    path.write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out-dir',type=Path,default=HERE/'generated')
    args=parser.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    output,blocks=analyze()
    (args.out_dir/'ensemble_analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    plot(output,blocks,args.out_dir/'fig_ensemble.pdf')
    write_table(output,args.out_dir/'ensemble_table.tex')
    print(json.dumps(output['by_register_size'],indent=2))


if __name__=='__main__':
    main()
