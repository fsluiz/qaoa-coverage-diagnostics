"""Generate the v7 campaign figure, table and numerical LaTeX macros."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE=Path(__file__).resolve().parent
KEYS=('joint_over_slice_gated','faithful51_correlation_energy_retained')


def scientific(value):
    if value == 0: return '0'
    power=math.floor(math.log10(abs(value)))
    return rf'{value/10**power:.3f}\times10^{{{power}}}'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=HERE/'results')
    parser.add_argument('--out-dir',type=Path,default=HERE/'generated')
    args=parser.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    epochs=[json.loads((args.results/f'e0{i}_analysis.json').read_text()) for i in range(1,4)]
    combined=json.loads((args.results/'between_jobs.json').read_text())['between_job_summary']
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    figure,axes=plt.subplots(1,2,figsize=(7.2,3.05))
    for axis,key,title,ylabel in zip(axes,KEYS,
        ['(a) PEF-24 gated blocks','(b) Faithful 51-qubit circuit'],
        [r'$R_e$: joint / isolated probability product',r'$C_e$: ideal covariance contribution retained']):
        for index,(epoch,color) in enumerate(zip(epochs,['#4477AA','#EE6677','#228833'])):
            summary=epoch['epoch_summary']; estimate=summary[key]['estimate']
            ci=np.array(summary['conditional_shot_epoch_ci95'][key])
            rounds=np.array(summary['round_values'][key])
            axis.scatter(index+np.linspace(-.13,.13,4),rounds,s=25,facecolors='none',edgecolors=color,zorder=2)
            axis.errorbar(index,estimate,yerr=np.array([[estimate-ci[0]],[ci[1]-estimate]]),
                           marker='o',markersize=5,color=color,capsize=3,lw=1.3,zorder=3)
        axis.set_xticks(range(3),['E01\nOct 8','E02\nOct 14','E03\nOct 14'])
        axis.set(xlim=(-.35,2.35),ylabel=ylabel,title=title)
        axis.grid(axis='y',color='#dddddd',lw=.6)
    figure.legend(handles=[Line2D([],[],marker='o',ls='',color='#444444',label='four-round job estimate; bars: conditional counting interval'),
                           Line2D([],[],marker='o',ls='',mfc='none',color='#777777',label='individual observed round')],
                   loc='lower center',bbox_to_anchor=(.5,-.03),ncol=1,fontsize=8,frameon=False)
    figure.tight_layout(rect=(0,.13,1,1),w_pad=2)
    figure.savefig(args.out_dir/'fig_three_jobs.pdf',bbox_inches='tight'); plt.close(figure)
    lines=[r'\begin{table}[t]',r'\caption{Fixed-circuit jobs, labeled by their associated properties snapshots (UTC$-3$). Brackets are conditional counting-model 95\% percentile intervals for the four-round job estimand, from 4000 bootstrap draws with the observed rounds held fixed. They do not quantify drift or between-job uncertainty.}',
           r'\label{tab:sm_threecal}',r'\setlength{\tabcolsep}{3pt}',r'\begin{ruledtabular}',
           r'\begin{tabular}{lccc}',r'Epoch (snapshot) & Endpoint & Estimate & Conditional interval \\',r'\hline']
    for epoch,label in zip(epochs,['E01 (Oct.~8, 08:55)','E02 (Oct.~14, 14:27)','E03 (Oct.~14, 18:25)']):
        for key,symbol in zip(KEYS,['R_e','C_e']):
            summary=epoch['epoch_summary']; ci=summary['conditional_shot_epoch_ci95'][key]
            lines.append(f"{label} & ${symbol}$ & {summary[key]['estimate']:.4f} & $[{ci[0]:.4f},{ci[1]:.4f}]$"+r' \\')
            label=''
    lines += [r'\end{tabular}',r'\end{ruledtabular}',r'\end{table}']
    (args.out_dir/'campaign_table.tex').write_text('\n'.join(lines)+'\n')
    macros=[]
    for prefix,key in zip(['R','C'],KEYS):
        row=combined[key]
        for suffix,value in [('ObservedVariance',row['observed_between_job_variance']),
                             ('MeanEstimationVariance',row['mean_epoch_estimation_variance']),
                             ('ExcessVariance',row['excess_between_job_variance_method_of_moments'])]:
            macros.append(rf'\newcommand{{\{prefix}{suffix}}}{{{scientific(value)}}}')
        macros.append(rf'\newcommand{{\{prefix}VarianceRatio}}{{{row["observed_to_estimation_variance_ratio"]:.1f}}}')
        macros.append(rf'\newcommand{{\{prefix}VariancePercent}}{{{100*row["observed_to_estimation_variance_ratio"]:.1f}}}')
    (args.out_dir/'campaign_numbers.tex').write_text('\n'.join(macros)+'\n')
    print('Generated v7 campaign figure, table and numerical macros.')


if __name__=='__main__':
    main()
