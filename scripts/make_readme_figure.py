"""README용 망각 곡선: 바꾼 타일(돌·table)이 시야 밖에 N스텝 있다가 다시 보인 순간 유지된 비율.
python scripts/make_readme_figure.py --out assets/forgetting_curve.png
"""
import argparse
import collections
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LINES = [  # (results.csv의 config, 범례, 스타일)
    ('memory_aligned_v3@80000+oracle_pos', 'Memory + true position (upper bound)', dict(color='#2ca02c', ls='--', marker='')),
    ('memory_aligned_v3@80000', 'Memory (ours)', dict(color='#2ca02c', marker='o')),
    ('baseline_k16@80000', 'Baseline, 16-frame context', dict(color='#ff7f0e', marker='s')),
    ('baseline_v2@80000', 'Baseline, 4-frame context', dict(color='#d62728', marker='^')),
    ('-', 'Copy last frame', dict(color='gray', marker='x', ls=':')),
]
p = argparse.ArgumentParser()
p.add_argument('--results', default='outputs/results.csv')
p.add_argument('--out', default='assets/forgetting_curve.png')
args = p.parse_args()
rows = list(csv.DictReader(open(args.results)))
latest = {}
for r in rows:
  latest[r['config']] = max(latest.get(r['config'], 0), int(r['time']))
g = collections.defaultdict(list)
for r in rows:
  if r['point'] == 'back' and r['kind'] in ('place_stone', 'place_table') and int(r['time']) >= latest[r['config']] - 600:
    g[r['config'], int(r['N'])].append(float(r['keep']))
Ns = sorted({n for _, n in g})
fig, ax = plt.subplots(figsize=(6.4, 4))
for cfg, label, style in LINES:
  ys = np.array([np.mean(g[cfg, n]) for n in Ns])
  es = np.array([1.96 * np.std(g[cfg, n]) / np.sqrt(len(g[cfg, n])) for n in Ns])
  ax.plot(Ns, ys, label=label, lw=2, **style)
  ax.fill_between(Ns, ys - es, ys + es, color=style['color'], alpha=0.12)
ax.set_xscale('log', base=2)
ax.set_xticks(Ns, [str(n) for n in Ns])
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel('N = steps the placed block stayed out of view')
ax.set_ylabel('block remembered when it comes back into view')
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc='center right')
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print('saved', args.out)
