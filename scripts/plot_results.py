"""누적 결과 그래프.
  1) 망각 곡선: outputs/results.csv → x = N, y = 유지율 (모델별 선), control은 배경 정확도 보조 선
  2) horizon별 rollout MSE: outputs/<run>/log.csv의 마지막 eval
python scripts/plot_results.py [--point back] [--runs outputs/baseline]
"""
import argparse
import collections
import csv
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--results', default='outputs/results.csv')
p.add_argument('--point', default='back')
p.add_argument('--runs', nargs='*', default=['outputs/baseline'])
p.add_argument('--out', default='outputs/figures')
p.add_argument('--include', default='.*', help='그릴 config 정규식')
p.add_argument('--exclude', default='$^', help='뺄 config 정규식')
p.add_argument('--tag', default='', help='그림 파일 이름 접미사')
args = p.parse_args()
out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(open(args.results)))
latest = {}  # 같은 config를 여러 번 평가했으면 마지막 것만
for r in rows:
  latest[r['config']] = max(latest.get(r['config'], 0), int(r['time']))
import re  # noqa: E402
rows = [r for r in rows if r['point'] == args.point and int(r['time']) >= latest[r['config']] - 600
        and re.search(args.include, r['config']) and not re.search(args.exclude, r['config'])]

data = collections.defaultdict(list)
for r in rows:
  data[r['config'], r['kind'], int(r['N'])].append(r)
configs = sorted({k[0] for k in data})
Ns = sorted({k[2] for k in data})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
panels = [('keep', ['place_stone', 'place_table'], 'changed tile kept (place stone/table)'),
          ('keep', ['collect_tree'], 'changed tile kept (tree → grass)'),
          ('bg_acc', ['control'], 'background tile accuracy (control)')]
for ax, (metric, kinds, title) in zip(axes, panels):
  for c in configs:
    ys, es = [], []
    for N in Ns:
      v = np.array([float(r[metric]) for k in kinds for r in data.get((c, k, N), [])])
      v = v[~np.isnan(v)]
      ys.append(v.mean() if len(v) else np.nan)
      es.append(1.96 * v.std() / np.sqrt(len(v)) if len(v) > 1 else 0)
    ax.errorbar(Ns, ys, yerr=es, marker='o', capsize=3, label='B0 (copy last frame)' if c == '-' else c)
  ax.set_xscale('log', base=2)
  ax.set_xticks(Ns, [str(n) for n in Ns])
  ax.set_ylim(-0.02, 1.02)
  ax.set_xlabel('N = steps the changed tile stayed out of view')
  ax.set_title(title, fontsize=10)
  ax.grid(alpha=0.3)
axes[0].set_ylabel(f'accuracy at "{args.point}" frame')
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / f'forgetting_curve_{args.point}{args.tag}.png', dpi=130)
print('saved', out / f'forgetting_curve_{args.point}{args.tag}.png')

# horizon별 rollout MSE
fig, ax = plt.subplots(figsize=(5, 4))
for run in args.runs:
  log = pathlib.Path(run) / 'log.csv'
  if not log.exists():
    continue
  evals = [r for r in csv.reader(open(log)) if r[0] == 'eval']
  if not evals:
    continue
  res = json.loads(evals[-1][2])
  hs = sorted(int(k.split('@')[1]) for k in res if k.startswith('rollout_mse'))
  ax.plot(hs, [res[f'rollout_mse@{h}'] for h in hs], marker='o', label=f'{pathlib.Path(run).name} @{evals[-1][1]}')
  ax.plot(hs, [res[f'b0_mse@{h}'] for h in hs], marker='x', ls='--', color='gray', label='copy last frame (B0)')
ax.set_xlabel('rollout horizon (steps)')
ax.set_ylabel('MSE (pixels in [-1, 1])')
ax.set_xscale('log')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / 'rollout_mse.png', dpi=130)
print('saved', out / 'rollout_mse.png')
