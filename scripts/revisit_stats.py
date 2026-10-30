"""데이터에서 '변화 → 시야 이탈 → 복귀' 장면이 얼마나 나오는지 센다 (memory 학습 신호의 양)."""
import collections
import glob
import sys

import numpy as np

sys.path.insert(0, '.')
from src.env import VIEW  # noqa: E402

d = sys.argv[1] if len(sys.argv) > 1 else 'data/debug'
hx, hy = VIEW[0] // 2, VIEW[1] // 2
cnt = collections.Counter()
for f in sorted(glob.glob(f'{d}/ep*.npz')):
  z = np.load(f)
  pol = str(z['policy'])
  mat, pos = z['material'], z['player_pos'].astype(int)
  diff = np.argwhere(mat[1:] != mat[:-1])  # (t, x, y): 프레임 t→t+1에서 바뀐 타일
  for t, x, y in diff:
    cnt[pol, 'changes'] += 1
    rel = np.abs(np.array([x, y]) - pos[t + 1:])
    vis = (rel[:, 0] <= hx) & (rel[:, 1] <= hy)
    out = np.flatnonzero(~vis)
    if len(out) == 0:
      continue
    # 첫 이탈 이후 다시 보이는 시점까지 시야 밖에 머문 스텝 수
    first_out = out[0]
    back = np.flatnonzero(vis[first_out:])
    if len(back):
      gap = back[0]
      for n in [4, 8, 16, 32, 64]:
        if gap >= n:
          cnt[pol, f'revisit_gap>={n}'] += 1
for k in sorted(cnt):
  print(k, cnt[k])
