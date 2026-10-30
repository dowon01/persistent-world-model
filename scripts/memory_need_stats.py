"""학습 샘플 중 memory가 필요한 비율 (memory M1 설계용).

샘플 = context 프레임 t-K+1..t → 다음 프레임 t+1.
  revealed  = 다음 시야의 타일 중 context 어느 프레임에도 안 보였던 타일
  seen      = 그중 context보다 이전 프레임(같은 에피소드)에서 본 적이 있는 타일 → memory가 도울 수 있음
  changed   = seen 중 에피소드 시작 이후 재질이 바뀐 타일 → 지형 사전지식으로는 못 맞추고 memory만 맞출 수 있음
  single    = seen 타일을 '가장 많이 겹치는 과거 프레임 1장'(M1)이 덮는 비율
"""
import glob
import sys

import numpy as np

sys.path.insert(0, '.')
from src.env import VIEW  # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
hx, hy = VIEW[0] // 2, VIEW[1] // 2
files = sorted(glob.glob('data/train/ep*.npz'))[::4]
n = n_rev = n_seen = n_chg = 0
cover = []
for f in files:
  z = np.load(f)
  pos, mat = z['player_pos'].astype(int), z['material']
  T = len(pos)

  def view(p):
    return {(x, y) for x in range(p[0] - hx, p[0] + hx + 1) for y in range(p[1] - hy, p[1] + hy + 1)}

  views = [view(p) for p in pos]
  for t in range(K - 1, T - 1):
    n += 1
    ctx = set().union(*views[t - K + 1:t + 1])
    rev = views[t + 1] - ctx
    if not rev:
      continue
    n_rev += 1
    past = range(0, t - K + 1)
    seen = {c for c in rev if any(c in views[s] for s in past)}
    if not seen:
      continue
    n_seen += 1
    chg = {c for c in seen if 0 <= c[0] < 64 and 0 <= c[1] < 64 and mat[t + 1][c] != mat[0][c]}
    n_chg += bool(chg)
    best = max(len(seen & views[s]) for s in past)  # 단일 프레임이 덮는 seen 타일 수
    cover.append(best / len(seen))
print(f'K={K}  samples {n}')
print(f'  새 타일이 드러나는 샘플     {n_rev / n:.3f}')
print(f'  그중 과거에 본 타일 포함     {n_seen / n:.3f}  (memory가 도울 수 있음)')
print(f'  그중 바뀐 타일 포함          {n_chg / n:.4f}  (memory만 맞출 수 있음)')
print(f'  단일 프레임 검색의 seen 커버 평균 {np.mean(cover):.3f}, 100% 커버 비율 {np.mean(np.array(cover) == 1):.3f}')
