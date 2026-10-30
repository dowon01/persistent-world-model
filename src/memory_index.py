"""memory 학습용 인덱스 미리 계산 (에피소드별, 한 번만).

에피소드의 각 시점 t (context 마지막 프레임)에 대해:
  query[t]  = 다음 프레임의 예상 위치 = p_t + (이동 행동이면 그 방향, 아니면 0)
              평가·데모에서도 다음 프레임을 만들기 전에는 이동 성공 여부를 모르므로 같은 규칙을 쓴다.
              (막힌 이동이면 검색 프레임이 한 칸 어긋난다. 모델이 이것도 학습 중에 본다.)
  ret[t]    = 검색할 과거 프레임 s <= t-K. 없으면 -1
              rule=overlap: 시야 겹침 최대 (같으면 가장 최근)
              rule=recent : 다음 스텝에 새로 드러나는 줄을 온전히 담은 가장 최근 프레임, 없으면 overlap
                            (평가 규칙과 같음, src/models/memory.py FrameMemory 참고)
  cat[t]    = 2: 다음 프레임에 새로 드러나는 타일 중 플레이어가 바꾼 타일이 과거에 보였음 (memory만 맞출 수 있음)
              1: 새로 드러나는 타일 중 과거에 본 지형이 있음 (memory가 도울 수 있음)
              0: 그 외
python src/memory_index.py --data_dir data/train --context 4
"""
import argparse
import glob
import multiprocessing as mp
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.env import MOVES, VIEW  # noqa: E402

HX, HY = VIEW[0] // 2, VIEW[1] // 2


def episode_index(args):
  f, K, rule = args
  z = np.load(f)
  pos, act, mat = z['player_pos'].astype(np.int64), z['action'].astype(np.int64), z['material']
  T = len(pos)
  delta = np.zeros((T, 2), np.int64)
  for a, d in MOVES.items():
    delta[:len(act)][act == a] = d
  query = pos + delta
  ret = np.full(T, -1, np.int64)
  cat = np.zeros(T, np.int8)
  # 시야 = 64×64 월드에 여백 HX/HY를 둔 격자 위의 사각형
  def vmask(p):
    m = np.zeros((64 + 2 * HX, 64 + 2 * HY), bool)
    m[p[0]:p[0] + VIEW[0], p[1]:p[1] + VIEW[1]] = True  # 여백 좌표: (p - h) + h = p
    return m
  views = np.stack([vmask(p) for p in pos])
  changed = np.zeros_like(views)
  changed[:, HX:HX + 64, HY:HY + 64] = mat != mat[0]
  seen_before = np.zeros_like(views[0])
  for t in range(K - 1, T - 1):
    if t - K >= 0:
      seen_before |= views[t - K]  # s <= t-K 까지 본 타일 누적
      d = np.abs(pos[:t - K + 1] - query[t])
      ov = np.clip(VIEW[0] - d[:, 0], 0, None) * np.clip(VIEW[1] - d[:, 1], 0, None)
      if ov.max() > 0:
        ret[t] = int(np.flatnonzero(ov == ov.max())[-1])
      if rule == 'recent' and np.any(query[t] != pos[t]):
        # 새로 드러나는 줄: 다음 시야 사각형에서 이동 방향 쪽 끝 줄
        lo, hi = query[t] - (HX, HY), query[t] + (HX, HY)
        rlo, rhi = lo.copy(), hi.copy()
        for ax in range(2):
          if query[t][ax] > pos[t][ax]:
            rlo[ax] = hi[ax]
          elif query[t][ax] < pos[t][ax]:
            rhi[ax] = lo[ax]
        past = pos[:t - K + 1]
        covers = np.all((past - (HX, HY) <= rlo) & (past + (HX, HY) >= rhi), 1)
        if covers.any():
          ret[t] = int(np.flatnonzero(covers)[-1])
    rev = views[t + 1] & ~views[t - K + 1:t + 1].any(0)
    seen = rev & seen_before
    if (seen & changed[t + 1]).any():
      cat[t] = 2
    elif seen.any():
      cat[t] = 1
  return pathlib.Path(f).stem, query, ret, cat


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--data_dir', default='data/train')
  p.add_argument('--context', type=int, default=4)
  p.add_argument('--workers', type=int, default=12)
  p.add_argument('--rule', default='overlap', choices=['overlap', 'recent'])
  args = p.parse_args()
  files = sorted(glob.glob(f'{args.data_dir}/ep*.npz'))
  out = {}
  with mp.Pool(args.workers) as pool:
    for stem, q, r, c in pool.imap(episode_index, [(f, args.context, args.rule) for f in files]):
      out[f'{stem}/query'], out[f'{stem}/ret'], out[f'{stem}/cat'] = q, r, c
  suffix = '' if args.rule == 'overlap' else f'_{args.rule}'
  path = pathlib.Path(args.data_dir) / f'memory_index_K{args.context}{suffix}.npz'
  np.savez(path, **out)
  cats = np.concatenate([out[k] for k in out if k.endswith('/cat')])
  rets = np.concatenate([out[k] for k in out if k.endswith('/ret')])
  print(f'saved {path}: {len(cats)} 시점, cat0 {np.mean(cats == 0):.3f} cat1 {np.mean(cats == 1):.3f} '
        f'cat2 {np.mean(cats == 2):.4f}, 검색 결과 있음 {np.mean(rets >= 0):.3f}')


if __name__ == '__main__':
  main()
