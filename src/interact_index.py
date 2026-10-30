"""상호작용 학습용 인덱스 (v2·v3 학습).

모델이 설치·채집 결과를 전혀 그리지 못했다 (성공 장면 1스텝 정확도 0%). 같은 행동의 성공률이 낮고
(place_stone 0.39, place_table 0.20, do 0.16) 바뀌는 칸이 화면의 1.2%뿐이라 MSE가 '안 바뀜' 평균으로 간다.
→ 상호작용 장면을 더 자주 뽑고, 바뀐 칸에 손실 가중치를 준다. 이를 위한 인덱스.

에피소드의 각 시점 t (행동 t: 프레임 t → t+1):
  inter[t] = 1: 상호작용 성공 (do / place_stone / place_table / place_furnace 뒤 바라보는 칸의 재질이 바뀜)
             2: 상호작용 시도했지만 아무것도 안 바뀜 (놓을 수 없는 칸, 동물 등). 성공 조건을 배우려면 실패도 필요
             3: 실패 중 인벤토리 부족 (설치 재료 또는 채집 도구가 없음). v3에서 따로 더 뽑는다
             0: 그 외 (place_plant는 재질이 아니라 object를 만들므로 제외)
  tgt[t]   = 상호작용 시 바라보는 월드 칸 (아니면 -1). v3에서 실패 장면도 이 칸에 가중치를 건다
  chg[t]   = t → t+1에서 재질이 바뀐 월드 칸 (최대 2개, 없으면 -1) [T, 2, 2]
  inv[t]   = 인벤토리가 바뀌었는가 (상호작용 성공일 때만 가중치에 쓴다. 배고픔·보충 같은 행동과 무관한 변화 제외)
python src/interact_index.py --data_dir data/train
"""
import argparse
import glob
import multiprocessing as mp
import pathlib
import sys

import crafter
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.collect import ITEMS  # noqa: E402
from src.env import A, ACTIONS  # noqa: E402

INTERACT_ACTS = [A['do'], A['place_stone'], A['place_table'], A['place_furnace']]


def episode_index(f):
  z = np.load(f)
  act, mat, pos, fac, inv = z['action'], z['material'], z['player_pos'].astype(int), z['facing'].astype(int), z['inventory']
  T = len(pos)
  inter = np.zeros(T, np.int8)
  chg = np.full((T, 2, 2), -1, np.int16)
  invc = np.zeros(T, bool)
  tgts = np.full((T, 2), -1, np.int16)
  names = {i: n for n, i in crafter.engine.World((64, 64), crafter.constants.materials, (12, 12))._mat_ids.items()}
  for t in range(len(act)):
    d = np.argwhere(mat[t + 1] != mat[t])[:2]
    chg[t, :len(d)] = d
    invc[t] = np.any(inv[t + 1] != inv[t])
    if act[t] in INTERACT_ACTS:
      tgt = pos[t] + fac[t]
      inside = 0 <= tgt[0] < 64 and 0 <= tgt[1] < 64
      if inside:
        tgts[t] = tgt
      if inside and mat[t + 1][tuple(tgt)] != mat[t][tuple(tgt)]:
        inter[t] = 1
      else:
        inter[t] = 3 if inside and lacks_item(ACTIONS[act[t]], inv[t], names[int(mat[t][tuple(tgt)])]) else 2
  return pathlib.Path(f).stem, inter, chg, invc, tgts


def lacks_item(action, inv, target_mat):
  """인벤토리 부족으로 실패했는가: 설치 재료 또는 채집 도구."""
  have = lambda k, v: inv[ITEMS.index(k)] >= v  # noqa: E731
  if action.startswith('place_'):
    uses = crafter.constants.place[action[len('place_'):]]['uses']
    return not all(have(k, v) for k, v in uses.items())
  info = crafter.constants.collect.get(target_mat)  # do
  return bool(info) and not all(have(k, v) for k, v in info['require'].items())


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--data_dir', default='data/train')
  args = p.parse_args()
  out = {}
  with mp.Pool(12) as pool:
    for stem, i, c, v, g in pool.imap(episode_index, sorted(glob.glob(f'{args.data_dir}/ep*.npz'))):
      out[f'{stem}/inter'], out[f'{stem}/chg'], out[f'{stem}/inv'], out[f'{stem}/tgt'] = i, c, v, g
  path = pathlib.Path(args.data_dir) / 'interact_index.npz'
  np.savez(path, **out)
  inter = np.concatenate([out[k] for k in out if k.endswith('/inter')])
  print(f'saved {path}: {len(inter)} 시점, 성공 {(inter == 1).sum()}, 실패(그 외) {(inter == 2).sum()}, '
        f'실패(인벤토리 부족) {(inter == 3).sum()}')


if __name__ == '__main__':
  main()
