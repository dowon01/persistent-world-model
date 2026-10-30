"""변화 그리기 정확도 (v2 평가).

검증 에피소드(학습에 안 쓴 마지막 5%)의 상호작용 장면에서, 실제 context로 1스텝 예측한 프레임의
'바라보는 칸'을 템플릿 매칭으로 판정한다.
  render  : 성공 장면에서 결과 재질을 그린 비율 (높을수록 좋음)
  false   : 실패 장면에서 재질을 바꿔 그린 비율 (낮을수록 좋음. 성공 조건을 배웠는지)
python src/eval_change.py --models outputs/baseline/last.pt outputs/baseline_v2/last.pt
"""
import argparse
import collections
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.dataset import FrameDataset, list_episodes, split_episodes  # noqa: E402
from src.env import ACTIONS, CrafterEnv, tile_patch  # noqa: E402
from src.eval_revisit import load_model  # noqa: E402
from src.interact_index import INTERACT_ACTS  # noqa: E402
from src.rollout import to_uint8  # noqa: E402
from src.tiles import TileClassifier  # noqa: E402


def collect(files, per=300, ctx=16):
  out = collections.defaultdict(list)
  for f in files:
    z = np.load(f)
    act, mat, pos, fac = z['action'], z['material'], z['player_pos'].astype(int), z['facing'].astype(int)
    for t in range(ctx - 1, len(act)):
      if act[t] not in INTERACT_ACTS:
        continue
      tgt = pos[t] + fac[t]
      if not (0 <= tgt[0] < 64 and 0 <= tgt[1] < 64):
        continue
      ok = mat[t + 1][tuple(tgt)] != mat[t][tuple(tgt)]
      key = (ACTIONS[act[t]], 'success' if ok else 'fail')
      if len(out[key]) < per:
        out[key].append(dict(ctx=z['obs'][t - ctx + 1:t + 1], a=int(act[t]), tile=(4 + fac[t][0], 3 + fac[t][1]),
                             before=int(mat[t][tuple(tgt)]), after=int(mat[t + 1][tuple(tgt)])))
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--models', nargs='+', required=True)
  p.add_argument('--data_dir', default='data/train')
  args = p.parse_args()
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  clf = TileClassifier(CrafterEnv(seed=0))
  _, val = split_episodes(list_episodes(args.data_dir))
  samples = collect(val)
  print('검증 장면 수:', {f'{a}/{k}': len(v) for (a, k), v in sorted(samples.items())})
  for spec in args.models:
    model, name, *_ = load_model(spec, dev)
    res = {}
    for (a, kind), ss in sorted(samples.items()):
      hits = []
      for i in range(0, len(ss), 64):
        b = ss[i:i + 64]
        ctx = FrameDataset.to_float(torch.from_numpy(np.stack([x['ctx'] for x in b])).to(dev))
        st = model.init_context(ctx)
        out, _ = model.step(st, torch.tensor([x['a'] for x in b], device=dev))
        out = to_uint8(out)
        for o, x in zip(out, b):
          m = int(clf.classify_patches(tile_patch(o, x['tile']))[0])
          hits.append(m == x['after'] if kind == 'success' else m != x['before'])
      res[a, kind] = np.mean(hits)
    line = '  '.join(f'{a} render {res[a, "success"]:.2f} / false {res.get((a, "fail"), np.nan):.2f}'
                     for a in ['do', 'place_stone', 'place_table'] if (a, 'success') in res)
    print(f'{name:22s} {line}')


if __name__ == '__main__':
  main()
