"""약점 2 분석: baseline_v2가 짧은 N(1~3)에서 v1보다 낮다 (N=1/2/3: 0.90/0.13/0.41 → 0.74/0.06/0.07).

변화 직후부터 rollout하면서, 변경 타일이 보이는 동안 매 스텝 '제대로(변화 후 재질로) 그렸는가'를 추적한다.
  leave_vis : 떠나는 동안 타일이 아직 보일 때 맞게 그린 비율 (가장자리로 밀려가는 동안 유지하는가)
  last_vis  : 시야에서 사라지기 직전 프레임에서 맞게 그렸는가
  back      : 다시 보인 순간 맞게 그렸는가 (재방문 지표)
위치(타일이 시야 어디에 있는지)별로도 나눈다.
"""
import collections
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.env import CrafterEnv, tile_patch, world_to_tile  # noqa: E402
from src.eval_revisit import load_model, load_scenarios, run_batch  # noqa: E402
from src.tiles import TileClassifier  # noqa: E402

clf = TileClassifier(CrafterEnv(seed=0))
dev = torch.device('cuda')
scs = [s for s in load_scenarios('data/scenarios') if str(s['kind']) in ('place_stone', 'place_table') and int(s['N']) <= 4]
for s in scs:
  s['start_idx'] = int(s['change_idx'])
for spec in sys.argv[1:] or ['outputs/baseline/last.pt', 'outputs/baseline_v2/last.pt', 'outputs/memory_aligned_v2/last.pt']:
  model, name, *_ = load_model(spec, dev)
  stats = collections.defaultdict(list)
  edge = collections.defaultdict(list)
  for i in range(0, len(scs), 120):
    chunk = scs[i:i + 120]
    preds = run_batch(model, chunk, 16, dev)
    for s, pred in zip(chunk, preds):
      c0, lv, fb, N = int(s['change_idx']), int(s['last_visible_idx']), int(s['first_back_idx']), int(s['N'])
      def ok(idx):
        tt = world_to_tile(s['player_pos'][idx], s['tile'])
        return int(clf.classify_patches(tile_patch(pred[idx - c0 - 1], tt))[0]) == int(s['after']), tt
      for idx in range(c0 + 1, lv + 1):
        good, tt = ok(idx)
        stats[N, 'leave_vis'].append(good)
        # 타일이 시야 가장자리(가로 0/8 또는 세로 0/6)에 있는가
        edge['edge' if tt[0] in (0, 8) or tt[1] in (0, 6) else 'inner'].append(good)
      stats[N, 'last_vis'].append(ok(lv)[0] if lv > c0 else True)
      stats[N, 'back'].append(ok(fb)[0])
  print(f'{name}')
  for N in [1, 2, 3, 4]:
    print(f'  N={N}  떠나는 동안 {np.mean(stats[N, "leave_vis"]):.2f}  사라지기 직전 {np.mean(stats[N, "last_vis"]):.2f}  '
          f'다시 보인 순간 {np.mean(stats[N, "back"]):.2f}')
  print(f'  떠나는 동안 위치별: 안쪽 {np.mean(edge["inner"]):.2f} (n={len(edge["inner"])}), 가장자리 {np.mean(edge["edge"]):.2f} (n={len(edge["edge"])})')
