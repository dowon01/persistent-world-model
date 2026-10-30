"""약점 1 분석: memory_v2가 실패한 설치를 성공한 것처럼 그린다 (place_stone 14%, place_table 30%).

검증 에피소드의 실패 장면을 실패 이유별로 나누고, 모델별 '바꿔 그림' 비율을 잰다.
  실패 이유: no_material (인벤토리 부족) / bad_target (놓을 수 없는 재질) / object (칸에 소·좀비 등) / other
memory 모델은 memory 입력을 끈 버전(memory_off)도 함께 재서, memory 입력이 원인인지 본다.
python scripts/analyze_false_change.py outputs/baseline_v2/last.pt outputs/memory_aligned_v3/last.pt
"""
import collections
import pathlib
import sys

import crafter
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.collect import ITEMS  # noqa: E402
from src.dataset import FrameDataset, list_episodes, split_episodes  # noqa: E402
from src.env import A, ACTIONS, CrafterEnv, tile_patch  # noqa: E402
from src.eval_revisit import load_model  # noqa: E402
from src.models import memory as M  # noqa: E402
from src.rollout import to_uint8  # noqa: E402
from src.tiles import TileClassifier  # noqa: E402

env0 = CrafterEnv(seed=0)
clf = TileClassifier(env0)
names = env0.material_names()
PLACE = crafter.constants.place
ACTS = ['place_stone', 'place_table']


def reason(a, inv, tgt_mat, has_obj):
  info = PLACE[a.replace('place_', '')]
  if has_obj:
    return 'object'
  if any(inv[ITEMS.index(k)] < v for k, v in info['uses'].items()):
    return 'no_material'
  if names[tgt_mat] not in info['where']:
    return 'bad_target'
  return 'other'


_, val = split_episodes(list_episodes('data/train'))
samples = collections.defaultdict(list)
for f in val:
  z = np.load(f)
  act, mat, sem, pos, fac, inv = (z['action'], z['material'], z['semantic'], z['player_pos'].astype(int),
                                  z['facing'].astype(int), z['inventory'])
  for t in range(15, len(act)):
    a = ACTIONS[act[t]]
    if a not in ACTS:
      continue
    tgt = tuple(pos[t] + fac[t])
    if not (0 <= tgt[0] < 64 and 0 <= tgt[1] < 64) or mat[t + 1][tgt] != mat[t][tgt]:
      continue
    r = reason(a, inv[t], int(mat[t][tgt]), sem[t][tgt] != mat[t][tgt])
    samples[a, r].append(dict(ctx=z['obs'][t - 15:t + 1], a=int(act[t]), tile=(4 + fac[t][0], 3 + fac[t][1]),
                              before=int(mat[t][tgt])))
print('실패 장면 수:', {f'{a}/{r}': len(v) for (a, r), v in sorted(samples.items())})

dev = torch.device('cuda')
orig_read = M.FrameMemory.read


def empty_read(self, query, current=None):
  B = query.shape[0]
  z = torch.zeros(B, 3, 72, 72, device=query.device)
  return M.make_condition(self.mode, z, query, query, torch.zeros(B, device=query.device),
                          torch.zeros(B, dtype=torch.bool, device=query.device))


specs = sys.argv[1:] or ['outputs/baseline_v2/last.pt', 'outputs/memory_aligned_v2/last.pt']
variants = [(pathlib.Path(sp).parent.name, sp, False) for sp in specs]
variants += [(pathlib.Path(sp).parent.name + ' (memory 끔)', sp, True) for sp in specs if 'memory' in sp]
for label, spec, off in variants:
  M.FrameMemory.read = empty_read if off else orig_read
  model, *_ = load_model(spec, dev)
  out = {}
  drawn = collections.Counter()
  for (a, r), ss in sorted(samples.items()):
    false = []
    for i in range(0, len(ss), 64):
      b = ss[i:i + 64]
      st = model.init_context(FrameDataset.to_float(torch.from_numpy(np.stack([x['ctx'] for x in b])).to(dev)))
      o, _ = model.step(st, torch.tensor([x['a'] for x in b], device=dev))
      for img, x in zip(to_uint8(o), b):
        m = int(clf.classify_patches(tile_patch(img, x['tile']))[0])
        false.append(m != x['before'])
        if m != x['before']:
          drawn[a, names[m]] += 1
    out[a, r] = (np.mean(false), len(ss))
  print(f'\n{label}')
  for (a, r), (v, n) in out.items():
    print(f'  {a:12s} {r:12s} 바꿔 그림 {v:.2f} (n={n})')
  print('  바꿔 그린 재질:', dict(drawn))
M.FrameMemory.read = orig_read
