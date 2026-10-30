"""긴 N에서 추정 위치가 왜 틀어지는지 분석 (N=64에서 유지율이 추정 위치 0.86, 실제 위치 0.97로 차이 나는 이유).

memory 모델을 재방문 시나리오에서 rollout하면서 매 스텝 추정 위치를 실제 위치(상대 좌표)와 비교한다.
  - back 시점에 위치가 틀렸는가, 처음 틀어진 스텝과 그때의 행동
  - 오류 종류: 놓친 이동(실제로 움직였는데 제자리로 판정) / 없는 이동(실제로 막혔는데 움직였다고 판정)
  - 처음 틀어진 스텝의 원인 후보: 모델이 그린 프레임에서 '제자리'와 '이동' 오차 차이가 작았는가 (애매한 화면),
    같은 판정을 실제 프레임에 했어도 틀렸는가 (판정 규칙 자체의 한계)
  - 위치가 맞은 시나리오와 틀린 시나리오의 keep 비교
python scripts/analyze_position.py --model outputs/memory_aligned/last.pt
"""
import argparse
import collections
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.dataset import FrameDataset  # noqa: E402
from src.env import MOVES, CrafterEnv, tile_patch, world_to_tile  # noqa: E402
from src.eval_revisit import load_model, load_scenarios  # noqa: E402
from src.models.memory_model import tile_errors  # noqa: E402
from src.rollout import to_uint8  # noqa: E402
from src.tiles import TileClassifier  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--model', default='outputs/memory_aligned/last.pt')
p.add_argument('--kinds', default='place_stone,place_table')
p.add_argument('--context', type=int, default=16)
args = p.parse_args()
dev = torch.device('cuda')
model, *_ = load_model(args.model, dev)
clf = TileClassifier(CrafterEnv(seed=0))
scs = [s for s in load_scenarios('data/scenarios') if str(s['kind']) in args.kinds.split(',')]
K = args.context

rows = []
for i in range(0, len(scs), 100):
  chunk = scs[i:i + 100]
  T = max(int(s['final_idx'] - s['change_idx']) for s in chunk)
  ctx = np.stack([s['frames'][s['change_idx'] - K + 1:s['change_idx'] + 1] for s in chunk])
  acts = np.zeros((len(chunk), T), np.int64)
  for j, s in enumerate(chunk):
    a = s['actions'][s['change_idx']:s['final_idx']]
    acts[j, :len(a)] = a
  frames0 = FrameDataset.to_float(torch.from_numpy(ctx).to(dev))
  state = model.init_context(frames0)
  est = [state['pos'].cpu().numpy()]
  prev = frames0[:, -1]
  gaps, preds = [], []
  acts_t = torch.from_numpy(acts).to(dev)
  for t in range(T):
    nxt, state = model.step(state, acts_t[:, t])
    # 이동 행동일 때 '제자리 - 이동' 오차 차이 (작을수록 화면이 애매)
    g = torch.full((len(chunk),), float('nan'), device=dev)
    for a, d in MOVES.items():
      sel = acts_t[:, t] == a
      if sel.any():
        g[sel] = tile_errors(prev[sel], nxt[sel], (0, 0)) - tile_errors(prev[sel], nxt[sel], d)
    gaps.append(g.cpu().numpy())
    est.append(state['pos'].cpu().numpy())
    preds.append(to_uint8(nxt))
    prev = nxt
  est, gaps, preds = np.stack(est, 1), np.stack(gaps, 1), np.stack(preds, 1)
  for j, s in enumerate(chunk):
    c0, fb = int(s['change_idx']), int(s['first_back_idx'])
    true = s['player_pos'][c0:int(s['final_idx']) + 1].astype(int)
    e = est[j, :len(true)] - est[j, 0] + true[0]     # 추정 위치를 변화 직후 실제 위치에 맞춰 비교
    # context 안(실제 프레임, 행동 없음)에서 이미 틀린 경우도 따로 센다
    ctx_true = s['player_pos'][c0 - K + 1:c0 + 1].astype(int)
    ctx_ok = True  # init_context의 누적 추정은 상대 좌표라 context 끝까지 정확했는지만 본다
    wrong = np.any(e != true, 1)
    first = int(np.argmax(wrong)) if wrong.any() else -1
    kind = ''
    real_gap = np.nan
    if first > 0:
      a = int(s['actions'][c0 + first - 1])
      moved_true = np.any(true[first] != true[first - 1])
      kind = 'missed_move' if moved_true else 'phantom_move'
      if a in MOVES:  # 같은 판정을 실제 프레임에 했을 때의 오차 차이
        fr = FrameDataset.to_float(torch.from_numpy(s['frames'][c0 + first - 1:c0 + first + 1]).to(dev))
        real_gap = float((tile_errors(fr[:1], fr[1:], (0, 0)) - tile_errors(fr[:1], fr[1:], MOVES[a]))[0])
    # back 시점 keep (재판정)
    pf = preds[j, fb - c0 - 1]
    tt = world_to_tile(s['player_pos'][fb], s['tile'])
    keep = int(clf.classify_patches(tile_patch(pf, tt))[0]) == int(s['after'])
    rows.append(dict(N=int(s['N']), wrong_back=bool(wrong[fb - c0]), wrong_any=bool(wrong.any()), first=first,
                     kind=kind, gap=float(gaps[j, first - 1]) if first > 0 else np.nan, real_gap=real_gap,
                     keep=keep, steps=len(true) - 1))

by = collections.defaultdict(list)
for r in rows:
  by[r['N']].append(r)
print('N   시나리오  back에서 위치 틀림  keep(위치 맞음)  keep(위치 틀림)  평균 rollout 길이')
for N in sorted(by):
  rs = by[N]
  ok = [r['keep'] for r in rs if not r['wrong_back']]
  bad = [r['keep'] for r in rs if r['wrong_back']]
  print(f'{N:2d}  {len(rs):6d}    {np.mean([r["wrong_back"] for r in rs]):.2f}            '
        f'{np.mean(ok) if ok else float("nan"):.2f} (n={len(ok)})     {np.mean(bad) if bad else float("nan"):.2f} (n={len(bad)})'
        f'      {np.mean([r["steps"] for r in rs]):.0f}')
errs = [r for r in rows if r['first'] > 0]
print(f'\n처음 틀어진 스텝 {len(errs)}건: ', dict(collections.Counter(r['kind'] for r in errs)))
g = np.array([r['gap'] for r in errs])
rg = np.array([r['real_gap'] for r in errs])
print(f'  모델 프레임의 (제자리 - 이동) 오차 차이: 중앙값 {np.nanmedian(g):.4f}  (판정 margin 0.001, 음수면 제자리로 판정)')
print(f'  같은 스텝 실제 프레임의 차이: 중앙값 {np.nanmedian(rg):.4f}, 실제 프레임으로도 틀렸을 비율 '
      f'{np.mean([(x < -1e-3) == (r["kind"] == "missed_move") for x, r in zip(rg, errs) if not np.isnan(x)]):.2f}')
print('  처음 틀어진 시점 (rollout 몇 스텝째) 분포:', np.percentile([r['first'] for r in errs], [10, 50, 90]).round())
