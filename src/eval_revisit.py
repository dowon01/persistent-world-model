"""재방문 persistence 평가. 모든 Phase, 모든 WorldModel 공용.

rollout: context = 변화 직후 프레임까지의 실제 K프레임, 이후 행동은 모두 모델이 생성 (scenarios.py 참고).

측정 시점 (두 가지 모두 기록):
  back  = 변경 타일이 처음 다시 보이는 프레임 (주 지표). 타일은 시야 가장자리에 있고,
          화면이 어떤 context 프레임과도 달라서 직전 프레임 복사(B0) 같은 지름길이 통하지 않는다.
  final = 변화 직후 위치로 완전히 돌아온 프레임. 화면이 변화 직후 프레임과 같아서 B0가 거의 만점이 된다.
          → B0는 back에서만 하한 기준선 의미가 있다.

지표 (타일 판정은 src/tiles.py 템플릿 매칭):
  keep      변경 타일의 판정 재질 == 정답(변화 후) 재질. control은 원래 재질이 유지됐는지
  revert    변경 타일이 변화 전 재질로 그려짐 (control 제외). "잊고 원래대로 그림"
  bg_acc    나머지 월드 뷰 타일 정확도 (object 타일·플레이어 타일 제외). 장면 전체가 어긋났는지
  self_con  떠나기 직전 모델이 그린 타일 재질 == 돌아왔을 때 모델이 그린 재질 (정답 불필요, Phase 4용)
  patch_mse 변경 타일 8×8 패치의 실제 프레임 대비 MSE (0~1 스케일)

결과는 outputs/results.csv에 시나리오 단위로 누적한다.
"""
import argparse
import csv
import glob
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.dataset import FrameDataset  # noqa: E402
from src.env import CrafterEnv, VIEW, tile_patch, world_to_tile  # noqa: E402
from src.models.interface import CopyLastFrame  # noqa: E402
from src.rollout import save_side_by_side_gif, to_uint8  # noqa: E402
from src.tiles import TileClassifier, view_ground_truth  # noqa: E402

FIELDS = ['model', 'config', 'seed', 'kind', 'N', 'scenario', 'point', 'keep', 'revert', 'bg_acc',
          'self_con', 'patch_mse', 'pred_mat', 'gt_mat', 'time']


def load_model(spec, dev):
  """'b0' 또는 체크포인트 경로 → (WorldModel, 이름, config 이름, seed)."""
  if spec == 'b0':
    return CopyLastFrame(), 'b0', '-', 0
  from src.train import build_model, wrap_model
  ck = torch.load(spec, map_location=dev)
  net = build_model(ck['cfg']).to(dev).eval()
  net.load_state_dict(ck['net'])
  return wrap_model(net, ck['cfg']), ck['cfg']['name'], f"{ck['cfg']['name']}@{ck['step']}", ck['cfg']['seed']


def load_scenarios(d):
  out = []
  for f in sorted(glob.glob(f'{d}/*.npz')):
    z = dict(np.load(f))
    z['name'] = pathlib.Path(f).stem
    out.append(z)
  return out


@torch.no_grad()
def run_batch(model, scs, K, dev):
  """시나리오 묶음 rollout. 반환: 각 시나리오의 예측 프레임 uint8 [T, H, W, 3] (index 0 = start_idx+1)."""
  T = max(int(s['final_idx'] - s['start_idx']) for s in scs)
  ctx = np.stack([s['frames'][s['start_idx'] - K + 1:s['start_idx'] + 1] for s in scs])
  acts = np.zeros((len(scs), T), np.int64)
  for i, s in enumerate(scs):
    a = s['actions'][s['start_idx']:s['final_idx']]
    acts[i, :len(a)] = a
  if getattr(model, 'use_oracle', False):  # 상한선 분석: 실제 위치 (context 첫 프레임부터)
    pos = np.zeros((len(scs), K + T, 2), np.int64)
    for i, s in enumerate(scs):
      p = s['player_pos'][s['start_idx'] - K + 1:s['final_idx'] + 1].astype(np.int64)
      pos[i, :len(p)], pos[i, len(p):] = p, p[-1]
    model.oracle_pos = torch.from_numpy(pos).to(dev)
  state = model.init_context(FrameDataset.to_float(torch.from_numpy(ctx).to(dev)))
  acts = torch.from_numpy(acts).to(dev)
  preds = []
  for t in range(T):
    frame, state = model.step(state, acts[:, t])
    preds.append(to_uint8(frame))
  preds = np.stack(preds, 1)
  return [p[:int(s['final_idx'] - s['start_idx'])] for p, s in zip(preds, scs)]


def score(s, pred, clf):
  """시나리오 하나 → 측정 시점별 지표 dict."""
  c0 = int(s['start_idx'])  # 이 프레임까지 실제, 이후는 모델 출력
  frame_at = lambda i: s['frames'][i] if i <= c0 else pred[i - c0 - 1]  # noqa: E731
  tile = s['tile']
  rows = {}
  # 떠나기 직전 모델이 그린 타일 (last_visible이 context 안이면 실제 프레임)
  lv = int(s['last_visible_idx'])
  lv_tile = world_to_tile(s['player_pos'][lv], tile)
  lv_mat = int(clf.classify_patches(tile_patch(frame_at(lv), lv_tile))[0])
  for point, idx in [('back', int(s['first_back_idx'])), ('final', int(s['final_idx']))]:
    f = frame_at(idx)
    real = s['frames'][idx]
    tt = world_to_tile(s['player_pos'][idx], tile)
    pm = int(clf.classify_patches(tile_patch(f, tt))[0])
    pos = s['player_pos'][idx]
    if point == 'final':
      gt, mask = view_ground_truth(s['material_final'], s['semantic_final'], pos)
    else:  # back 시점 정답 재질은 실제 프레임을 템플릿으로 판정 (실제 프레임 정확도 100% 확인됨, check_env.py)
      gt, d = clf.classify_frame(real)
      mask = d < 1.0  # object가 겹친 타일은 템플릿과 안 맞음 → 제외
    mask = mask.copy()
    mask[tt[1], tt[0]] = False
    mask[VIEW[1] // 2, VIEW[0] // 2] = False  # 플레이어 타일
    pred_view, _ = clf.classify_frame(f)
    rows[point] = dict(
        keep=float(pm == int(s['after'])),
        revert=float(pm == int(s['before'])) if s['kind'] != 'control' else float('nan'),
        bg_acc=float((pred_view == gt)[mask].mean()) if mask.any() else float('nan'),
        self_con=float(pm == lv_mat),
        patch_mse=float(((tile_patch(f, tt) / 255.0 - tile_patch(real, tt) / 255.0) ** 2).mean()),
        pred_mat=pm, gt_mat=int(s['after']))
  return rows


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--model', required=True, help="'b0' 또는 체크포인트 경로")
  p.add_argument('--scenarios', default='data/scenarios')
  p.add_argument('--context', type=int, default=16, help='넘겨줄 실제 context 프레임 수. 모델은 이 중 마지막 K장만 쓴다')
  p.add_argument('--batch', type=int, default=200)
  p.add_argument('--results', default='outputs/results.csv')
  p.add_argument('--gif_dir', default=None, help='실패/성공 예시 GIF 저장 위치')
  p.add_argument('--oracle_pos', action='store_true', help='memory 모델에 실제 위치를 준다 (상한선 분석용)')
  p.add_argument('--retrieval', default=None, help="memory 검색 규칙 덮어쓰기 (예: recent, 진단용)")
  p.add_argument('--start', default='after', choices=['after', 'before'],
                 help='after (주 지표): 변화 직후 프레임까지 실제. before (보조): 변화 직전에서 끊어 설치·채집도 모델이 그린다')
  args = p.parse_args()
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model, name, cfg_name, seed = load_model(args.model, dev)
  if args.oracle_pos:
    assert hasattr(model, 'use_oracle'), 'oracle_pos는 memory 모델에만 의미가 있다'
    model.use_oracle = True
    name, cfg_name = name + '+oracle_pos', cfg_name + '+oracle_pos'
  if args.retrieval:
    model.retrieval = args.retrieval
    name, cfg_name = name + f'+ret_{args.retrieval}', cfg_name + f'+ret_{args.retrieval}'
  clf = TileClassifier(CrafterEnv(seed=0))
  scs = load_scenarios(args.scenarios)
  for s in scs:
    s['start_idx'] = int(s['change_idx']) - (1 if args.start == 'before' else 0)
  if args.start == 'before':
    name, cfg_name = name + '+start_before', cfg_name + '+start_before'
  scs.sort(key=lambda s: int(s['final_idx'] - s['start_idx']))  # 길이가 비슷한 것끼리 묶기
  new_file = not pathlib.Path(args.results).exists()
  f = open(args.results, 'a', newline='')
  w = csv.DictWriter(f, FIELDS)
  if new_file:
    w.writeheader()
  agg = {}
  gifs = {}
  t0 = time.time()
  for i in range(0, len(scs), args.batch):
    chunk = scs[i:i + args.batch]
    preds = run_batch(model, chunk, args.context, dev)
    for s, pred in zip(chunk, preds):
      for point, r in score(s, pred, clf).items():
        w.writerow(dict(model=name, config=cfg_name, seed=seed, kind=str(s['kind']), N=int(s['N']),
                        scenario=s['name'], point=point, time=int(time.time()), **r))
        agg.setdefault((point, str(s['kind']), int(s['N'])), []).append(r)
      key = (str(s['kind']), int(s['N']))
      if args.gif_dir and key not in gifs and s['kind'] != 'control':
        gifs[key] = (s, pred)
  f.close()
  print(f'{name} ({cfg_name}) {len(scs)} scenarios, {time.time() - t0:.0f}s')
  for point in ['back', 'final']:
    print(f'[{point}] kind    N  keep  revert bg_acc self_con')
    for (pt, kind, N), rs in sorted(agg.items()):
      if pt != point:
        continue
      m = {k: np.nanmean([r[k] for r in rs]) for k in ['keep', 'revert', 'bg_acc', 'self_con']}
      print(f'  {kind:13s} {N:2d} {m["keep"]:.3f} {m["revert"]:.3f} {m["bg_acc"]:.3f} {m["self_con"]:.3f}')
  if args.gif_dir:
    out = pathlib.Path(args.gif_dir)
    out.mkdir(parents=True, exist_ok=True)
    for (kind, N), (s, pred) in gifs.items():
      c0 = int(s['start_idx'])
      real = s['frames'][c0 + 1:int(s['final_idx']) + 1]
      save_side_by_side_gif(out / f'{name}_{kind}_N{N:02d}.gif', real, pred, fps=6)


if __name__ == '__main__':
  main()
