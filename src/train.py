"""학습 엔트리포인트. python src/train.py --config configs/baseline.yaml

loss = MSE(x̂, x) + l1_weight · L1(x̂, x)   (-1~1 공간)
- unroll U > 1: 첫 스텝은 실제 context, 이후 스텝은 직전 예측(detach)을 context에 넣는다.
  자기회귀 rollout에서 자기 오차가 쌓인 입력을 학습 때도 보게 해 붕괴를 늦춘다.
- ctx_noise: context 프레임에 N(0, σ²), σ ~ U(0, ctx_noise) 노이즈. 같은 목적의 더 싼 방법.
"""
import argparse
import csv
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.dataset import FrameDataset, list_episodes, split_episodes  # noqa: E402
from src.models.memory import CHANNELS  # noqa: E402
from src.models.memory_model import MemoryWorldModel  # noqa: E402
from src.models.unet import FrameContextModel, UNet  # noqa: E402
from src.rollout import rollout, save_side_by_side_gif, to_uint8  # noqa: E402


def device():
  if torch.cuda.is_available():
    return torch.device('cuda')
  if torch.backends.mps.is_available():
    return torch.device('mps')
  return torch.device('cpu')


def build_model(cfg):
  m = cfg['model']
  extra = CHANNELS[m['memory']['mode']] if m.get('memory') else 0
  return UNet(context=m['context'], channels=tuple(m['channels']), blocks=m['blocks'], cond_dim=m['cond_dim'],
              extra_in=extra)


def wrap_model(net, cfg, oracle_pos=None):
  """학습된 net → WorldModel (평가·데모 공용)."""
  m = cfg['model']
  if m.get('memory'):
    # 평가·데모 검색 규칙 기본값 = recent. v2·v3는 학습도 recent, v1 memory 모델만 overlap 규칙으로 학습했다.
    return MemoryWorldModel(net, m['memory']['mode'], oracle_pos=oracle_pos,
                            retrieval=m['memory'].get('eval_retrieval', 'recent'))
  return FrameContextModel(net)


@torch.no_grad()
def evaluate(net, val, cfg, dev, out_dir, step):
  net.eval()
  model = wrap_model(net, cfg)
  K, T = cfg['model']['context'], cfg['eval']['rollout_len']
  g = torch.Generator(device=dev).manual_seed(123)  # 매번 같은 검증 구간
  frames, actions = val.sample(cfg['eval']['num_rollouts'], K, T, generator=g)
  pred = rollout(model, frames[:, :K], actions)
  real = frames[:, K:]
  mse_t = ((pred - real) ** 2).mean((0, 2, 3, 4)).cpu().numpy()  # horizon별
  res = {f'rollout_mse@{h}': float(mse_t[h - 1]) for h in cfg['eval']['horizons'] if h <= T}
  # B0 (직전 실제 context 프레임 복사)도 같은 구간에서 계산해 기준선으로 남긴다
  b0 = ((frames[:, K - 1:K] - real) ** 2).mean((0, 2, 3, 4)).cpu().numpy()
  res.update({f'b0_mse@{h}': float(b0[h - 1]) for h in cfg['eval']['horizons'] if h <= T})
  save_side_by_side_gif(out_dir / f'rollout_{step:07d}.gif', to_uint8(real[0]), to_uint8(pred[0]))
  net.train()
  return res


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--config', required=True)
  p.add_argument('--resume', default=None)
  args = p.parse_args()
  cfg = yaml.safe_load(open(args.config))
  tc = cfg['train']
  torch.manual_seed(cfg['seed'])
  np.random.seed(cfg['seed'])
  dev = device()
  out_dir = pathlib.Path(cfg['out_dir'])
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True))

  tr_files, va_files = split_episodes(list_episodes(cfg['data_dir']))
  t0 = time.time()
  train = FrameDataset(tr_files, dev)
  val = FrameDataset(va_files, dev)
  mem_cfg = cfg['model'].get('memory')
  if mem_cfg:
    rule = mem_cfg.get('train_retrieval', 'overlap')  # v1 = overlap, v2 = recent (평가 규칙과 통일)
    suffix = '' if rule == 'overlap' else f'_{rule}'
    idx_path = pathlib.Path(cfg['data_dir']) / f"memory_index_K{cfg['model']['context']}{suffix}.npz"
    train.load_memory_index(idx_path)
  mix = tc.get('mix')  # v2: 풀별 배치 비율 + 바뀐 칸 가중치
  if mix:
    train.load_interact_index(pathlib.Path(cfg['data_dir']) / 'interact_index.npz')
  print(f'data: train {len(train)} frames / {len(tr_files)} eps, val {len(val)} / {len(va_files)} eps '
        f'({time.time() - t0:.0f}s)', flush=True)

  net = build_model(cfg).to(dev)
  print(f'params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M', flush=True)
  opt = torch.optim.AdamW(net.parameters(), lr=tc['lr'], weight_decay=tc['weight_decay'])
  sched = torch.optim.lr_scheduler.LambdaLR(
      opt, lambda s: min(1, (s + 1) / tc['warmup']) * 0.5 * (1 + math.cos(math.pi * min(1, s / tc['steps']))))
  step = 0
  if args.resume:
    ck = torch.load(args.resume, map_location=dev)
    net.load_state_dict(ck['net'])
    opt.load_state_dict(ck['opt'])
    sched.load_state_dict(ck['sched'])
    step = ck['step']

  K, U = cfg['model']['context'], tc['unroll']
  log_f = open(out_dir / 'log.csv', 'a', newline='')
  log = csv.writer(log_f)
  amp = dev.type == 'cuda'
  t0, run = time.time(), []
  while step < tc['steps']:
    extra = weight = None
    if mix:
      assert U == 1, '혼합 샘플링은 1-step만 지원'
      frames, actions, extra, weight = train.sample_mixed(
          tc['batch'], K, mix, mem_cfg['mode'] if mem_cfg else None, mem_cfg['drop'] if mem_cfg else 0.0,
          lam=tc.get('change_weight', 0.0), fail_target=tc.get('fail_target_weight', False))
    elif mem_cfg:
      assert U == 1, 'memory 학습은 1-step만 지원'
      frames, actions, extra = train.sample_memory(tc['batch'], K, mem_cfg['mode'], tuple(mem_cfg['mix']),
                                                   mem_cfg['drop'])
    else:
      frames, actions = train.sample(tc['batch'], K, U)
    ctx = frames[:, :K]
    if tc['ctx_noise'] > 0:
      sig = torch.rand(ctx.shape[0], 1, 1, 1, 1, device=dev) * tc['ctx_noise']
      ctx = ctx + sig * torch.randn_like(ctx)
    loss = 0.0
    for u in range(U):
      target = frames[:, K + u]
      with torch.autocast(dev.type, torch.bfloat16, enabled=amp):
        pred = net(ctx, actions[:, u], extra=extra).float()
      # 픽셀별 (MSE + L1)에 가중치 w = 1 + λ·(바뀐 칸) 를 곱해 평균. weight가 없으면 v1과 같다
      err = (pred - target) ** 2 + tc['l1_weight'] * (pred - target).abs()
      loss = loss + (err if weight is None else err * weight).mean() / U
      ctx = torch.cat([ctx[:, 1:], pred.detach().clamp(-1, 1)[:, None]], 1)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), tc['grad_clip'])
    opt.step()
    sched.step()
    step += 1
    run.append(loss.item() if step % 10 == 0 else None)

    if step % tc['log_every'] == 0:
      vals = [v for v in run if v is not None]
      rate = tc['log_every'] / (time.time() - t0)
      print(f'step {step} loss {np.mean(vals):.5f} gn {gn:.3f} lr {sched.get_last_lr()[0]:.2e} '
            f'{rate:.1f} it/s', flush=True)
      log.writerow(['train', step, float(np.mean(vals)), float(gn), time.time()])
      log_f.flush()
      run, t0 = [], time.time()
    if step % tc['eval_every'] == 0 or step == tc['steps']:
      res = evaluate(net, val, cfg, dev, out_dir, step)
      print(f'eval {step} ' + json.dumps({k: round(v, 5) for k, v in res.items()}), flush=True)
      log.writerow(['eval', step, json.dumps(res), '', time.time()])
      log_f.flush()
      t0 = time.time()
    if step % tc['ckpt_every'] == 0 or step == tc['steps']:
      ck = dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, cfg=cfg)
      torch.save(ck, out_dir / 'last.pt')


if __name__ == '__main__':
  main()
