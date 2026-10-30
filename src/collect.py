"""Crafter 궤적 수집. 에피소드 하나 = .npz 파일 하나 (압축).

저장 형식 (T = 행동 수, 프레임은 T+1장):
  obs        uint8 [T+1, 72, 72, 3]   obs[t+1] = step(action[t]) 결과
  action     uint8 [T]
  reward     float32 [T], done bool [T]
  player_pos int16 [T+1, 2], facing int8 [T+1, 2]   (월드 좌표 [x, y])
  material   uint8 [T+1, 64, 64]  object에 가려지지 않은 정답 재질 ([x, y] 인덱싱)
  semantic   uint8 [T+1, 64, 64]  crafter semantic (object id 포함)
  inventory  uint8 [T+1, 16]      crafter.constants.items 순서
  policy     str, seed int
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time

import crafter
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.env import ACTIONS, CrafterEnv  # noqa: E402
from src.policies import POLICIES, random_inventory  # noqa: E402

ITEMS = list(crafter.constants.items)


class Recorder:

  def __init__(self, env):
    self.env = env
    self.buf = {k: [] for k in ['obs', 'action', 'reward', 'done', 'player_pos', 'facing',
                                'material', 'semantic', 'inventory']}
    self.finished = False

  def _state(self, obs):
    s = self.env.state()
    self.buf['obs'].append(obs)
    for k in ['player_pos', 'facing', 'material', 'semantic']:
      self.buf[k].append(s[k])
    self.buf['inventory'].append([self.env._player.inventory[k] for k in ITEMS])

  def reset(self, obs):
    self._state(obs)

  def __call__(self, action):
    obs, reward, done, info = self.env.step(action)
    self.buf['action'].append(action)
    self.buf['reward'].append(reward)
    self.buf['done'].append(done)
    self._state(obs)
    self.finished = done
    return obs, reward, done, info

  def arrays(self):
    dt = dict(obs=np.uint8, action=np.uint8, reward=np.float32, done=bool, player_pos=np.int16,
              facing=np.int8, material=np.uint8, semantic=np.uint8, inventory=np.uint8)
    return {k: np.asarray(v, dt[k]) for k, v in self.buf.items()}


def run_episode(args):
  ep, cfg = args
  seed = cfg['seed_base'] + ep
  rng = np.random.default_rng(seed)
  policy = 'scripted' if rng.random() < cfg['scripted_ratio'] else 'random'
  env = CrafterEnv(seed=seed, length=cfg['episode_length'])
  obs = env.reset()
  random_inventory(env, rng)
  obs = env.render()  # 인벤토리 지급이 화면(인벤토리 줄)에 반영되도록 다시 그림
  rec = Recorder(env)
  rec.reset(obs)
  POLICIES[policy](env, rng, rec, lambda: rec.finished)
  data = rec.arrays()
  path = pathlib.Path(cfg['out_dir']) / f'ep{ep:06d}.npz'
  np.savez_compressed(path, policy=policy, seed=seed, **data)
  ach = {k: int(v) for k, v in env._player.achievements.items() if v}
  return dict(ep=ep, policy=policy, length=len(data['action']),
              actions=np.bincount(data['action'], minlength=len(ACTIONS)).tolist(),
              changed_tiles=int((data['material'][1:] != data['material'][:-1]).any((1, 2)).sum()),
              achievements=ach)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--config', default='configs/data.yaml')
  p.add_argument('--frames', type=int, default=None, help='config 값 덮어쓰기 (디버그용)')
  p.add_argument('--out_dir', default=None)
  args = p.parse_args()
  cfg = yaml.safe_load(open(args.config))
  if args.frames:
    cfg['frames'] = args.frames
  if args.out_dir:
    cfg['out_dir'] = args.out_dir
  out = pathlib.Path(cfg['out_dir'])
  out.mkdir(parents=True, exist_ok=True)

  stats, frames, ep, t0 = [], 0, 0, time.time()
  with mp.Pool(cfg['workers']) as pool:
    while frames < cfg['frames']:
      # 남은 프레임을 평균 길이로 나눠 한 번에 제출 (마지막 배치에서 조금 넘칠 수 있음)
      avg = np.mean([s['length'] for s in stats]) if stats else cfg['episode_length']
      n = max(cfg['workers'], int(np.ceil((cfg['frames'] - frames) / avg)))
      for s in pool.imap_unordered(run_episode, [(i, cfg) for i in range(ep, ep + n)]):
        stats.append(s)
        frames += s['length']
      ep += n
      print(f'{frames} frames, {len(stats)} episodes, {time.time() - t0:.0f}s', flush=True)

  summarize(stats, out)


def summarize(stats, out):
  lengths = np.array([s['length'] for s in stats])
  acts = np.sum([s['actions'] for s in stats], 0)
  summary = {
      'episodes': len(stats), 'frames': int(lengths.sum()),
      'policy_episodes': {p: sum(s['policy'] == p for s in stats) for p in POLICIES},
      'length_mean': float(lengths.mean()), 'length_min': int(lengths.min()),
      'action_frac': {a: round(float(c / acts.sum()), 4) for a, c in zip(ACTIONS, acts)},
      'material_change_steps': {p: int(sum(s['changed_tiles'] for s in stats if s['policy'] == p))
                                for p in POLICIES},
      'achievement_totals': {},
  }
  for s in stats:
    for k, v in s['achievements'].items():
      if k.startswith(('place_', 'collect_')):
        summary['achievement_totals'][k] = summary['achievement_totals'].get(k, 0) + v
  (out / 'stats.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
  print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
  main()
