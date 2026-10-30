"""재방문 평가 시나리오 생성.

시나리오 하나 = 실제 env에서 실행·검증된 행동 시퀀스와 그 정답.
  1. warm-up: 무작위 이동 (context로 쓸 실제 프레임 확보)
  2. 변화: 바라보는 타일에 place_stone / place_table / 나무 채집 / 돌 채집. control은 같은 자리에서 noop
  3. 이탈: 변화한 타일이 시야 밖으로 나갈 때까지 한 방향으로 이동
  4. 체류: 시야 밖에 머무는 동안 무작위 이동 (타일이 다시 보이지 않는 이동만 고른다)
  5. 복귀: 성공한 이동을 역순으로 되돌려 변화 직후의 위치로 돌아온다

N = 변경 타일이 연속으로 시야 밖에 있던 프레임 수 (결정 D4).
  이탈이 끝난 프레임(1) + 체류 스텝 D + 체류 이동의 되돌리기 S = N 이 되도록 체류를 조절한다.
  (되돌리기 경로는 체류 때 지나간 위치를 다시 지나므로 그동안도 시야 밖이다.)

검증 (하나라도 실패하면 버린다): 플레이어 생존, 복귀 위치 = 변화 직후 위치,
  측정 시점에 타일 재질 = 기대 재질, 실제로 센 시야 밖 프레임 수 = N.

평가 rollout의 기본 정의: 모델 context = 변화 직후 프레임까지의 실제 K프레임,
  그 뒤 행동은 모두 모델이 생성. 이렇게 하면 "변화를 그릴 수 있나"가 아니라 "변화를 기억하나"만 잰다.
  전체 프레임과 인덱스를 저장하므로 다른 정의(처음부터 rollout)도 나중에 계산할 수 있다.
"""
import argparse
import multiprocessing as mp
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.env import A, CrafterEnv, MOVES, OPPOSITE, VIEW, in_view  # noqa: E402

TYPES = ['place_stone', 'place_table', 'collect_tree', 'control']
NS = [1, 2, 3, 4, 8, 16, 32, 64]  # 1~3: K=4 context 창 경계 근처
SEED_BASE = 1_000_000  # 학습 데이터(seed 0~)와 겹치지 않게
PLACEABLE = {'grass', 'sand', 'path'}
WALKABLE = {'grass', 'sand', 'path'}  # lava는 player가 걸을 수 있지만 죽으므로 제외


class Failed(Exception):
  pass


class Runner:
  """env를 한 스텝씩 진행하며 프레임·행동·위치를 기록한다."""

  def __init__(self, env, obs):
    self.env = env
    self.frames, self.actions, self.pos = [obs], [], [env.player_pos]

  def step(self, a):
    obs, _, done, _ = self.env.step(int(a))
    self.frames.append(obs)
    self.actions.append(int(a))
    self.pos.append(self.env.player_pos)
    if done:
      raise Failed('died')
    return obs

  def material(self, xy):
    return self.env._world[tuple(xy)][0]

  def free(self, xy):
    mat, obj = self.env._world[tuple(xy)]
    return mat in WALKABLE and obj is None

  def safe_move(self, rng, options=None):
    """lava/막힌 칸이 아닌 이동 중 하나. 없으면 noop."""
    p = self.env.player_pos
    opts = [a for a in (options or list(MOVES)) if self.free(p + MOVES[a])]
    return self.step(rng.choice(opts) if opts else A['noop'])


def find_change(run, kind, rng):
  """현재 위치에서 kind 변화를 만들 수 있게 방향을 맞춘다. 반환: 변경 타일 좌표, 기대 재질 (전, 후)."""
  for _ in range(40):
    p = run.env.player_pos
    for a in rng.permutation(list(MOVES)):
      d = np.array(MOVES[a])
      if kind in ('place_stone', 'place_table', 'control'):
        # 한 칸 이동하면 facing = d, 대상 = p + 2d. 이동 칸은 비어 있고 대상은 설치 가능해야 함
        tgt = p + 2 * d
        if run.free(p + d) and run.material(tgt) in PLACEABLE and run.env._world[tuple(tgt)][1] is None:
          run.step(a)
          return tgt, run.material(tgt)
      elif kind == 'collect_tree':
        tgt = p + d
        if run.material(tgt) == 'tree':
          run.step(a)  # 나무는 못 지나가므로 이동은 실패하고 facing만 바뀜
          return tgt, 'tree'
    run.safe_move(rng)
  raise Failed('no target')


def generate(args):
  kind, N, idx, K = args
  seed = SEED_BASE + hash((kind, N, idx)) % 10_000_000
  rng = np.random.default_rng(seed)
  env = CrafterEnv(seed=seed, length=10000)
  env.reset()
  env.set_inventory(stone=9, wood=9, sapling=3, wood_pickaxe=1, stone_pickaxe=1)
  run = Runner(env, env.render())
  try:
    for _ in range(K + int(rng.integers(0, 6))):  # warm-up
      run.safe_move(rng)
    tile, before = find_change(run, kind, rng)
    after = {'place_stone': 'stone', 'place_table': 'table', 'collect_tree': 'grass', 'control': before}[kind]
    run.step({'place_stone': A['place_stone'], 'place_table': A['place_table'],
              'collect_tree': A['do'], 'control': A['noop']}[kind])
    if run.material(tile) != after:
      raise Failed('change failed')
    change_idx = len(run.frames) - 1  # 변화가 반영된 첫 프레임
    home = env.player_pos

    # 이탈: 한 방향으로 시야 밖까지
    away = rng.permutation(list(MOVES))
    path = []
    for a in away:
      path = []
      ok = True
      while in_view(env.player_pos, tile):
        p = env.player_pos
        nxt = p + MOVES[a]
        if not run.free(nxt) or np.array_equal(nxt, tile):  # 변경 타일은 밟지 않는다
          ok = False
          break
        run.step(a)
        path.append(a)
      if ok:
        break
      for b in reversed(path):  # 막혔으면 되돌아와서 다른 방향
        run.step(OPPOSITE[int(b)])
      if not np.array_equal(env.player_pos, home):
        raise Failed('leave reset')
    if in_view(env.player_pos, tile):
      raise Failed('cannot leave')
    last_visible_idx = len(run.frames) - 2

    # 체류: 1 + D + S = N
    D = S = 0
    dwell = []
    while 1 + D + S < N:
      p = env.player_pos
      opts = [a for a in MOVES if run.free(p + MOVES[a]) and not in_view(p + MOVES[a], tile)]
      if 1 + D + S + 2 <= N and opts and rng.random() < 0.8:
        a = rng.choice(opts)
        run.step(a)
        moved = not np.array_equal(env.player_pos, p)
        D += 1
        if moved:
          S += 1
          dwell.append(a)
      else:
        run.step(A['noop'])
        D += 1

    # 복귀
    for b in reversed(path + dwell):  # 체류 이동을 먼저, 이탈 경로를 나중에 되돌린다
      run.step(OPPOSITE[int(b)])
    if not np.array_equal(env.player_pos, home):
      raise Failed('return blocked')
    final_idx = len(run.frames) - 1
  except Failed as e:
    return None, str(e)

  pos = np.array(run.pos)
  vis = np.array([in_view(q, tile) for q in pos])
  first_back = change_idx + int(np.flatnonzero(~vis[change_idx:])[0]) + N
  n_out = int((~vis[change_idx:]).sum())
  if n_out != N or not vis[first_back] or vis[first_back - 1]:
    return None, f'N mismatch {n_out}'
  if run.material(tile) != after:
    return None, 'tile changed'
  mats = {n: i for i, n in env.material_names().items()}
  return dict(
      kind=kind, N=N, seed=seed, K=K,
      frames=np.array(run.frames, np.uint8), actions=np.array(run.actions, np.uint8),
      player_pos=pos.astype(np.int16), tile=np.array(tile, np.int16),
      before=mats[before], after=mats[after],
      change_idx=change_idx, last_visible_idx=last_visible_idx, first_back_idx=first_back, final_idx=final_idx,
      material_final=env.material_map(), semantic_final=env.semantic(),
  ), 'ok'


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--out_dir', default='data/scenarios')
  p.add_argument('--per_cell', type=int, default=60, help='(종류, N) 조합당 시나리오 수')
  p.add_argument('--context', type=int, default=16, help='warm-up 길이 (K=16 모델까지 쓸 수 있게 변화 전 실제 프레임을 16장 이상 확보)')
  p.add_argument('--workers', type=int, default=12)
  args = p.parse_args()
  out = pathlib.Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  report = {}
  with mp.Pool(args.workers) as pool:
    for kind in TYPES:
      for N in NS:
        got, idx, reasons = 0, 0, {}
        while got < args.per_cell and idx < args.per_cell * 20:
          jobs = [(kind, N, i, args.context) for i in range(idx, idx + args.per_cell * 2)]
          idx += len(jobs)
          for sc, why in pool.imap(generate, jobs):
            reasons[why] = reasons.get(why, 0) + 1
            if sc is not None and got < args.per_cell:
              np.savez_compressed(out / f'{kind}_N{N:02d}_{got:03d}.npz', **sc)
              got += 1
        report[kind, N] = (got, reasons)
        print(kind, N, got, reasons, flush=True)


if __name__ == '__main__':
  main()
