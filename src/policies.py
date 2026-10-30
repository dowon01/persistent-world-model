"""수집용 정책. scenarios.py와 이동 유틸을 공유한다.

- random: sleep을 뺀 16개 행동 균등 샘플링
- scripted: 이동 → 상호작용(do / place_*) → 한 방향으로 멀리 다녀오기(excursion) → 같은 경로로 복귀
  "변화를 만들고 떠났다 돌아오는" 장면을 학습 데이터에 충분히 넣기 위함.
"""
import numpy as np

from src.env import A, ACTIONS, MOVES, OPPOSITE

NON_SLEEP = [i for i, a in enumerate(ACTIONS) if a != 'sleep']
MOVE_ACTS = list(MOVES)
INTERACT = {  # 상호작용 행동 분포
    A['do']: 0.45, A['place_stone']: 0.25, A['place_table']: 0.15,
    A['place_plant']: 0.05, A['make_wood_pickaxe']: 0.05, A['make_stone_pickaxe']: 0.05,
}


def random_inventory(env, rng):
  """에피소드 시작 시 인벤토리 무작위 지급 (랜덤 정책으로는 돌 채집·설치에 거의 도달 못 함)."""
  env.set_inventory(
      wood=int(rng.integers(0, 10)), stone=int(rng.integers(0, 10)), sapling=int(rng.integers(0, 4)),
      wood_pickaxe=int(rng.random() < 0.6), stone_pickaxe=int(rng.random() < 0.3))


def refill(env, rng):
  """설치 재료가 바닥나면 보충. 화면 아래 인벤토리 줄만 바뀌고 월드 뷰에는 영향 없다."""
  inv = env._player.inventory
  if inv['stone'] == 0 and rng.random() < 0.8:
    inv['stone'] = int(rng.integers(1, 10))
  if inv['wood'] < 2 and rng.random() < 0.5:
    inv['wood'] = int(rng.integers(2, 10))
  if inv['sapling'] == 0 and rng.random() < 0.3:
    inv['sapling'] = int(rng.integers(1, 4))


class Mover:
  """env.step을 감싸 이동 성공 여부를 기록한다."""

  def __init__(self, env, record):
    self.env = env
    self.record = record  # (action) -> (obs, reward, done, info); 수집기가 프레임을 저장하는 콜백

  def step(self, action):
    before = self.env.player_pos
    out = self.record(int(action))
    moved = not np.array_equal(before, self.env.player_pos)
    return out, moved


def random_episode(env, rng, record, done_fn):
  t = 0
  while not done_fn():
    if t % 50 == 0:
      refill(env, rng)
    record(int(rng.choice(NON_SLEEP)))
    t += 1


def scripted_episode(env, rng, record, done_fn):
  mover = Mover(env, record)
  while not done_fn():
    refill(env, rng)
    # 1) 잠깐 돌아다니기
    for _ in range(int(rng.integers(1, 7))):
      if done_fn():
        return
      mover.step(rng.choice(MOVE_ACTS))
    # 2) 바라보는 타일에 상호작용 (직전 이동 방향이 facing)
    for _ in range(int(rng.integers(1, 4))):
      if done_fn():
        return
      acts, probs = zip(*INTERACT.items())
      mover.step(rng.choice(acts, p=np.array(probs) / sum(probs)))
    # 3) excursion: 한 방향 위주로 멀리 이동 → 머물기 → 성공한 이동을 역순으로 되돌리기
    main = rng.choice(MOVE_ACTS)
    path = []
    for _ in range(int(rng.integers(3, 16))):
      if done_fn():
        return
      a = main if rng.random() < 0.75 else rng.choice(MOVE_ACTS)
      _, moved = mover.step(a)
      if moved:
        path.append(a)
    dwell = rng.integers(0, 12) if rng.random() < 0.7 else rng.integers(12, 48)  # 가끔 오래 머문다 (N=32, 64용)
    for _ in range(int(dwell)):
      if done_fn():
        return
      a = rng.choice(MOVE_ACTS + [A['noop'], A['do']])
      _, moved = mover.step(a)
      if moved:
        path.append(a)
    for a in reversed(path):
      if done_fn():
        return
      mover.step(OPPOSITE[int(a)])


POLICIES = {'random': random_episode, 'scripted': scripted_episode}
