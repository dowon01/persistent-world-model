"""env 래퍼 확인: 렌더링 기하, info, 낮밤 고정, 결정성, 템플릿 매칭 정확도, 속도."""
import pathlib
import sys
import time

import imageio
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.env import CrafterEnv, ACTIONS, TILE, VIEW, VIEW_OFFSET  # noqa: E402
from src.tiles import TileClassifier, view_ground_truth  # noqa: E402

out = ROOT / 'outputs' / 'check_env'
out.mkdir(parents=True, exist_ok=True)

env = CrafterEnv(seed=0)
obs = env.reset()
obs, _, _, info = env.step(0)
print('obs', obs.shape, obs.dtype, '| tile', TILE, 'px | view', VIEW, '| player tile', VIEW_OFFSET)
print('actions', len(ACTIONS))
print('info keys', sorted(info), '| semantic', info['semantic'].shape)
print('materials', env.material_names())

grid = np.kron(obs, np.ones((4, 4, 1), np.uint8))
grid[::TILE * 4] = 255
grid[:, ::TILE * 4] = 255
imageio.imwrite(out / 'grid.png', grid)

# 낮밤 고정 + 템플릿 매칭 정확도 + 속도 (랜덤 행동, sleep 제외)
clf = TileClassifier(env)
rng = np.random.default_rng(0)
acts = [i for i, a in enumerate(ACTIONS) if a != 'sleep']
correct = total = 0
worst = 0.0
lights = []
t0 = time.time()
steps = 3000
for t in range(steps):
  obs, _, done, _ = env.step(int(rng.choice(acts)))
  lights.append(env._world.daylight)
  if t % 10 == 0:
    s = env.state()
    gt, mask = view_ground_truth(s['material'], s['semantic'], s['player_pos'])
    pred, dist = clf.classify_frame(obs)
    correct += (pred == gt)[mask].sum()
    total += mask.sum()
    worst = max(worst, dist[mask].max())
  if done:
    env.reset()
dt = time.time() - t0
print(f'daylight min/max {min(lights)} {max(lights)}')
print(f'template accuracy {correct}/{total} = {correct / total:.4f}, max matched dist {worst:.2f}')
print(f'speed ~{steps / dt:.0f} steps/s (분류 포함)')


def rollout(seed):
  e = CrafterEnv(seed=seed)
  e.reset()
  r = np.random.default_rng(1)
  return np.stack([e.step(int(r.integers(0, 17)))[0] for _ in range(200)])


print('deterministic', np.array_equal(rollout(3), rollout(3)))
