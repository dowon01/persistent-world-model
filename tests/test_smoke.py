"""shape / smoke 테스트. pytest tests/ (GPU 없어도 동작, 수십 초)"""
import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.env import CrafterEnv, A, in_view, world_to_tile  # noqa: E402
from src.models.interface import CopyLastFrame  # noqa: E402
from src.models.unet import FrameContextModel, UNet  # noqa: E402
from src.rollout import rollout  # noqa: E402
from src.tiles import TileClassifier, view_ground_truth  # noqa: E402


def test_env_geometry_and_daylight():
  env = CrafterEnv(seed=0)
  obs = env.reset()
  assert obs.shape == (72, 72, 3)
  for _ in range(400):
    env.step(A['noop'])
  assert env._world.daylight == 1.0
  p = env.player_pos
  assert in_view(p, p + [4, 3]) and not in_view(p, p + [5, 0]) and not in_view(p, p + [0, 4])
  assert tuple(world_to_tile(p, p)) == (4, 3)


def test_template_matching_exact_on_real_frames():
  env = CrafterEnv(seed=1)
  clf = TileClassifier(env)
  obs = env.reset()
  rng = np.random.default_rng(0)
  for _ in range(50):
    obs, *_ = env.step(int(rng.integers(1, 5)))
  s = env.state()
  gt, mask = view_ground_truth(s['material'], s['semantic'], s['player_pos'])
  pred, dist = clf.classify_frame(obs)
  assert (pred == gt)[mask].all()
  assert dist[mask].max() == 0


def test_unet_shapes_and_rollout():
  net = UNet(context=4, channels=(32, 32, 32, 32), blocks=1, cond_dim=32)
  x = torch.randn(2, 4, 3, 72, 72)
  a = torch.randint(0, 17, (2,))
  assert net(x, a).shape == (2, 3, 72, 72)
  assert net(x, a, sigma=torch.ones(2)).shape == (2, 3, 72, 72)  # Phase 2 자리
  model = FrameContextModel(net)
  out = rollout(model, x[:, :1], torch.randint(0, 17, (2, 3)))  # 이미지 1장 시작도 동작
  assert out.shape == (2, 3, 3, 72, 72)
  b0 = rollout(CopyLastFrame(), x, torch.zeros(2, 3, dtype=torch.long))
  assert torch.equal(b0[:, -1], x[:, -1])


def test_scenario_generation():
  from src.scenarios import generate
  sc = None
  for i in range(20):
    sc, why = generate(('place_stone', 8, i, 4))
    if sc is not None:
      break
  assert sc is not None
  pos, tile = sc['player_pos'], sc['tile']
  vis = np.array([in_view(q, tile) for q in pos])
  assert (~vis[sc['change_idx']:]).sum() == 8
  assert np.array_equal(pos[sc['final_idx']], pos[sc['change_idx']])
  assert len(sc['frames']) == len(sc['actions']) + 1


def test_memory_align_and_model():
  from src.models.memory import align, make_condition, FrameMemory
  from src.models.memory_model import MemoryWorldModel, estimate_move
  env = CrafterEnv(seed=2)
  f0 = env.reset()
  f1, *_ = env.step(A['move_right'])
  t = lambda f: torch.from_numpy(f).permute(2, 0, 1)[None].float() / 127.5 - 1  # noqa: E731
  x0, x1 = t(f0), t(f1)
  # 실제로 한 칸 이동했다면, 정렬한 f0은 f1과 (마스크 영역에서) 거의 같아야 한다
  p0 = torch.tensor([[0, 0]])
  moved = estimate_move(x0, x1, torch.tensor([A['move_right']]))
  aligned, mask = align(x0, p0, p0 + moved)
  err = (((aligned - x1) ** 2) * mask).sum() / mask.sum() / 3
  assert err < 0.02
  for mode, ch in [('aligned', 5), ('raw', 7)]:
    c = make_condition(mode, x0, p0, p0 + moved, torch.tensor([1.0]), torch.tensor([True]))
    assert c.shape == (1, ch, 72, 72)
    net = UNet(context=4, channels=(32, 32, 32, 32), blocks=1, cond_dim=32, extra_in=ch)
    m = MemoryWorldModel(net, mode)
    out = rollout(m, torch.cat([x0, x1])[None].repeat(2, 3, 1, 1, 1)[:, :6], torch.randint(0, 17, (2, 5)))
    assert out.shape == (2, 5, 3, 72, 72)
