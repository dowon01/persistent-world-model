"""자기회귀 rollout 유틸. WorldModel 인터페이스만 사용한다."""
import imageio
import numpy as np
import torch


@torch.no_grad()
def rollout(model, context, actions):
  """context [B, k, 3, H, W] 실제 프레임, actions [B, T] → 예측 프레임 [B, T, 3, H, W]."""
  state = model.init_context(context)
  outs = []
  for t in range(actions.shape[1]):
    frame, state = model.step(state, actions[:, t])
    outs.append(frame)
  return torch.stack(outs, 1)


def to_uint8(x):
  """[-1, 1] float [..., 3, H, W] → uint8 [..., H, W, 3] numpy."""
  x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte()
  return x.movedim(-3, -1).cpu().numpy()


def save_side_by_side_gif(path, real, pred, fps=8, scale=4):
  """real, pred uint8 [T, H, W, 3] → 좌: 실제, 우: 예측 GIF."""
  frames = []
  for r, p in zip(real, pred):
    f = np.concatenate([r, np.full((r.shape[0], 2, 3), 255, np.uint8), p], 1)
    frames.append(np.kron(f, np.ones((scale, scale, 1), np.uint8)))
  imageio.mimsave(path, frames, duration=1 / fps, loop=0)
