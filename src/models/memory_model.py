"""memory를 쓰는 WorldModel 래퍼 + 모델이 그린 화면으로 위치 추정.

위치 추정: Crafter에서 이동이 성공하면 월드 뷰가 정확히 한 타일(8px) 밀린다.
  새 프레임이 '직전 프레임을 d만큼 민 모습'과 얼마나 가까운지 타일 단위 오차로 비교해 d를 고른다.
  - 이동 행동일 때 후보 = {제자리, 행동 방향}. 오차 차이가 작으면(풀밭처럼 밀어도 똑같아 보이는 경우) 이동 성공으로 본다.
  - context(실제 프레임)끼리는 행동을 모르므로 후보 = {제자리, 4방향}.
  정답 위치를 쓰지 않으므로 baseline과 같은 정보(프레임 + 행동)만 쓴다.
  oracle_pos를 주면 실제 위치를 쓴다 (상한선 분석용).
"""
import torch

from src.env import MOVES, TILE, VIEW
from src.models.memory import FrameMemory

DIRS = [(0, 0)] + list(MOVES.values())


def tile_errors(prev, nxt, d):
  """d만큼 이동했다는 가정에서 nxt와 (밀린) prev의 타일별 MSE 평균 [B].
  nxt 타일 u ↔ prev 타일 u + d. 두 프레임의 플레이어 타일은 뺀다."""
  dx, dy = d
  vx, vy = VIEW
  x0, x1 = max(0, -dx), min(vx, vx - dx)
  y0, y1 = max(0, -dy), min(vy, vy - dy)
  a = nxt[:, :, y0 * TILE:y1 * TILE, x0 * TILE:x1 * TILE]
  b = prev[:, :, (y0 + dy) * TILE:(y1 + dy) * TILE, (x0 + dx) * TILE:(x1 + dx) * TILE]
  B = a.shape[0]
  e = ((a - b) ** 2).reshape(B, 3, y1 - y0, TILE, x1 - x0, TILE).mean((1, 3, 5))  # [B, ty, tx]
  keep = torch.ones_like(e, dtype=torch.bool)
  for cx, cy in [(vx // 2, vy // 2), (vx // 2 - dx, vy // 2 - dy)]:  # nxt의 플레이어, prev 플레이어가 옮겨진 자리
    if x0 <= cx < x1 and y0 <= cy < y1:
      keep[:, cy - y0, cx - x0] = False
  return (e * keep).sum((1, 2)) / keep.sum((1, 2))


def estimate_move(prev, nxt, action=None, margin=1e-3):
  """반환: 추정 이동 [B, 2] (long)."""
  B = prev.shape[0]
  dev = prev.device
  if action is None:
    errs = torch.stack([tile_errors(prev, nxt, d) for d in DIRS], 1)  # [B, 5]
    return torch.tensor(DIRS, device=dev)[errs.argmin(1)]
  out = torch.zeros(B, 2, dtype=torch.long, device=dev)
  for a, d in MOVES.items():
    sel = action == a
    if sel.any():
      stay = tile_errors(prev[sel], nxt[sel], (0, 0))
      move = tile_errors(prev[sel], nxt[sel], d)
      moved = ~(stay + margin < move)  # 애매하면 이동 성공
      out[sel] = moved[:, None].long() * torch.tensor(d, device=dev)
  return out


def action_delta(action):
  d = torch.zeros(action.shape[0], 2, dtype=torch.long, device=action.device)
  for a, v in MOVES.items():
    d[action == a] = torch.tensor(v, device=action.device)
  return d


class MemoryWorldModel:

  def __init__(self, net, mode, oracle_pos=None, retrieval='overlap'):
    self.net, self.mode, self.K = net, mode, net.context
    self.retrieval = retrieval  # FrameMemory rule
    self.oracle_pos = oracle_pos  # [B, T, 2] 실제 위치 (init_context에 준 프레임부터) 또는 None
    self.use_oracle = oracle_pos is not None  # 평가 하네스가 배치마다 oracle_pos를 채운다

  @torch.no_grad()
  def init_context(self, frames, actions=None):
    B, k = frames.shape[:2]
    mem = FrameMemory(self.mode, exclude_last=self.K, rule=self.retrieval)
    pos = torch.zeros(B, 2, dtype=torch.long, device=frames.device)
    self.t = 0
    if self.oracle_pos is not None:
      pos = self.oracle_pos[:, 0].clone()
    mem.write(frames[:, 0], pos)
    for i in range(1, k):
      self.t += 1
      pos = self.oracle_pos[:, i].clone() if self.oracle_pos is not None else \
          pos + estimate_move(frames[:, i - 1], frames[:, i])
      mem.write(frames[:, i], pos)
    ctx = frames[:, -self.K:]
    if ctx.shape[1] < self.K:
      ctx = torch.cat([ctx[:, :1].expand(-1, self.K - ctx.shape[1], -1, -1, -1), ctx], 1)
    return dict(ctx=ctx, mem=mem, pos=pos)

  @torch.no_grad()
  def step(self, state, action):
    ctx, mem, pos = state['ctx'], state['mem'], state['pos']
    query = pos + action_delta(action)  # 다음 프레임의 예상 위치 (학습 때와 같은 규칙, src/memory_index.py)
    extra = mem.read(query, current=pos)
    with torch.autocast('cuda', torch.bfloat16, enabled=ctx.is_cuda):
      nxt = self.net(ctx, action, extra=extra).float().clamp(-1, 1)
    self.t += 1
    if self.oracle_pos is not None:
      pos = self.oracle_pos[:, self.t].clone()
    else:
      pos = pos + estimate_move(ctx[:, -1], nxt, action)
    mem.write(nxt, pos)
    return nxt, dict(ctx=torch.cat([ctx[:, 1:], nxt[:, None]], 1), mem=mem, pos=pos)
