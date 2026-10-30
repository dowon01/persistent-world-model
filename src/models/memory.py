"""Memory M1: 위치 기반 프레임 검색.

쓰기: 매 스텝 (위치 p, 프레임 x)를 저장한다. 실제 context 프레임과 모델이 생성한 프레임 모두.
읽기: 다음 프레임의 (예상) 위치 q로, 다음 시야와 가장 많이 겹치는 과거 프레임 1장을 고른다 (같으면 가장 최근).
      context에 이미 있는 최근 K장은 검색 대상에서 뺀다 (같은 정보를 두 번 주면 memory를 쓸 이유가 없다).

조건 입력으로 바꾸는 방식 두 가지 (mode):
  aligned (본 모델): 검색 프레임을 위치 차이만큼 8px 단위로 평행이동해 다음 시야에 정렬한다.
      Crafter는 이동이 정확히 타일 단위라 정렬 오차가 없다. GameGAN처럼 기억을 월드 좌표로 읽는 셈.
      채널 = [정렬 RGB 3, 유효 마스크 1, 나이 1] = 5
      유효 마스크 = 0: 검색 프레임 밖으로 나간 영역, 인벤토리 줄, 옛 프레임 속 플레이어 타일
  raw (ablation): 검색 프레임을 그대로 주고 위치 차이를 상수 채널로 준다. 정렬은 모델이 배워야 한다.
      채널 = [원본 RGB 3, 있음 1, 나이 1, dx 1, dy 1] = 7
      K=16 baseline이 context 안의 어긋난 타일을 활용하지 못했으므로,
      raw가 aligned보다 크게 못하다면 "정렬이 결정적"이라는 근거가 된다.

나이 채널 = min(지난 스텝 수 / AGE_SCALE, 1).
위치는 상대 좌표면 충분하다 (평가·데모에서는 모델이 그린 화면의 밀림으로 추정, src/models/memory_model.py).
"""
import torch

from src.env import TILE, VIEW

H = W = 72
AGE_SCALE = 64.0
CHANNELS = {'aligned': 5, 'raw': 7}


def overlap(src, dst):
  """시야 겹침 타일 수. src, dst [..., 2] (x, y) 타일 좌표."""
  d = (src - dst).abs()
  return (VIEW[0] - d[..., 0]).clamp(min=0) * (VIEW[1] - d[..., 1]).clamp(min=0)


def align(frames, src_pos, dst_pos):
  """frames [B, 3, H, W] (src_pos에서 본 화면) → dst_pos에서 본 화면으로 정렬.

  dst 시야의 타일 u(월드 = dst + u - c)는 src 시야의 타일 u + (dst - src)에 있다.
  반환: aligned [B, 3, H, W], mask [B, 1, H, W]
  """
  B = frames.shape[0]
  out = torch.zeros_like(frames)
  mask = torch.zeros(B, 1, H, W, device=frames.device, dtype=frames.dtype)
  s = (dst_pos - src_pos).long().tolist()
  vx, vy = VIEW
  for b in range(B):
    sx, sy = s[b]
    # dst 타일 x 범위: 0 <= x < vx 이면서 0 <= x + sx < vx
    x0, x1 = max(0, -sx), min(vx, vx - sx)
    y0, y1 = max(0, -sy), min(vy, vy - sy)
    if x0 >= x1 or y0 >= y1:
      continue
    out[b, :, y0 * TILE:y1 * TILE, x0 * TILE:x1 * TILE] = \
        frames[b, :, (y0 + sy) * TILE:(y1 + sy) * TILE, (x0 + sx) * TILE:(x1 + sx) * TILE]
    mask[b, :, y0 * TILE:y1 * TILE, x0 * TILE:x1 * TILE] = 1
    # 옛 프레임 속 플레이어 타일 (src 시야 중앙) → dst 시야에서는 중앙 - s
    px, py = vx // 2 - sx, vy // 2 - sy
    if 0 <= px < vx and 0 <= py < vy:
      mask[b, :, py * TILE:(py + 1) * TILE, px * TILE:(px + 1) * TILE] = 0
  return out * mask, mask


def make_condition(mode, frames, src_pos, dst_pos, age, has):
  """검색 결과 → UNet extra 입력 [B, C, H, W].

  frames [B, 3, H, W] (-1~1), src_pos/dst_pos [B, 2], age [B] (스텝), has [B] bool (검색 결과 있음).
  """
  B = frames.shape[0]
  has_f = has.float().view(B, 1, 1, 1)
  age_c = (age.float() / AGE_SCALE).clamp(max=1).view(B, 1, 1, 1) * has_f
  if mode == 'aligned':
    rgb, mask = align(frames, src_pos, dst_pos)
    mask = mask * has_f
    return torch.cat([rgb * mask, mask, age_c * mask], 1)
  if mode == 'raw':
    d = (src_pos - dst_pos).float() * has_f.view(B, 1)
    ones = torch.ones(B, 1, H, W, device=frames.device)
    return torch.cat([frames * has_f, has_f * ones, age_c * ones,
                      (d[:, 0] / VIEW[0]).view(B, 1, 1, 1) * ones, (d[:, 1] / VIEW[1]).view(B, 1, 1, 1) * ones], 1)
  raise ValueError(mode)


class FrameMemory:
  """rollout 중 쓰고 읽는 memory (배치 단위). 평가·데모용.

  rule:
    overlap (학습과 같은 규칙): 최근 exclude_last장 제외, 시야 겹침 최대 → 같으면 최근
    recent  (평가 전용 진단, 2026-10-10): 학습과 같이 최근 exclude_last장은 제외하되, 그중 다음 스텝에
            새로 드러나는 줄을 온전히 담은 가장 최근 프레임을 고른다. 없으면 overlap 규칙과 같다.
            N이 짧을 때 '변화 전' 프레임이 검색되는 문제가 원인인지 확인용.
            (처음엔 context까지 검색했으나, 학습 때 본 적 없는 나이 1~4 memory가 들어가 모델이 망가져 진단이 안 됐다.)
  """

  def __init__(self, mode, exclude_last, rule='overlap'):
    self.mode, self.exclude_last, self.rule = mode, exclude_last, rule
    self.frames, self.pos = [], []  # 리스트 길이 = 저장된 스텝 수, 원소 [B, 3, H, W] / [B, 2]

  def write(self, frame, pos):
    self.frames.append(frame)
    self.pos.append(pos.clone())

  def read(self, query, current=None):
    """query [B, 2] → extra [B, C, H, W]. current [B, 2]: 지금 위치 (rule='recent'에서 새로 드러나는 줄 계산)."""
    if self.rule == 'recent':
      return self._read_recent(query, current)
    B = query.shape[0]
    n = len(self.frames) - self.exclude_last
    if n <= 0:
      zero = torch.zeros(B, 3, H, W, device=query.device)
      return make_condition(self.mode, zero, query, query, torch.zeros(B, device=query.device),
                            torch.zeros(B, dtype=torch.bool, device=query.device))
    pos = torch.stack(self.pos[:n], 1)                        # [B, n, 2]
    ov = overlap(pos, query[:, None])                         # [B, n]
    score = ov * (n + 1) + torch.arange(n, device=query.device)  # 겹침 우선, 같으면 최근
    idx = score.argmax(1)
    has = ov.gather(1, idx[:, None])[:, 0] > 0
    frames = torch.stack([self.frames[i][b] for b, i in enumerate(idx.tolist())])
    src = pos[torch.arange(B), idx]
    age = (len(self.frames) - idx).float()
    return make_condition(self.mode, frames, src, query, age, has)

  def _read_recent(self, query, current):
    B, dev = query.shape[0], query.device
    n = len(self.frames) - self.exclude_last
    if n <= 0:
      return self.read(query) if self.rule != 'recent' else make_condition(
          self.mode, torch.zeros(B, 3, H, W, device=dev), query, query, torch.zeros(B, device=dev),
          torch.zeros(B, dtype=torch.bool, device=dev))
    pos = torch.stack(self.pos[:n], 1)                        # [B, n, 2]
    hx, hy = VIEW[0] // 2, VIEW[1] // 2
    # 다음 시야 사각형 중 지금 시야 밖인 부분 = 새로 드러나는 줄 (이동하지 않으면 빈 영역)
    d = query - current
    lo = query - torch.tensor([hx, hy], device=dev)
    hi = query + torch.tensor([hx, hy], device=dev)
    rlo, rhi = lo.clone(), hi.clone()
    for ax in range(2):
      pos_mv, neg_mv = d[:, ax] > 0, d[:, ax] < 0
      rlo[pos_mv, ax] = hi[pos_mv, ax]   # 오른쪽/아래로 이동 → 끝 줄만
      rhi[neg_mv, ax] = lo[neg_mv, ax]   # 왼쪽/위로 이동 → 첫 줄만
    moved = (d != 0).any(1)
    vlo = pos - torch.tensor([hx, hy], device=dev)
    vhi = pos + torch.tensor([hx, hy], device=dev)
    covers = ((vlo <= rlo[:, None]) & (vhi >= rhi[:, None])).all(-1) & moved[:, None]   # [B, n]
    ar = torch.arange(n, device=dev)
    ov = overlap(pos, query[:, None])
    big = (VIEW[0] * VIEW[1] + 1) * (n + 1)  # 겹침 점수의 상한보다 큼
    score = torch.where(covers, big + ar, ov * (n + 1) + ar)  # 덮는 프레임이 있으면 그중 최근
    idx = score.argmax(1)
    has = ov.gather(1, idx[:, None])[:, 0] > 0
    frames = torch.stack([self.frames[i][b] for b, i in enumerate(idx.tolist())])
    src = pos[torch.arange(B), idx]
    age = (len(self.frames) - idx).float()
    return make_condition(self.mode, frames, src, query, age, has)

