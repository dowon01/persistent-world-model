"""Phase 1 baseline: 행동 조건 다음 프레임 예측 UNet.

입력  x = concat(과거 K프레임) [B, 3K, 72, 72],  조건 c = ActionEncoder(a) (+ noise level 임베딩 자리)
출력  다음 프레임 x̂_{t+1} [B, 3, 72, 72] (-1~1)

설계 메모:
- 조건 주입 = AdaGN: h ← GroupNorm(h)·(1 + γ(c)) + β(c).
  행동 하나가 화면 전체를 8px 평행이동시키는 Crafter에서는 행동이 모든 층에 영향을 줘야 하므로,
  입력 채널에 붙이는 방식보다 모든 블록에 들어가는 AdaGN을 쓴다 (DIAMOND, EDM 계열과 같은 방식).
- 해상도 72→36→18→9. 9×9는 Crafter 타일 그리드와 같아서 가장 낮은 층의 한 칸 = 타일 한 칸.
  9×9에 self-attention을 둬 화면 전체(시야 끝에서 끝)를 한 번에 볼 수 있게 한다.
- noise level 조건 자리(sigma): Phase 2에서 같은 backbone을 EDM diffusion denoiser로 재사용하기 위함.
  Phase 1에서는 항상 None (임베딩 0).
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from src.models.interface import ActionEncoder


def sinusoidal(x, dim):
  half = dim // 2
  freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
  a = x[:, None].float() * freqs[None]
  return torch.cat([a.sin(), a.cos()], -1)


class AdaGN(nn.Module):

  def __init__(self, ch, cond_dim, groups=32):
    super().__init__()
    self.norm = nn.GroupNorm(groups, ch, affine=False)
    self.proj = nn.Linear(cond_dim, 2 * ch)
    nn.init.zeros_(self.proj.weight)
    nn.init.zeros_(self.proj.bias)

  def forward(self, h, c):
    gamma, beta = self.proj(c)[:, :, None, None].chunk(2, 1)
    return self.norm(h) * (1 + gamma) + beta


class ResBlock(nn.Module):

  def __init__(self, cin, cout, cond_dim):
    super().__init__()
    self.n1 = AdaGN(cin, cond_dim)
    self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
    self.n2 = AdaGN(cout, cond_dim)
    self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
    nn.init.zeros_(self.c2.weight)
    nn.init.zeros_(self.c2.bias)
    self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

  def forward(self, h, c):
    x = self.c1(F.silu(self.n1(h, c)))
    x = self.c2(F.silu(self.n2(x, c)))
    return self.skip(h) + x


class Attention(nn.Module):

  def __init__(self, ch, cond_dim, heads=4):
    super().__init__()
    self.norm = AdaGN(ch, cond_dim)
    self.qkv = nn.Conv2d(ch, 3 * ch, 1)
    self.out = nn.Conv2d(ch, ch, 1)
    nn.init.zeros_(self.out.weight)
    nn.init.zeros_(self.out.bias)
    self.heads = heads

  def forward(self, h, c):
    B, C, H, W = h.shape
    q, k, v = self.qkv(self.norm(h, c)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
    o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
    return h + self.out(o.transpose(-1, -2).reshape(B, C, H, W))


class UNet(nn.Module):

  def __init__(self, context=4, channels=(64, 128, 256, 256), blocks=2, cond_dim=256,
               num_actions=17, extra_in=0):
    """extra_in: 추가 입력 채널 (Phase 1 memory M1의 검색 프레임 등)."""
    super().__init__()
    self.context = context
    self.action = ActionEncoder(num_actions, cond_dim)
    self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
    self.cond_dim = cond_dim
    self.inp = nn.Conv2d(3 * context + extra_in, channels[0], 3, padding=1)
    self.down, self.up = nn.ModuleList(), nn.ModuleList()
    skips, ch = [channels[0]], channels[0]
    for i, c in enumerate(channels):
      for _ in range(blocks):
        self.down.append(ResBlock(ch, c, cond_dim))
        ch = c
        skips.append(ch)
      if i < len(channels) - 1:
        self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        skips.append(ch)
    self.mid1 = ResBlock(ch, ch, cond_dim)
    self.mid_attn = Attention(ch, cond_dim)
    self.mid2 = ResBlock(ch, ch, cond_dim)
    for i, c in reversed(list(enumerate(channels))):
      for _ in range(blocks + 1):
        self.up.append(ResBlock(ch + skips.pop(), c, cond_dim))
        ch = c
      if i > 0:
        self.up.append(nn.Upsample(scale_factor=2, mode='nearest'))
    self.out_norm = nn.GroupNorm(32, ch)
    self.out = nn.Conv2d(ch, 3, 3, padding=1)

  def forward(self, context_frames, action, sigma=None, extra=None):
    """context_frames [B, K, 3, H, W], action [B] → [B, 3, H, W]."""
    B = context_frames.shape[0]
    x = context_frames.reshape(B, -1, *context_frames.shape[-2:])
    if extra is not None:
      x = torch.cat([x, extra], 1)
    c = self.action(action)
    if sigma is not None:  # Phase 2 diffusion용 자리
      c = c + sinusoidal(sigma.log() / 4, self.cond_dim)
    c = self.cond_mlp(c)
    h = self.inp(x)
    hs = [h]
    for m in self.down:
      h = m(h, c) if isinstance(m, ResBlock) else m(h)
      hs.append(h)
    h = self.mid2(self.mid_attn(self.mid1(h, c), c), c)
    for m in self.up:
      if isinstance(m, ResBlock):
        h = m(torch.cat([h, hs.pop()], 1), c)
      else:
        h = m(h)
    return self.out(F.silu(self.out_norm(h)))


class FrameContextModel:
  """UNet을 WorldModel 인터페이스로 감싼다. State = 최근 K프레임."""

  def __init__(self, net):
    self.net = net
    self.K = net.context

  @torch.no_grad()
  def init_context(self, frames, actions=None):
    assert frames.shape[1] >= 1
    if frames.shape[1] < self.K:  # 이미지 1장 등 짧은 시작: 첫 프레임 반복
      pad = frames[:, :1].expand(-1, self.K - frames.shape[1], -1, -1, -1)
      frames = torch.cat([pad, frames], 1)
    return frames[:, -self.K:]

  @torch.no_grad()
  def step(self, state, action):
    with torch.autocast('cuda', torch.bfloat16, enabled=state.is_cuda):
      nxt = self.net(state, action).float().clamp(-1, 1)
    return nxt, torch.cat([state[:, 1:], nxt[:, None]], 1)
