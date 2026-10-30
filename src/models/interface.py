"""모든 world model의 공통 인터페이스.

평가(eval_revisit.py)와 데모(play.py)는 이 인터페이스만 호출한다.
UNet, diffusion, RSSM, Genie식 모델이 모두 같은 하네스로 평가된다.
"""
from typing import Any, Protocol

import torch
from torch import nn

State = Any


class WorldModel(Protocol):

  def init_context(self, frames: torch.Tensor, actions: torch.Tensor | None = None) -> State:
    """frames [B, k, 3, H, W] (-1~1): 실제 프레임 k장 (또는 이미지 1장)으로 시작 상태를 만든다."""

  def step(self, state: State, action: torch.Tensor) -> tuple[torch.Tensor, State]:
    """action [B] → 다음 프레임 [B, 3, H, W] (-1~1)과 갱신된 상태."""


class ActionEncoder(nn.Module):
  """행동 → 조건 벡터. 지금은 Crafter 17개 이산 행동 embedding.
  Phase 4에서 latent action 모델 출력으로 교체한다."""

  def __init__(self, num_actions=17, dim=128):
    super().__init__()
    self.emb = nn.Embedding(num_actions, dim)

  def forward(self, action):
    return self.emb(action)


class CopyLastFrame:
  """B0: 직전 프레임을 그대로 복사하는 하한 기준선."""

  def init_context(self, frames, actions=None):
    return frames[:, -1]

  def step(self, state, action):
    return state, state
