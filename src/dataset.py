"""학습 데이터 로더.

전체 프레임을 uint8로 GPU에 올려 두고 인덱스로 배치를 뽑는다.
30만 프레임 × 72×72×3 ≈ 4.7GB라 16GB GPU에 들어가고, DataLoader 병목이 없다.

샘플 하나 = 연속 (K + U)프레임과 그 사이 행동 K+U-1개 중 뒤 U개
  frames  [B, K+U, 3, 72, 72] float (-1~1)
  actions [B, U]  (frames[:, K-1+u] → frames[:, K+u]를 만드는 행동)
U=1이면 1-step teacher forcing, U>1이면 자기회귀 unroll 학습에 쓴다.
"""
import glob
import pathlib

import numpy as np
import torch


def list_episodes(data_dir):
  return sorted(glob.glob(str(pathlib.Path(data_dir) / 'ep*.npz')))


def split_episodes(files, val_frac=0.05):
  """파일 이름 순서 기준 마지막 val_frac을 검증용으로 (seed가 겹치지 않음)."""
  n_val = max(1, int(len(files) * val_frac))
  return files[:-n_val], files[-n_val:]


class FrameDataset:

  def __init__(self, files, device, load_actions=True, keys=('player_pos',)):
    obs, acts, starts, extra = [], [], [], {k: [] for k in keys}
    n = 0
    for f in files:
      z = np.load(f)
      o = z['obs']
      obs.append(o)
      # 행동은 프레임과 길이를 맞추려고 마지막에 dummy 하나 추가: action[t]는 obs[t]→obs[t+1]
      a = np.append(z['action'], 0) if load_actions else np.zeros(len(o), np.uint8)
      acts.append(a)
      for k in keys:
        extra[k].append(z[k])
      starts.append((n, len(o)))
      n += len(o)
    self.device = device
    self.obs = torch.from_numpy(np.concatenate(obs)).to(device)  # [N, 72, 72, 3] uint8
    self.actions = torch.from_numpy(np.concatenate(acts).astype(np.int64)).to(device)
    self.extra = {k: torch.from_numpy(np.concatenate(v).astype(np.int64)).to(device) for k, v in extra.items()}
    self.episodes = starts
    self.load_actions = load_actions
    self.files = list(files)

  def __len__(self):
    return len(self.obs)

  def valid_starts(self, length):
    """에피소드 경계를 넘지 않는 길이 length 구간의 시작 인덱스."""
    idx = [np.arange(s, s + l - length + 1) for s, l in self.episodes if l >= length]
    return torch.from_numpy(np.concatenate(idx)).to(self.device)

  @staticmethod
  def to_float(x):
    """uint8 [..., H, W, 3] → float [..., 3, H, W], -1~1."""
    return x.permute(*range(x.dim() - 3), -1, -3, -2).float().div(127.5).sub(1)

  def sample(self, batch, K, U, generator=None):
    if not hasattr(self, '_starts') or self._starts_len != K + U:
      self._starts, self._starts_len = self.valid_starts(K + U), K + U
    i = self._starts[torch.randint(len(self._starts), (batch,), device=self.device, generator=generator)]
    idx = i[:, None] + torch.arange(K + U, device=self.device)
    frames = self.to_float(self.obs[idx])
    actions = self.actions[idx[:, K - 1:K - 1 + U]]
    return frames, actions

  # memory M1 학습용 ------------------------------------------------------------
  def load_memory_index(self, path):
    """src/memory_index.py 결과를 이 데이터셋의 전역 인덱스로 옮긴다."""
    z = np.load(path)
    q, r, c = [], [], []
    for (start, _), f in zip(self.episodes, self.files):
      stem = pathlib.Path(f).stem
      rr = z[f'{stem}/ret']
      q.append(z[f'{stem}/query'])
      r.append(np.where(rr >= 0, rr + start, -1))
      c.append(z[f'{stem}/cat'])
    self.mem_query = torch.from_numpy(np.concatenate(q)).to(self.device)
    self.mem_ret = torch.from_numpy(np.concatenate(r)).to(self.device)
    self.mem_cat = torch.from_numpy(np.concatenate(c).astype(np.int64)).to(self.device)

  def memory_pools(self, K):
    """context 마지막 시점 t의 후보: (전체, 과거 지형이 드러남 cat>=1, 바뀐 타일이 드러남 cat==2)."""
    t = self.valid_starts(K + 1) + K - 1
    return t, t[self.mem_cat[t] >= 1], t[self.mem_cat[t] == 2]

  def sample_memory(self, batch, K, mode, mix=(0.5, 0.3, 0.2), drop=0.2, generator=None):
    """memory 조건까지 포함한 1-step 샘플.

    mix = (일반, 과거 지형, 바뀐 타일) 비율. memory 없이는 못 맞추는 샘플이 3%뿐이라
    그대로 뽑으면 모델이 memory를 무시할 수 있어 비율을 올린다.
    drop = memory를 비우는 확률 (memory 없이도 동작하도록).
    반환: frames [B, K+1, 3, H, W], actions [B, 1], extra [B, C, H, W]
    """
    from src.models.memory import make_condition
    if not hasattr(self, '_pools') or self._pools_K != K:
      self._pools, self._pools_K = self.memory_pools(K), K
    counts = [int(round(batch * m)) for m in mix]
    counts[0] = batch - sum(counts[1:])
    ts = []
    for pool, n in zip(self._pools, counts):
      ts.append(pool[torch.randint(len(pool), (n,), device=self.device, generator=generator)])
    t = torch.cat(ts)
    idx = t[:, None] + torch.arange(-K + 1, 2, device=self.device)
    frames = self.to_float(self.obs[idx])
    actions = self.actions[t][:, None]
    r = self.mem_ret[t]
    has = r >= 0
    if drop > 0:
      has &= torch.rand(len(t), device=self.device, generator=generator) >= drop
    r = r.clamp(min=0)
    pos = self.extra['player_pos']
    extra = make_condition(mode, self.to_float(self.obs[r]), pos[r], self.mem_query[t],
                           (t + 1 - r).float(), has)
    return frames, actions, extra


  # 혼합 샘플링 + 바뀐 칸 가중치 (v2) ----------------------------------------------
  def load_interact_index(self, path):
    """src/interact_index.py 결과를 전역 인덱스로."""
    z = np.load(path)
    i, c, v = [], [], []
    for f in self.files:
      stem = pathlib.Path(f).stem
      i.append(z[f'{stem}/inter'])
      c.append(z[f'{stem}/chg'])
      v.append(z[f'{stem}/inv'])
    self.inter = torch.from_numpy(np.concatenate(i).astype(np.int64)).to(self.device)
    self.chg = torch.from_numpy(np.concatenate(c).astype(np.int64)).to(self.device)
    self.inv_changed = torch.from_numpy(np.concatenate(v)).to(self.device)
    self.tgt = torch.from_numpy(np.concatenate([z[f'{pathlib.Path(f).stem}/tgt'] for f in self.files])
                                .astype(np.int64)).to(self.device)

  def mixed_pools(self, K):
    """context 마지막 시점 t의 후보 풀."""
    t = self.valid_starts(K + 1) + K - 1
    pools = {'all': t}
    if hasattr(self, 'inter'):
      pools['inter_ok'] = t[self.inter[t] == 1]
      pools['inter_fail'] = t[self.inter[t] >= 2]         # 모든 실패 (v2와 같은 정의: 2와 3을 합친 것)
      pools['inter_fail_nomat'] = t[self.inter[t] == 3]   # 인벤토리 부족 실패 (v3)
    if hasattr(self, 'mem_cat'):
      pools['mem_terrain'] = t[self.mem_cat[t] >= 1]
      pools['mem_changed'] = t[self.mem_cat[t] == 2]
    return pools

  def change_weight(self, t, lam, fail_target=False):
    """w = 1 + λ·M [B, 1, H, W]. M = t→t+1에서 재질이 바뀐 칸 (다음 시야 기준)
    + 상호작용 성공이면서 인벤토리가 바뀐 경우 인벤토리 줄
    + (fail_target, v3) 상호작용 실패 장면의 바라보던 칸. '안 바뀜'을 맞히는 것에도 같은 가중치를 줘서,
      애매하면 '바뀜'으로 그리는 쪽이 손실상 유리한 비대칭을 없앤다."""
    from src.env import TILE, VIEW, VIEW_OFFSET
    B = len(t)
    m = torch.zeros(B, VIEW[1], VIEW[0], device=self.device)
    p1 = self.extra['player_pos'][t + 1]                       # [B, 2]
    for k in range(self.chg.shape[1]):
      w = self.chg[t, k]                                        # [B, 2] 월드 좌표 (-1이면 없음)
      u = w - p1 + torch.as_tensor(VIEW_OFFSET, device=self.device)
      ok = (w[:, 0] >= 0) & (u[:, 0] >= 0) & (u[:, 0] < VIEW[0]) & (u[:, 1] >= 0) & (u[:, 1] < VIEW[1])
      b = torch.nonzero(ok)[:, 0]
      m[b, u[b, 1], u[b, 0]] = 1
    if fail_target:
      w = self.tgt[t]
      u = w - p1 + torch.as_tensor(VIEW_OFFSET, device=self.device)
      ok = (self.inter[t] >= 2) & (w[:, 0] >= 0) & (u[:, 0] >= 0) & (u[:, 0] < VIEW[0]) & (u[:, 1] >= 0) & (u[:, 1] < VIEW[1])
      b = torch.nonzero(ok)[:, 0]
      m[b, u[b, 1], u[b, 0]] = 1
    full = torch.zeros(B, 1, 72, 72, device=self.device)
    full[:, 0, :VIEW[1] * TILE] = m.repeat_interleave(TILE, 1).repeat_interleave(TILE, 2)
    inv_on = (self.inter[t] == 1) & self.inv_changed[t]
    full[inv_on, :, VIEW[1] * TILE:] = 1
    return 1 + lam * full

  def sample_mixed(self, batch, K, mix, memory_mode=None, drop=0.2, lam=0.0, fail_target=False, generator=None):
    """mix = {풀 이름: 비율}. 반환: frames [B, K+1, ...], actions [B, 1], extra (또는 None), weight (또는 None)."""
    from src.models.memory import make_condition
    if not hasattr(self, '_mixed') or self._mixed_K != K:
      self._mixed, self._mixed_K = self.mixed_pools(K), K
    names = list(mix)
    counts = [int(round(batch * mix[n])) for n in names]
    counts[names.index('all')] = batch - sum(c for n, c in zip(names, counts) if n != 'all')
    ts = []
    for n, c in zip(names, counts):
      pool = self._mixed[n]
      ts.append(pool[torch.randint(len(pool), (c,), device=self.device, generator=generator)])
    t = torch.cat(ts)
    idx = t[:, None] + torch.arange(-K + 1, 2, device=self.device)
    frames = self.to_float(self.obs[idx])
    actions = self.actions[t][:, None]
    extra = None
    if memory_mode:
      r = self.mem_ret[t]
      has = r >= 0
      if drop > 0:
        has &= torch.rand(len(t), device=self.device, generator=generator) >= drop
      r = r.clamp(min=0)
      pos = self.extra['player_pos']
      extra = make_condition(memory_mode, self.to_float(self.obs[r]), pos[r], self.mem_query[t],
                             (t + 1 - r).float(), has)
    weight = self.change_weight(t, lam, fail_target) if lam > 0 else None
    return frames, actions, extra, weight
