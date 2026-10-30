"""타일 판정 = 템플릿 매칭.

Crafter는 재질 텍스처(16×16 PNG)를 타일 크기(8px)로 NEAREST 축소해 그대로 그리고,
낮밤을 고정하면 조명 처리도 바뀌지 않는다. 그래서 재질 타일은 항상 같은 8×8 패치가 된다.
각 패치를 가장 가까운 템플릿(L2)의 재질로 판정한다. 학습이 필요 없다.

- 월드 밖 타일은 회색(127)으로 그려진다 → id 0 ('none')으로 둔다.
- 플레이어·소·좀비 등 object는 재질 위에 겹쳐 그려지므로 템플릿과 맞지 않는다.
  이런 타일은 지표에서 제외한다 (마스크는 정답 semantic으로 만든다).
"""
import numpy as np

from src.env import CrafterEnv, TILE, VIEW, tile_patch


def build_templates(env=None):
  """재질 id → (8, 8, 3) float32 템플릿. 반환: (ids [M], templates [M, 8, 8, 3])."""
  env = env or CrafterEnv(seed=0)
  names = env.material_names()  # {0: None, 1: 'water', ...}
  ids, temps = [], []
  for i, name in sorted(names.items()):
    if name is None:
      t = np.full((TILE, TILE, 3), 127, np.uint8)
    else:
      t = env._textures.get(name, (TILE, TILE))[..., :3]
      t = t.transpose(1, 0, 2)  # Textures는 [x, y]로 저장 → 프레임과 같은 [y, x]로
    ids.append(i)
    temps.append(t.astype(np.float32))
  return np.array(ids), np.stack(temps)


class TileClassifier:

  def __init__(self, env=None):
    self.ids, self.templates = build_templates(env)
    self._flat = self.templates.reshape(len(self.ids), -1)

  def classify_patches(self, patches):
    """patches [..., 8, 8, 3] (uint8 또는 0~255 float) → (재질 id [...], 최소 거리 [...])."""
    p = np.asarray(patches, np.float32).reshape(*patches.shape[:-3], -1)
    d = ((p[..., None, :] - self._flat) ** 2).mean(-1)  # [..., M] 픽셀당 MSE
    k = d.argmin(-1)
    return self.ids[k], np.take_along_axis(d, k[..., None], -1)[..., 0]

  def classify_frame(self, frame):
    """frame [..., 72, 72, 3] → 월드 뷰 재질 [..., VIEW_y, VIEW_x], 거리 같은 shape."""
    f = np.asarray(frame)[..., :VIEW[1] * TILE, :, :]
    lead = f.shape[:-3]
    f = f.reshape(*lead, VIEW[1], TILE, VIEW[0], TILE, 3)
    f = np.moveaxis(f, -4, -3)  # [..., Vy, Vx, 8, 8, 3]
    return self.classify_patches(f)


def view_ground_truth(material, semantic, player_pos):
  """정답 재질 맵에서 월드 뷰 영역을 잘라 [Vy, Vx]로. object가 있는 타일은 mask=False.

  material, semantic: (64, 64) [x, y] 인덱싱. 월드 밖은 재질 0, mask True (회색으로 그려지므로 판정 가능).
  """
  gt = np.zeros((VIEW[1], VIEW[0]), np.int64)
  mask = np.ones((VIEW[1], VIEW[0]), bool)
  for ty in range(VIEW[1]):
    for tx in range(VIEW[0]):
      wx, wy = player_pos[0] + tx - VIEW[0] // 2, player_pos[1] + ty - VIEW[1] // 2
      if 0 <= wx < material.shape[0] and 0 <= wy < material.shape[1]:
        gt[ty, tx] = material[wx, wy]
        mask[ty, tx] = semantic[wx, wy] == material[wx, wy]  # object가 덮으면 id가 달라진다
  return gt, mask
