"""Crafter 래퍼.

- 낮밤 고정: 밤에는 engine.LocalView._light가 픽셀에 랜덤 노이즈를 넣어 지표를 오염시킨다.
- 72×72 렌더링: 9×9 타일 그리드에서 타일 하나가 정확히 8px.
- 정답 재질 맵: info['semantic']은 object id가 재질을 덮어쓰므로 world._mat_map을 따로 꺼낸다.
- crafter==1.8.3의 내부 속성(_world, _player, _update_time)에 의존한다.
"""
import crafter
import numpy as np

SIZE = 72
GRID = (9, 9)          # 전체 화면 타일 (가로, 세로)
VIEW = (9, 7)          # 월드 뷰 타일 (가로, 세로). 아래 9×2는 인벤토리
TILE = SIZE // GRID[0]  # 8px
VIEW_OFFSET = np.array([VIEW[0] // 2, VIEW[1] // 2])  # 플레이어의 월드 뷰 내 타일 좌표 (4, 3)

ACTIONS = list(crafter.constants.actions)  # 17개
A = {name: i for i, name in enumerate(ACTIONS)}
MOVES = {A['move_left']: (-1, 0), A['move_right']: (1, 0), A['move_up']: (0, -1), A['move_down']: (0, 1)}
OPPOSITE = {A['move_left']: A['move_right'], A['move_right']: A['move_left'],
            A['move_up']: A['move_down'], A['move_down']: A['move_up']}


class CrafterEnv(crafter.Env):

  def __init__(self, seed=None, length=1000, fixed_daylight=True):
    super().__init__(area=(64, 64), view=GRID, size=(SIZE, SIZE), length=length, seed=seed)
    self.fixed_daylight = fixed_daylight

  def _update_time(self):
    if self.fixed_daylight:
      self._world.daylight = 1.0
    else:
      super()._update_time()

  # 정답 상태 ------------------------------------------------------------
  @property
  def player_pos(self):
    return np.array(self._player.pos)

  @property
  def facing(self):
    return np.array(self._player.facing)

  def material_map(self):
    """(64, 64) uint8, [x, y] 인덱싱. object에 가려지지 않은 재질."""
    return self._world._mat_map.astype(np.uint8).copy()

  def semantic(self):
    return self._sem_view().astype(np.uint8)

  def material_names(self):
    return {i: n for n, i in self._world._mat_ids.items()}

  def set_inventory(self, **items):
    for name, value in items.items():
      self._player.inventory[name] = value

  def state(self):
    """프레임마다 저장할 정답 정보."""
    return dict(
        player_pos=self.player_pos.astype(np.int16),
        facing=self.facing.astype(np.int8),
        material=self.material_map(),
        semantic=self.semantic(),
    )


def tile_to_world(player_pos, tile_xy):
  """월드 뷰 타일 좌표 (x, y) → 월드 좌표."""
  return np.asarray(player_pos) + np.asarray(tile_xy) - VIEW_OFFSET


def world_to_tile(player_pos, world_xy):
  """월드 좌표 → 월드 뷰 타일 좌표. 시야 밖이면 None."""
  t = np.asarray(world_xy) - np.asarray(player_pos) + VIEW_OFFSET
  if 0 <= t[0] < VIEW[0] and 0 <= t[1] < VIEW[1]:
    return t
  return None


def in_view(player_pos, world_xy):
  return world_to_tile(player_pos, world_xy) is not None


def tile_patch(frame, tile_xy):
  """프레임(H, W, 3)에서 타일 (x, y)의 8×8 패치. 프레임은 [y, x] 인덱싱."""
  x, y = tile_xy
  return frame[..., y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE, :]
