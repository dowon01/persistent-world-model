"""pygame 키보드 플레이 데모. 모든 WorldModel 공용.

python src/play.py --model outputs/baseline/last.pt            # 창에서 플레이
python src/play.py --model outputs/baseline/last.pt --keys "dddd1aaaaaaaadddddddd" --gif outputs/demo.gif
                                                               # 화면 없이 키 시퀀스로 GIF만 만들기

- 시작 K프레임은 실제 Crafter, 이후는 모델 출력만으로 진행 (키 하나 = 한 스텝, Crafter는 턴제).
- --compare(기본 켜짐): 같은 행동을 실제 Crafter에도 넣어 왼쪽 실제 / 오른쪽 모델로 나란히 보여준다.
  "돌아왔는데 설치한 블록이 사라지는 장면"을 바로 비교할 수 있다.

키: WASD 이동 · Space do · 1 돌 · 2 table · 3 furnace · 4 plant · 5~0 제작 · N noop
    G 녹화 시작/끝 (GIF 저장) · R 처음부터 · Esc 종료
"""
import argparse
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.dataset import FrameDataset  # noqa: E402
from src.env import A, CrafterEnv  # noqa: E402
from src.eval_revisit import load_model  # noqa: E402
from src.policies import random_inventory  # noqa: E402
from src.rollout import to_uint8  # noqa: E402

KEYS = {'w': 'move_up', 'a': 'move_left', 's': 'move_down', 'd': 'move_right', ' ': 'do', 'n': 'noop',
        '1': 'place_stone', '2': 'place_table', '3': 'place_furnace', '4': 'place_plant',
        '5': 'make_wood_pickaxe', '6': 'make_stone_pickaxe', '7': 'make_iron_pickaxe',
        '8': 'make_wood_sword', '9': 'make_stone_sword', '0': 'make_iron_sword'}


class Session:

  def __init__(self, model, K, seed, dev):
    self.model, self.K, self.seed, self.dev = model, K, seed, dev
    self.reset()

  def reset(self):
    self.env = CrafterEnv(seed=self.seed, length=100000)
    self.env.reset()
    random_inventory(self.env, np.random.default_rng(self.seed))
    self.env.set_inventory(stone=9, wood=9, sapling=3)
    frames = [self.env.render()]
    for _ in range(self.K - 1):  # 실제 프레임으로 context 채우기
      frames.append(self.env.step(A['noop'])[0])
    ctx = FrameDataset.to_float(torch.from_numpy(np.stack(frames)[None]).to(self.dev))
    self.state = self.model.init_context(ctx)
    self.real, self.pred = frames[-1], frames[-1]
    self.steps = 0

  def step(self, action):
    t = time.time()
    frame, self.state = self.model.step(self.state, torch.tensor([action], device=self.dev))
    self.pred = to_uint8(frame[0])
    dt = time.time() - t
    self.real = self.env.step(action)[0]
    self.steps += 1
    return dt

  def canvas(self, compare, scale):
    f = np.concatenate([self.real, np.full((72, 2, 3), 255, np.uint8), self.pred], 1) if compare else self.pred
    return np.kron(f, np.ones((scale, scale, 1), np.uint8))


def save_gif(path, frames):
  import imageio
  pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(path, frames, duration=1 / 6, loop=0)
  print('saved', path)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--model', required=True)
  p.add_argument('--seed', type=int, default=2_000_000)
  p.add_argument('--context', type=int, default=4)
  p.add_argument('--no_compare', action='store_true')
  p.add_argument('--scale', type=int, default=6)
  p.add_argument('--keys', default=None, help='화면 없이 이 키 시퀀스를 실행하고 GIF로 저장')
  p.add_argument('--gif', default='outputs/play/demo.gif')
  args = p.parse_args()
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model, *_ = load_model(args.model, dev)
  sess = Session(model, args.context, args.seed, dev)
  compare = not args.no_compare

  if args.keys is not None:  # headless
    frames = [sess.canvas(compare, args.scale)]
    for k in args.keys:
      sess.step(A[KEYS[k]])
      frames.append(sess.canvas(compare, args.scale))
    save_gif(args.gif, frames)
    return

  import pygame
  pygame.init()
  img = sess.canvas(compare, args.scale)
  screen = pygame.display.set_mode((img.shape[1], img.shape[0] + 24))
  pygame.display.set_caption('Neural Crafter — left: real, right: model' if compare else 'Neural Crafter')
  font = pygame.font.SysFont(None, 22)
  recording, rec, ms = False, [], 0.0
  while True:
    for ev in pygame.event.get():
      if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
        pygame.quit()
        return
      if ev.type == pygame.KEYDOWN:
        ch = ev.unicode.lower()
        if ch == 'r':
          sess.reset()
        elif ch == 'g':
          if recording and rec:
            save_gif(args.gif.replace('.gif', f'_{int(time.time())}.gif'), rec)
          recording, rec = not recording, []
        elif ch in KEYS:
          ms = sess.step(A[KEYS[ch]]) * 1000
          if recording:
            rec.append(sess.canvas(compare, args.scale))
    img = sess.canvas(compare, args.scale)
    screen.fill((0, 0, 0))
    screen.blit(pygame.surfarray.make_surface(img.transpose(1, 0, 2)), (0, 0))
    txt = f'step {sess.steps}  model {ms:.0f} ms/step ({1000 / max(ms, 1e-3):.0f} FPS)' + ('  [REC]' if recording else '')
    screen.blit(font.render(txt, True, (255, 255, 255)), (6, img.shape[0] + 4))
    pygame.display.flip()
    pygame.time.wait(10)


if __name__ == '__main__':
  main()
