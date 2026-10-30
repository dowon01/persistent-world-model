"""README용 데모 GIF: 실제 Crafter | baseline (memory 없음) | memory 를 같은 키 입력으로 나란히.

돌을 놓고 → 멀리 갔다가 → 같은 길로 돌아오는 키 시퀀스를 여러 seed에 돌려,
실제 env에서 이동이 모두 성공하고 돌아왔을 때 돌이 보이는 seed를 고른다.
(시작 16프레임만 실제 Crafter, 이후는 각 모델의 출력만으로 진행)
python scripts/make_demo_gif.py --out assets/demo.gif
"""
import argparse
import pathlib
import sys

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.env import CrafterEnv, tile_patch  # noqa: E402
from src.tiles import TileClassifier  # noqa: E402
from src.web_demo import Demo  # noqa: E402

KEYS = 'd1' + 'a' * 8 + 'wwssss' + 'ww' + 'd' * 8 + 'nn'   # 오른쪽에 돌 → 왼쪽으로 8칸 → 위아래 → 돌아옴
LABELS = ['Real Crafter', 'Baseline (no memory)', 'Memory']


def frame_with_labels(imgs, scale=4):
  pad = 18
  w = 72 * scale
  canvas = Image.new('RGB', (len(imgs) * (w + 6) - 6, w + pad), (20, 20, 20))
  draw = ImageDraw.Draw(canvas)
  for i, (img, label) in enumerate(zip(imgs, LABELS)):
    x = i * (w + 6)
    canvas.paste(Image.fromarray(np.kron(img, np.ones((scale, scale, 1), np.uint8))), (x, pad))
    draw.text((x + 4, 3), label, fill=(230, 230, 230))
  return np.array(canvas)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--baseline', default='outputs/baseline_v2/last.pt')
  p.add_argument('--memory', default='outputs/memory_aligned_v3/last.pt')
  p.add_argument('--out', default='assets/demo.gif')
  p.add_argument('--seeds', type=int, default=40)
  args = p.parse_args()
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  clf = TileClassifier(CrafterEnv(seed=0))
  stone = {n: i for i, n in CrafterEnv(seed=0).material_names().items()}['stone']
  demo = None
  for seed in range(3_000_000, 3_000_000 + args.seeds):
    if demo is None:
      demo = Demo([args.baseline, args.memory], seed, 16, dev)
    else:
      demo.seed = seed
      demo.reset()
    frames = [frame_with_labels([demo.real] + demo.preds)]
    start = None
    for i, k in enumerate(KEYS):
      demo.key(k)
      if i == 1:
        start = demo.env.player_pos.copy()
      frames.append(frame_with_labels([demo.real] + demo.preds))
    right = lambda img: int(clf.classify_patches(tile_patch(img, (5, 3)))[0])  # noqa: E731  플레이어 오른쪽 칸
    real_ok = np.array_equal(demo.env.player_pos, start) and right(demo.real) == stone
    base, mem = right(demo.preds[0]), right(demo.preds[1])
    print(f'seed {seed}: 실제 경로 {"OK" if real_ok else "막힘"}, baseline 돌 {base == stone}, memory 돌 {mem == stone}')
    if real_ok and mem == stone and base != stone:
      pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
      frames += [frames[-1]] * 6  # 마지막 장면 잠깐 멈춤
      imageio.mimsave(args.out, frames, duration=0.25, loop=0)
      print('saved', args.out, f'({len(frames)} frames, seed {seed})')
      return
  print('조건을 만족하는 seed를 찾지 못함')


if __name__ == '__main__':
  main()
