"""브라우저 플레이 데모.

python src/web_demo.py                      # 기본: 실제 Crafter | baseline_v2 | memory_aligned_v3
python src/web_demo.py --models outputs/baseline/last.pt outputs/memory_aligned/last.pt --port 8765
→ VS Code가 포트를 자동 포워딩. 브라우저에서 http://localhost:8765

- 같은 키 입력을 실제 Crafter와 모든 모델에 넣어 나란히 보여준다. 시작 K프레임만 실제, 이후는 모델 출력만으로 진행.
- 키: WASD 이동 · Space do · 1 돌 · 2 table · 3 furnace · 4 plant · 5~0 제작 · N noop · R 처음부터 · G 녹화 시작/끝 (GIF 저장)
- 표준 라이브러리 http.server만 쓴다 (추가 패키지 없음).
"""
import argparse
import base64
import io
import json
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import imageio
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.dataset import FrameDataset  # noqa: E402
from src.env import A, CrafterEnv  # noqa: E402
from src.eval_revisit import load_model  # noqa: E402
from src.play import KEYS  # noqa: E402
from src.policies import random_inventory  # noqa: E402
from src.rollout import to_uint8  # noqa: E402

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Neural Crafter</title>
<style>
 body{background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:16px}
 .row{display:flex;gap:12px;flex-wrap:wrap}
 .panel{text-align:center} .panel img{width:min(30vw,288px);image-rendering:pixelated;border:1px solid #444}
 .info{margin-top:10px;font-size:14px;color:#aaa} kbd{background:#333;padding:1px 5px;border-radius:3px}
 .rec{color:#f55;font-weight:bold}
</style></head><body>
<div class="row" id="panels"></div>
<div class="info" id="status">loading…</div>
<div class="info"><kbd>WASD</kbd> 이동 · <kbd>Space</kbd> do · <kbd>1</kbd> 돌 <kbd>2</kbd> table <kbd>3</kbd> furnace
 <kbd>4</kbd> plant · <kbd>5~0</kbd> 제작 · <kbd>N</kbd> noop · <kbd>R</kbd> 처음부터 · <kbd>G</kbd> 녹화 (GIF 저장)</div>
<script>
let busy=false;
function render(d){
  const p=document.getElementById('panels');
  if(p.children.length!==d.panels.length){p.innerHTML='';
    d.panels.forEach(x=>{const e=document.createElement('div');e.className='panel';
      e.innerHTML='<img><div class="info"></div>';p.appendChild(e);});}
  d.panels.forEach((x,i)=>{const e=p.children[i];e.querySelector('img').src='data:image/png;base64,'+x.png;
    e.querySelector('.info').textContent=x.name+(x.ms!==null?'  ('+x.ms.toFixed(0)+' ms)':'');});
  document.getElementById('status').innerHTML='step '+d.step+(d.recording?' <span class="rec">● REC</span>':'')+(d.msg?' — '+d.msg:'');
}
async function send(key){ if(busy) return; busy=true;
  try{const r=await fetch('/key',{method:'POST',body:JSON.stringify({key})}); render(await r.json());}
  finally{busy=false;} }
document.addEventListener('keydown',e=>{const k=e.key===' '?' ':e.key.toLowerCase();
  if('wasd 1234567890nrg'.includes(k)&&k.length===1){e.preventDefault();send(k);}});
send('');
</script></body></html>"""


def png_b64(img, scale=1):
  buf = io.BytesIO()
  imageio.imwrite(buf, np.kron(img, np.ones((scale, scale, 1), np.uint8)), format='png')
  return base64.b64encode(buf.getvalue()).decode()


class Demo:

  def __init__(self, specs, seed, K, dev):
    self.models = [load_model(s, dev)[:2] for s in specs]  # (WorldModel, 이름)
    self.seed, self.K, self.dev = seed, K, dev
    self.lock = threading.Lock()
    self.recording, self.rec = False, []
    self.reset()

  def reset(self):
    self.env = CrafterEnv(seed=self.seed, length=100000)
    self.env.reset()
    random_inventory(self.env, np.random.default_rng(self.seed))
    self.env.set_inventory(stone=9, wood=9, sapling=3)
    frames = [self.env.render()]
    for _ in range(self.K - 1):  # 실제 프레임으로 context 채우기 (memory 모델은 이 프레임들도 기억에 쓴다)
      frames.append(self.env.step(A['noop'])[0])
    ctx = FrameDataset.to_float(torch.from_numpy(np.stack(frames)[None]).to(self.dev))
    self.states = [m.init_context(ctx) for m, _ in self.models]
    self.real = frames[-1]
    self.preds = [frames[-1] for _ in self.models]
    self.ms = [None for _ in self.models]
    self.steps = 0

  def key(self, k):
    msg = ''
    with self.lock:
      if k == 'r':
        self.reset()
      elif k == 'g':
        if self.recording and self.rec:
          path = pathlib.Path('outputs/play') / f'web_{int(time.time())}.gif'
          path.parent.mkdir(parents=True, exist_ok=True)
          imageio.mimsave(path, self.rec, duration=1 / 6, loop=0)
          msg = f'saved {path}'
        self.recording, self.rec = not self.recording, []
      elif k in KEYS:
        a = torch.tensor([A[KEYS[k]]], device=self.dev)
        for i, (m, _) in enumerate(self.models):
          t = time.time()
          frame, self.states[i] = m.step(self.states[i], a)
          self.preds[i] = to_uint8(frame[0])
          self.ms[i] = (time.time() - t) * 1000
        self.real = self.env.step(int(a))[0]
        self.steps += 1
        if self.recording:
          sep = np.full((72, 2, 3), 255, np.uint8)
          row = [self.real]
          for p in self.preds:
            row += [sep, p]
          self.rec.append(np.kron(np.concatenate(row, 1), np.ones((4, 4, 1), np.uint8)))
      panels = [dict(name='real Crafter', png=png_b64(self.real), ms=None)]
      panels += [dict(name=name, png=png_b64(p), ms=ms) for (_, name), p, ms in zip(self.models, self.preds, self.ms)]
      return dict(panels=panels, step=self.steps, recording=self.recording, msg=msg)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--models', nargs='+', default=['outputs/baseline_v2/last.pt', 'outputs/memory_aligned_v3/last.pt'])
  p.add_argument('--seed', type=int, default=2_000_000)
  p.add_argument('--context', type=int, default=16, help='처음에 실제 Crafter로 채울 프레임 수')
  p.add_argument('--port', type=int, default=8765)
  args = p.parse_args()
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  demo = Demo(args.models, args.seed, args.context, dev)

  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
      pass

    def _send(self, body, ctype):
      self.send_response(200)
      self.send_header('Content-Type', ctype)
      self.send_header('Content-Length', str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self):
      self._send(PAGE.encode(), 'text/html; charset=utf-8')

    def do_POST(self):
      n = int(self.headers.get('Content-Length', 0))
      k = json.loads(self.rfile.read(n) or b'{}').get('key', '')
      self._send(json.dumps(demo.key(k)).encode(), 'application/json')

  print(f'http://localhost:{args.port}  (models: {", ".join(n for _, n in demo.models)})', flush=True)
  ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
  main()
