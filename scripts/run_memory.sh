#!/bin/bash
# memory M1 두 모델을 차례로 학습하고, 끝나면 평가·그래프까지 자동으로 이어서 실행한다.
# 세션이 끊겨도 결과가 남도록 nohup으로 실행: nohup bash scripts/run_memory.sh > outputs/run_memory.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcwm
for mode in aligned raw; do
  echo "=== train memory_$mode $(date)"
  python src/train.py --config configs/memory_$mode.yaml > outputs/train_memory_$mode.log 2>&1
  echo "=== eval memory_$mode $(date)"
  python -W ignore src/eval_revisit.py --model outputs/memory_$mode/last.pt --gif_dir outputs/gifs > outputs/eval_memory_$mode.log 2>&1
done
echo "=== eval memory_aligned + oracle position $(date)"
python -W ignore src/eval_revisit.py --model outputs/memory_aligned/last.pt --oracle_pos > outputs/eval_memory_aligned_oracle.log 2>&1
python scripts/plot_results.py --point back --runs outputs/baseline outputs/baseline_k16 outputs/memory_aligned outputs/memory_raw
python scripts/plot_results.py --point final --runs outputs/baseline outputs/baseline_k16 outputs/memory_aligned outputs/memory_raw
echo "=== done $(date)"
