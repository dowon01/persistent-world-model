#!/bin/bash
# v2 (상호작용 샘플링 + 바뀐 칸 가중치, memory 학습 검색 규칙 recent) 학습 → 평가 → 그래프를 한 번에.
# nohup bash scripts/run_v2.sh > outputs/run_v2.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcwm
ev() { python -W ignore src/eval_revisit.py "$@"; }
for cfg in baseline_v2 memory_aligned_v2; do
  echo "=== train $cfg $(date)"
  python src/train.py --config configs/$cfg.yaml > outputs/train_$cfg.log 2>&1
  echo "=== eval $cfg $(date)"
  ev --model outputs/$cfg/last.pt --gif_dir outputs/gifs > outputs/eval_$cfg.log 2>&1
  ev --model outputs/$cfg/last.pt --start before --gif_dir outputs/gifs_before > outputs/eval_${cfg}_before.log 2>&1
done
echo "=== extra evals $(date)"
ev --model outputs/memory_aligned_v2/last.pt --oracle_pos > outputs/eval_memory_aligned_v2_oracle.log 2>&1
for m in baseline memory_aligned; do  # v1도 '처음부터 rollout'으로 비교
  ev --model outputs/$m/last.pt --start before > outputs/eval_${m}_before.log 2>&1
done
ev --model b0 --start before > outputs/eval_b0_before.log 2>&1
python -W ignore src/eval_change.py --models outputs/baseline/last.pt outputs/memory_aligned/last.pt \
  outputs/baseline_v2/last.pt outputs/memory_aligned_v2/last.pt > outputs/eval_change.log 2>&1
R="outputs/baseline outputs/baseline_k16 outputs/memory_aligned outputs/baseline_v2 outputs/memory_aligned_v2"
python scripts/plot_results.py --point back --exclude 'start_before|oracle|raw' --runs $R
python scripts/plot_results.py --point back --include 'start_before' --tag _start_before --runs $R
echo "=== done $(date)"
