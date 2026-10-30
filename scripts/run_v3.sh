#!/bin/bash
# v3: memory_aligned_v3 (실패한 설치를 그리는 문제 대응) + baseline_v2 seed 1 (짧은 N 결과가 run 간 편차인지 판별).
# nohup bash scripts/run_v3.sh > outputs/run_v3.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcwm
ev() { python -W ignore src/eval_revisit.py "$@"; }
for cfg in memory_aligned_v3 baseline_v2_s1; do
  echo "=== train $cfg $(date)"
  python src/train.py --config configs/$cfg.yaml > outputs/train_$cfg.log 2>&1
  echo "=== eval $cfg $(date)"
  ev --model outputs/$cfg/last.pt --gif_dir outputs/gifs > outputs/eval_$cfg.log 2>&1
  ev --model outputs/$cfg/last.pt --start before --gif_dir outputs/gifs_before > outputs/eval_${cfg}_before.log 2>&1
done
echo "=== analysis $(date)"
ev --model outputs/memory_aligned_v3/last.pt --oracle_pos > outputs/eval_memory_aligned_v3_oracle.log 2>&1
python -W ignore src/eval_change.py --models outputs/baseline_v2/last.pt outputs/baseline_v2_s1/last.pt \
  outputs/memory_aligned_v2/last.pt outputs/memory_aligned_v3/last.pt > outputs/eval_change_v3.log 2>&1
python -W ignore scripts/analyze_false_change.py outputs/baseline_v2/last.pt outputs/memory_aligned_v2/last.pt \
  outputs/memory_aligned_v3/last.pt > outputs/analyze_false_change_v3.log 2>&1
python -W ignore scripts/analyze_short_n.py outputs/baseline/last.pt outputs/baseline_v2/last.pt \
  outputs/baseline_v2_s1/last.pt outputs/memory_aligned_v3/last.pt > outputs/analyze_short_n_v3.log 2>&1
R="outputs/baseline outputs/memory_aligned outputs/baseline_v2 outputs/baseline_v2_s1 outputs/memory_aligned_v2 outputs/memory_aligned_v3"
python scripts/plot_results.py --point back --exclude 'start_before|oracle|raw|k16' --runs $R
python scripts/plot_results.py --point back --include 'start_before' --tag _start_before --runs $R
echo "=== done $(date)"
