#!/bin/bash
# Usage: [EVAL_TEST=1] sbatch slurm/train.sbatch configs/cornell_rgbd.yaml
# EVAL_TEST=1 also scores best.pt on the test split -- only for runs that are final, never for sweeps.
#SBATCH --job-name=grasp
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/miniconda3/bin/activate /mnt/users/$USER/envs/grasp
CONFIG=${1:?config path}; shift
echo "job $SLURM_JOB_ID on $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-?} config $CONFIG"
# Always try to resume: a requeued or timed-out job continues from checkpoints/<name>/last.pt
python -m grasp.train --config "$CONFIG" --resume "$@"
python scripts/eval_model.py --config "$CONFIG" --subsets val ${EVAL_TEST:+test}
