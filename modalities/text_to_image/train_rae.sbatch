#!/bin/bash
#SBATCH --job-name=rae-ditdhs-raemean
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=14-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out

# Trains the diffusion model of the text-to-image experiments (App. B): DiT-DH-S on RAE DINOv2-B latents, conditioned
# on the mean-pooled RAE latent of the image, on ImageNet-1K train for 400 epochs, global batch 256 on 4 A40 GPUs, fp32.
# The paper uses the checkpoint written at step 2,010,000 (checkpoints/ep-0000399.pt, in the last epoch).
#
#     sbatch modalities/text_to_image/train_rae.sbatch /path/to/imagenet/train   (from the repository root)
#
# The job survives preemption and the time limit: SLURM signals this shell 300 s early, the trap drops a PREEMPT
# marker (which the training loop polls) and forwards USR1 to the ranks, which checkpoint, requeue the job and exit.
# On restart, train.py resumes from the newest checkpoint of the experiment directory.

set -u
DATA_PATH=${1:?usage: sbatch train_rae.sbatch <ImageNet train folder>}
REPOSITORY=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}  # sbatch runs a copy of this file: submit from the repository root
RESULTS_DIR=${UNPAIRED_ROSETTA_ROOT:-$REPOSITORY/storage}/checkpoints/rae
export EXPERIMENT_NAME=ditdh-s_dinov2b_raemean
cd "$REPOSITORY/external/RAE"  # the config names the RAE decoder and statistics relative to this folder

# USR1 is ignored along the exec chain (pixi, torchrun), so only the training ranks, which install their own
# handler, react to the forwarded signal.
(
    trap '' USR1
    exec env PYTHONDONTWRITEBYTECODE=1 pixi run --manifest-path "$REPOSITORY/pyproject.toml" -e text2image \
        torchrun --standalone --nproc_per_node=4 src/train.py \
        --config "$REPOSITORY/modalities/text_to_image/rae/DiTDH-S_DINOv2-B_raemean.yaml" \
        --data-path "$DATA_PATH" \
        --results-dir "$RESULTS_DIR" \
        --precision fp32 \
        --compile
) &
training_pid=$!

forward_preemption() {
    echo "[sbatch] USR1 received; dropping the PREEMPT marker and forwarding the signal"
    touch "$RESULTS_DIR/$EXPERIMENT_NAME/PREEMPT"
    pkill -USR1 -f "src/train.py" || echo "[sbatch] no training rank found"
}
trap forward_preemption USR1

while kill -0 $training_pid 2>/dev/null; do
    wait $training_pid
done
echo "[sbatch] training process exited"
