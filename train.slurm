#!/bin/bash
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --requeue
#SBATCH --account=scavenger
#SBATCH --qos scavenger
#SBATCH --partition scavenger
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx2080ti:1

cd /vulcanscratch/mukunds/downloads/TMR
export MKL_THREADING_LAYER=GNU
source /vulcanscratch/mukunds/anaconda3/bin/activate
conda activate tmr
export HYDRA_FULL_ERROR=1

# python train.py run_dir=outputs/new_lmd model.run_dir=outputs/new_lmd model.lmd.dtw=5.0 model.lmd.contrastive=0 model.dtw_loss_type="cosine" model.use_dtw=True model.dtw_margin=0.15 model.wandb_name="new_lmd" model.use_contrastive=False
python train.py run_dir=outputs/new_lmd model.run_dir=outputs/new_lmd model.lmd.dtw=5.0 model.lmd.contrastive=0 model.dtw_loss_type="cosine" model.use_dtw=True model.dtw_margin=0.15 model.wandb_name="new_lmd" model.use_contrastive=False lower=5 upper=8