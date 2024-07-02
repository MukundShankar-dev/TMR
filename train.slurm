#!/bin/bash
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --requeue
#SBATCH --account=scavenger
#SBATCH --qos scavenger
#SBATCH --partition scavenger
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx2080ti:1

cd /vulcanscratch/mukunds/downloads/TMR
export MKL_THREADING_LAYER=GNU
source /vulcanscratch/mukunds/anaconda3/bin/activate
conda activate tmr
export HYDRA_FULL_ERROR=1

python train.py model.lmd.dtw=10.0 model.lmd.contrastive=0.9 model.log_wandb=True model.dtw_loss_type=other model.use_dtw=True model.dtw_margin=0.15 model.use_contrastive=True lower=10 upper=13 model.wandb_name=new_infoNCE_attempt