#!/bin/bash
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --requeue
#SBATCH --account vulcan-abhinav
#SBATCH --qos vulcan-scavenger
#SBATCH --partition vulcan-scavenger
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task 8
#SBATCH --exclude cbcb25
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx2080ti:1

cd /vulcanscratch/mukunds/downloads/TMR
export MKL_THREADING_LAYER=GNU
source /vulcanscratch/mukunds/anaconda3/bin/activate
conda activate tmr
export HYDRA_FULL_ERROR=1

python train.py run_dir=outputs/zerod_smplh model.run_dir=outputs/zerod_smplh model.lmd.text_contrastive=0.9 model.lmd.motion_contrastive=0.9 model.lmd.dtw=0.9 model.lmd.contrastive=0.9 model.dtw_loss_type="cosine" model.use_dtw=True model.dtw_margin=0.15 model.wandb_name="zerod_smplh" model.use_contrastive=True lower=10 upper=13 model.log_wandb=True

# python train.py run_dir=outputs/combined_infonce model.run_dir=outputs/combined_infonce model.lmd.text_contrastive=0.9 model.lmd.motion_contrastive=0.9 model.lmd.dtw=0.9 model.lmd.contrastive=0.9 model.dtw_loss_type="cosine" model.use_dtw=True model.dtw_margin=0.15 model.wandb_name="combined_infonce" model.use_contrastive=True lower=10 upper=13 model.log_wandb=True