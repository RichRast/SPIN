#!/bin/bash
#SBATCH -J NPT_imputation                      # Job name
#SBATCH -t 96:00:00                          # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:3090:1
#SBATCh --ntasks=1
#SBATCH --mem=60000
#SBATCH --array=0

source /home/rr568/NPT/non-parametric-transformers/scripts/init_env.sh

echo 'Begin npt'
free -h 
python $USER_PATH/image_npt.py --model_checkpoint_key=resnet_npt --train_batch_size=480 --train_at_test_batch_size=400 --test_batch_size=80 --use_pretrained_resnet=False --npt_type=npt --train_label_masking_perc=0.5 --project Image_Data --exp_name Resnet_NPT --data_set 'cifar-10'
echo 'Submission finished'
# sample command sbatch --requeue ./scripts/sbatch_image.sub
