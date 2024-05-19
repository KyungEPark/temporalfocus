#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=4000
#SBATCH --gres=gpu:1
#SBATCH --job-name=testfinetune
#SBATCH --output /home/ma/ma_ma/ma_kyupark/is809/logs/slurm-%j.out

now=$(date +"%T")

echo "Program starts:  $now"
cd
source /home/ma/ma_ma/ma_kyupark/.bashrc
conda_initialize
micromamba activate is809

echo "Test finetuning code"
cd /home/ma/ma_ma/ma_kyupark/is809

python codes/run_training.py --bertn 3 --liwcn 3 --output_file data/models/finetuned_models/2_4_10_60_test


end=$(date +"%T")
echo "Completed: $end"