#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:2
#SBATCH --job-name=zeroshotperf
#SBATCH --output /home/ma/ma_ma/ma_kyupark/is809/logs/slurm-%j.out

now=$(date +"%T")

echo "Program starts:  $now"
cd
source /home/ma/ma_ma/ma_kyupark/.bashrc
conda_initialize
micromamba activate is809

cd /home/ma/ma_ma/ma_kyupark/is809
echo "Zeroshot performance"
python codes/run_zeroshot.py --filename data/rawdata/synthdata.pkl --labeled_file data/output/validation/zerolabel.pkl --output_file data/output/validation/zeroperf.pkl


end=$(date +"%T")
echo "Completed: $end"