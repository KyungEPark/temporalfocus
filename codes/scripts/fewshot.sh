#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:2
#SBATCH --job-name=fewshotperf
#SBATCH --output /home/ma/ma_ma/ma_kyupark/is809/logs/slurm-%j.out


now=$(date +"%T")

echo "Program starts:  $now"
cd
source /home/ma/ma_ma/ma_kyupark/.bashrc
conda_initialize
micromamba activate is809

cd /home/ma/ma_ma/ma_kyupark/is809

n_values=(1)

for n in "${n_values[@]}"
do
    labeled_file="data/output/validation/few${n}labeled.pkl"
    output_file="data/output/validation/few${n}perf.pkl"
    
    echo "Running with n=$n"
    
    python codes/run_fewshot.py --filename data/rawdata/synthdata.pkl --n $n --labeled_file $labeled_file --output_file $output_file
    
    echo "Completed run with n=$n"
done


end=$(date +"%T")
echo "Completed: $end"