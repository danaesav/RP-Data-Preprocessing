#!/bin/sh
#
#SBATCH --job-name="d2stgnn adj"
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G

module load 2023r1
module load python/3.9.8
module load py-pip/22.2.2
module load py-future
module load py-numpy
module load py-scipy

cd ..

pip install -r requirements.txt

srun  python main.py  > output_train.log
