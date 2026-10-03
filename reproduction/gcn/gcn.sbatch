#!/bin/bash
#SBATCH --job-name=gcn
#SBATCH --account=YOUR_PROJECT
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
#SBATCH --output=gcn_%j.log
# GCN (PLOS Comput Biol 2025) reruns: 6 runs, 3 per A100. If the job ends early, `sbatch gcn.sbatch` again.
cd "$HOME/leakage"
module load miniconda3/24.1.2-py310
source activate diffmic
nvidia-smi
python run_gcn.py
