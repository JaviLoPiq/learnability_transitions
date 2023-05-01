#!/bin/bash
#SBATCH -a 1-10
#SBATCH -c 1 # Number of Cores per Task
#SBATCH --mem=1000  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -t 4:00:00  # Job time limit
#SBATCH -o data/slurm-%j.out  # %j = job ID
#SBATCH -e data/err-%j.out

module load python/3.11.0
module load miniconda
conda activate mipt
python U1MRC_cluster.py $SLURM_ARRAY_TASK_ID 
