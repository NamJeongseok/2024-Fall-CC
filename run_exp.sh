#!/bin/sh
#SBATCH -J <your_project_name>
#SBATCH -p edu_2080ti
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:1

module purge
module load conda/miniconda
source /home/edgpu/<your_accont_name>/.bashrc
conda activate <your_env_name>

srun python3 <file_name>.py