#!/bin/bash
#SBATCH --job-name=fml_training
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4 
#SBATCH --mem=4gb
#SBATCH --time=10:00:00
#SBATCH --output=/blue/bianjiang/huangyu/logs/task_%j.out
#SBATCH --partition=hpg-default

pwd; hostname; date

module load conda
conda activate aif360

python3 $1 -s $2 -g $3 -m $4