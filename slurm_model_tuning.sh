#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --account=yonghui.wu
#SBATCH --qos=yonghui.wu
#SBATCH --mail-type=NONE
#SBATCH --mail-user=NONE
#SBATCH --ntasks=20                 # Number of MPI tasks (i.e. processes)
#SBATCH --cpus-per-task=2           # Number of cores per MPI task 
#SBATCH --gpus=1
#SBATCH --nodes=1                    # Maximum number of nodes to be allocated
#SBATCH --gpus-per-task=0 
#SBATCH --mem=16g
#SBATCH --time=10:00:00
#SBATCH --output=/blue/bianjiang/huangyu/logs/task_%j.out
#SBATCH --partition=hpg-ai
pwd; hostname; date

module load conda
conda activate aif360

profile=job_${SLURM_JOB_ID}
 
echo "Creating profile_${profile}"
ipython profile create ${profile}
 
ipcontroller --ip="*" --profile=${profile} &
sleep 10
 
#srun: runs ipengine on each available core
srun ipengine --profile=${profile} --location=$(hostname) &
sleep 25
 
echo "Launching job for script"
python3 model_tuning.py -s $1 -p ${profile}
