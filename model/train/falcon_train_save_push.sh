#!/bin/bash
#SBATCH -J falcon_train_save_push        # job name
#SBATCH -o falcon_train_save_push.o%j    # output and error file name (%j expands to jobID)
#SBATCH -n 1                             # total number of tasks requested
#SBATCH -N 1                             # number of nodes to run on
#SBATCH --cpus-per-task=16               # request cores
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 168:00:00                     # run time (hh:mm:ss)
#SBATCH --mail-type=end
#SBATCH --mail-user=X

. ~/.bashrc
mamba activate X

# Run your Python training script
python /X/X/X/X-Project-1/models/master/cross-validation/falcon/falcon_train_save_push.py