#!/bin/bash
#SBATCH -J falcon_inference        # job name
#SBATCH -o falcon_inference.o%j    # output and error file name (%j expands to jobID)
#SBATCH -n 1                # total number of tasks requested
#SBATCH -N 1                # number of nodes you want to run on
#SBATCH --cpus-per-task 16  # request cores (64 per node)
#SBATCH --gres=gpu:1        # request a gpu (4 per node)
#SBATCH -p gpu-l40          # queue (partition)
#SBATCH -t 168:00:00         # run time (hh:mm:ss)
#SBATCH --mail-type=end
#SBATCH --mail-user=X

module purge
module load pytorch_2.4.1_tensorflow_2.17_cuda_12.4

# Run your Python training script
python /X/X/X/X-Project-1/models/master/cross-validation/falcon/falcon_inference.py