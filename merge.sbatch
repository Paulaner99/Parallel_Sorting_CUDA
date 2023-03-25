#!/bin/bash
#SBATCH --job-name=testMerge
#SBATCH --mail-type=ALL
#SBATCH --mail-user=riccardo.paolini5@studio.unibo.it
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputMerge
#SBATCH --gres=gpu:1

nvcc merge.cu -o merge.out
./merge.out