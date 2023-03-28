#!/bin/bash
#SBATCH --job-name=testBitonic
#SBATCH --mail-type=ALL
#SBATCH --mail-user=riccardo.paolini5@studio.unibo.it
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputBitonic
#SBATCH --gres=gpu:1

nvcc bitonic.cu -o bitonic.out
./bitonic.out