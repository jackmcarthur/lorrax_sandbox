#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=regular
#SBATCH --time=03:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --account=m2651
#SBATCH -J lx-alloc-jackm
sleep 10600
