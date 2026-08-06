#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --constraint="gpu&hbm80g"
#SBATCH --gpus=4
#SBATCH --account=m2651
#SBATCH -J lx-alloc-jackm
sleep 21400
