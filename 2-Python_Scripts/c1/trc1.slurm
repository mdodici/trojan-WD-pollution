#!/bin/bash
#SBATCH --job-name=tr-a1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=13:30:00 
#SBATCH --mail-type=begin
#SBATCH --mail-type=end  
#SBATCH --mail-type=fail 
#SBATCH --mail-user=mdodici@princeton.edu

module load anaconda3/2020/11

python TrojA1k.py
