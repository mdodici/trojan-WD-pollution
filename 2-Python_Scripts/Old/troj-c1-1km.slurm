#!/bin/bash
#SBATCH --jup-name=tr-c1-1k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=9G
#SBATCH --time=07:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mdodici@princeton.edu

module load anaconda3/2020.11

python Trojans_C1_1km.py