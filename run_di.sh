#!/bin/bash  
#
#SBATCH --job-name=yolodi
#
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1 

#SBATCH --qos=preemptible
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:TeslaA100:1 
#SBATCH --mail-type='None'
#SBATCH --mail-user='dany.rimez@uclouvain.be' 
#SBATCH --output='/auto/home/users/d/a/darimez/slurmJob10.out'
#SBATCH --error='/auto/home/users/d/a/darimez/slurmJob10.err'

cd /auto/home/users/d/a/darimez/FedYOLO/
python src/train_DI.py