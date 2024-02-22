#!/bin/bash  
#
#SBATCH --job-name=fedyolo
#
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1 

#SBATCH --mem-per-cpu=7000M
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:GeForceRTX2080Ti:1 
#SBATCH --mail-type='None'
#SBATCH --mail-user='dany.rimez@uclouvain.be' 
#SBATCH --output='/auto/home/users/d/a/darimez/slurmJob.out'
#SBATCH --error='/auto/home/users/d/a/darimez/slurmJob.err'

cd $LOCALSCRATCH
tar -xf /CECI/home/users/d/a/darimez/WALT.tar
tar -xf /CECI/home/users/d/a/darimez/walt_test.tar
for cam in 1 2 3 4 5 6 7 8 9
do 
    rm WALT/cam$cam/test/* -r
    mv walt_test/cam$cam/* WALT/cam$cam/test
done
cd /auto/home/users/d/a/darimez/FedYOLO 
# ulimit -n 4096

python src/train.py init_model=yolov8n.pt optimiser.epochs=50 data.path=[[$LOCALSCRATCH/WALT/cam4/train,$LOCALSCRATCH/WALT/cam5/train,$LOCALSCRATCH/WALT/cam6/train,$LOCALSCRATCH/WALT/cam7/train,$LOCALSCRATCH/WALT/cam8/train],[$LOCALSCRATCH/WALT/cam1/train,$LOCALSCRATCH/WALT/cam3/train],[$LOCALSCRATCH/WALT/cam2/train,$LOCALSCRATCH/WALT/cam9/train]]
# sh run.sh  model.augmentation_percent=0.5 ml.augmentation_percent=0.5 model.cn_use=lllyasviel_canny active.abled=True
# sh run.sh coco model.augmentation_percent=0.5 ml.augmentation_percent=0.5 model.cn_use=lllyasviel_canny active.abled=True ml.train_nb=250
# sh run.sh create_dataset model.augmentation_percent=0.0 ml.augmentation_percent=0.0 model.cn_use=controlnet_segmentation active.abled=True logs.experiment=active logs.sampling=hung_coreset active.rounds=6 active.sampling=coreset
# sh run.sh train model.augmentation_percent=0.0 ml.augmentation_percent=0.0 model.cn_use=controlnet_segmentation active.abled=True logs.experiment=active logs.sampling=confidence active.rounds=6 active.sampling=confidence

# python -c "from subprocess import Popen ; cmd = 'sh run.sh create_dataset model.augmentation_percent=0.0 ml.augmentation_percent=0.0 model.cn_use=controlnet_segmentation active.abled=True logs.experiment=active logs.sampling=hung_coreset active.rounds=6 active.sampling=coreset'.split() ; proc1 = Popen(cmd, shell=False) ; cmd = 'sh run.sh create_dataset model.augmentation_percent=0.0 ml.augmentation_percent=0.0 model.cn_use=lllyasviel_canny active.abled=True logs.experiment=active logs.sampling=hung_coreset active.rounds=6 active.sampling=coreset'.split() ; proc2 = Popen(cmd, shell=False) ; cmd = 'sh run.sh create_dataset model.augmentation_percent=0.0 ml.augmentation_percent=0.0 model.cn_use=lllyasviel_openpose active.abled=True logs.experiment=active logs.sampling=hung_coreset active.rounds=6 active.sampling=coreset'.split() ; proc3 = Popen(cmd, shell=False) ; cmd = 'sh run.sh create_dataset model.augmentation_percent=0.0 ml.augmentation_percent=0.0 model.cn_use=crucible_mediapipe_face active.abled=True logs.experiment=active logs.sampling=hung_coreset active.rounds=6 active.sampling=coreset'.split() ; proc4 = Popen(cmd, shell=False) ; proc1.communicate() ; proc2.communicate() ; proc3.communicate() ; proc4.communicate()"  


# python -c "from subprocess import Popen ; cmd = 'sh run.sh train model.augmentation_percent=0.0 ml.augmentation_percent=0.0 model.cn_use=controlnet_segmentation active.abled=True logs.experiment=active logs.sampling=hung_coreset active.rounds=6 active.sampling=coreset'.split() ; proc1 = Popen(cmd, shell=False) ; cmd = 'sh run.sh train model.augmentation_percent=0.0 ml.augmentation_percent=0.0 model.cn_use=lllyasviel_canny active.abled=True logs.experiment=active logs.sampling=hung_coreset active.rounds=6 active.sampling=coreset'.split() ; proc2 = Popen(cmd, shell=False) ; cmd = 'sh run.sh train model.augmentation_percent=0.0 ml.augmentation_percent=0.0 model.cn_use=lllyasviel_openpose active.abled=True logs.experiment=active logs.sampling=hung_coreset active.rounds=6 active.sampling=coreset'.split() ; proc3 = Popen(cmd, shell=False) ; cmd = 'sh run.sh train model.augmentation_percent=0.0 ml.augmentation_percent=0.0 model.cn_use=crucible_mediapipe_face active.abled=True logs.experiment=active logs.sampling=hung_coreset active.rounds=6 active.sampling=coreset'.split() ; proc4 = Popen(cmd, shell=False) ; proc1.communicate() ; proc2.communicate() ; proc3.communicate() ; proc4.communicate()"
 








