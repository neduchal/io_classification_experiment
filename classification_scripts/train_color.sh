#PBS -N train_color
#PBS -q default
#PBS -l walltime=18:00:00
#PBS -l select=1:ncpus=7:mem=50gb
module add python36-modules-gcc

cd /storage/plzen1/home/neduchal/projekty/inout/src/

python train_pca.py $ARGS1 "svm" "$PBS_JOBID.$1.pca" "64;128;256;512;1024"


