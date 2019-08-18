export  PYTHONPATH=`pwd`/chofer_nips2017/chofer_torchex:/home/amahon/research/delve
module add Python/3.7.0-foss-2018b
sbatch -N 1 --exclusive batch.sh
