srun -p $1 -n$2 --gres=gpu:$2  --ntasks-per-node=$2 --job-name=5 python -u main_allreduce.py -j 16 -b 16 -a resnet152 2>&1 | tee log7.txt
