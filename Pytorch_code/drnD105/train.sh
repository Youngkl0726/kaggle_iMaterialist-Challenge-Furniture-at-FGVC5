srun -p $1 -n$2 --gres=gpu:$2 --ntasks-per-node=$2 --job-name=22 python -u main_allreduce.py -j 16 -b 24 2>&1 | tee log2.txt

