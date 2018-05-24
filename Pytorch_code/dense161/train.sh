srun -p VIBackEnd2 -n2 --gres=gpu:2 --ntasks-per-node=2 --job-name=3 python -u main_allreduce.py -j 16 -b 32 -a densenet161 2>&1 | tee log4.txt

