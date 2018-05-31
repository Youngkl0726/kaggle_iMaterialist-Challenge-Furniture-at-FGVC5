srun -p AD2 -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=test python -u test.py
