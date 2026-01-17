import torch
import os

# CPU Configuration for 32-core system
NUM_THREADS = 1
torch.set_num_threads(NUM_THREADS)
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
os.environ['MKL_NUM_THREADS'] = str(NUM_THREADS)

Z_DIM = 64
D_HIDDEN = 80
BATCH_SIZE = 4096
LR_EMBED = 1e-4
LR_SUR = 1e-3
DEVICE = torch.device("cuda")
