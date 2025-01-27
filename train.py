import os
import sys
import numpy as np
import torch
import copy
import argparse
from functools import partial

from src.channel_models import Toeplitz_channel

# Own libraries
sys.path.append('src/')
import trainer
import models
from channel_models import construct_Toeplitz_matrix
from utils import EbNo_to_noise
from utils_plot import *
import utils_standard_codec


'''
parser = argparse.ArgumentParser()
parser.add_argument("f_Dn", help="Normalized Doppler frequency. It should be between 0 to 1.")
parser.add_argument("g", help="gamma used for training. It should be between 0 to 1.")
parser.add_argument("ebn0", help="The training Eb/N0 value used for training." )
args = parser.parse_args()
'''
f_Dn = 0.1
g= 0
ebn0 = 15
## If you want to load a model trained previously.
LOADMODEL = False

## Communication and coding parameters
k = 6  # math.log2(M)
M = 2 ** k  # Number of different bit sequences (after source coding)
n = 4  # should be an even number to represent a complex channel
n_r = 2 * n  # dimension in real datatype. It beacomes 2 times.
rate = k / n  # bits per symbol rate. average power is n for a symbol.
P = 1 # number of paths
T = 1e-9 # Symbol period
tau_max = 2 * T

## Trainining Parameters.
auto_save = True
dataset_size = 2 ** 10  # ~10 ** 6
batch_size = 16
max_epoch = 20  # Set as 0 to turn off training the AE model.
lr_seq = [1e-3, 1e-4, 1e-5]
ep_seq = [20, 20, 20]

path = 'state_dict/'

## calculate the noise standard deviation
TRAINING_SNR = int(ebn0)
noise_std = EbNo_to_noise(TRAINING_SNR, rate)  # for a normalized signal (per symbol)

## Define the covariance matrix for communication channel
f_dopp_norm = float(f_Dn)

print("System Parameters")
print(f"Block Length: {n}, Bit Sequence Length: {k}, Rate: {k/n:.2} (bits/symbol)\n")

print("Communication Channel")
print(f"Normalized Doppler freqency: {f_dopp_norm}\n")

print("Sensing Channel")
print(f"Number of paths: {P}, Random Delay~U[0, {tau_max * 1e+9:.4}] ns.")


## Usage of GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


#dataset_m = np.random.randint(M, size=dataset_size)
loss_ratio_c_to_r = float(g) #input("Enter the value of gamma. It is the ratio of the communication loss to the sensing loss.") #0  # should be in (0,1)

## File tag for distinguishing the files from this simulation.
file_tag = f"_M{M:d}N{n:d}_{TRAINING_SNR}dB_f{f_dopp_norm:.3f}_gamma{loss_ratio_c_to_r:.1f}"

encoder_ML = models.Encoder_1h_n2_deeper_ee(M, n_r).to(device)
if LOADMODEL:  ## To load the model if necessary.
    encoder_ML.load_state_dict(torch.load(path + 'encoder' + file_tag + '.pt'))
    encoder_ML.eval()

for count, lr in enumerate(lr_seq):
    t_conf = trainer.TrainerConfig(M=M, n=n, rate=rate, f_dopp_norm=f_dopp_norm, noise_std_c=noise_std,
                                   P=P, T=T, tau_max=tau_max, noise_std_r=noise_std,
                                   max_epochs=ep_seq[count], dataset_size=dataset_size,
                                   batch_size=batch_size, lr_enc=lr,
                                   loss_ratio_c_to_r=loss_ratio_c_to_r, auto_save=auto_save,
                                   file_tag=file_tag, path=path, device=device)
    t_joint = trainer.Trainer(encoder_ML, t_conf, one_hot=True)
    _, _ = t_joint.train()
