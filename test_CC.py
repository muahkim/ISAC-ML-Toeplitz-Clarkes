import sys
import torch
import math
from functools import partial

from datetime import datetime
# Own libraries
sys.path.append('src')

from src.get_args import get_args
from src.utils_standard_codec import test_ER_CC_Mod_batched
import src.utils_standard_codec as utils_standard_codec
from src.channel_models import *

#########################################
# Turbo AE requires additional arguments and use arguments. All can be turned on by TURBO flag in the source code.
#########################################

args = get_args()
print(args)

timestamp=datetime.now().strftime("%m%d%H%M%S")

###############################################
# Set the device to train
###############################################
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#################################################
# Set the channel parameters and learning model parameters. To be moved to args file later on.
#################################################

# Generating covariance based on Clakre's model
f_Dn = args.f_Dn            # In case Clarke's model used, the normalized Doppler frequency decides the covariance matrix for communication channel.
f_dopp_norm = float(f_Dn)   # Define the covariance matrix for communication channel

T = args.T                  # Symbol period
P = args.P                  # number of paths in sensing channel
tau_max = 2 * args.T  # maximum delay of paths. Delay follows a uniform distribution U[0, tau_max].


k = args.block_len * args.code_rate_k
block_length = 9
M_mod = 2
GLRT = True
mod_type = args.mod_type #'DPSK'

if mod_type == 'ASK':
    n = int(args.block_len * args.code_rate_n / math.log2(M_mod))
    rate = args.code_rate_k / args.code_rate_n  # only the channel coding part without considering modulation atm.
    rate_total = k / n
else:
    n = int(args.block_len * args.code_rate_n / math.log2(M_mod))
    rate = args.code_rate_k / args.code_rate_n  # only the channel coding part without considering modulation atm.
    rate_total = k / n
    M_mod = int( 2**(rate_total / rate) )

batch_size = args.batch_size
dataset_size = 10 ** 8

###############################################
# Channels are defined by the given parameters
###############################################

print("System Parameters")
print(f"Code Length: {n}, Bit Sequence Length: {k}, Rate: {k/n:.2} (bits/symbol)\n")

print("Communication Channel")
print(f"Normalized Doppler freqency: {f_dopp_norm}, Block Length: {block_length}\n")

print("Sensing Channel")
print(f"Number of paths: {P}, Random Delay~U[0, {tau_max * 1e+9:.4}] ns.\n")

cov = utils_standard_codec.Rayleigh_covariance_generator(block_length, f_dopp_norm, dtype_cov=torch.cfloat)

channel_c = partial(ch_corr_Rayleigh_by_cov, cov=cov, device=device, decomposition_method='eigenvector')
#channel_s = partial(Toeplitz_channel, P=P, T=T,
#                         tau_max=tau_max, without_output=True) # N0=2 * noise_std_r ** 2,

if args.snr_points > 1:
    snr_interval = (args.snr_test_end - args.snr_test_start) * 1.0 / (args.snr_points - 1)
else:
    snr_interval = 0
snrs = [snr_interval * item + args.snr_test_start for item in range(args.snr_points)]

BER_list = test_ER_CC_Mod_batched(n, M_mod, channel_c, snrs, k=k, code_rate=1/rate, num_samples=dataset_size, batch_size=batch_size, mod_type=mod_type, GLRT=GLRT, block_length=block_length)

print(snrs)
print(BER_list)

file_tag = f"_k{k:d}N{n:d}_f{f_dopp_norm:.3f}_P{P}_taumax{tau_max}_CC-{M_mod}{mod_type}_{timestamp}"

with open(f'test_results/{file_tag}.txt', 'w') as f:
    f.write('SNR\t')
    f.write(','.join(f"{snr:.2f}" for snr in snrs))
    f.write('\nBER\t')
    f.write(','.join(f"{num:.4e}" for num in BER_list))

