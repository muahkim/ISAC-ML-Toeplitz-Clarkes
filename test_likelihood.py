## To test the mutual coherence of the codewords for given encoder.

import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import itertools

from numpy import arange
from numpy.random import mtrand

# Own libraries
sys.path.append('src')
import src.trainer as trainer
import src.models as models
from src.utils import save_arguments_to_file, SNR_to_noise
from src.utils_plot import *
from src.encoders import import_enc
from src.decoders import import_dec
from src.get_args import get_args

ML=False

set_the_font()

#########################################
# Turbo AE requires additional arguments and use arguments. All can be turned on by TURBO flag in the source code.
#########################################

args = get_args()
print(args)

## To turn on TurboAE-based source code
TURBO = args.TURBO

## Training algorithm
training_strategy = args.training_strategy

#if args.f_Dn == 0:
#    training_strategy = 'likelihood_training'
if training_strategy == 'likelihood_training':
    loss_c = 'likelihood'
elif training_strategy in {'iterative_training', 'joint_training'}:
    loss_c = 'decoder'
else:
    raise ValueError('Invalid training strategy. Valid options: likelihood_training, iterative_training, joint_training')


###############################################
# Set the device to train
###############################################
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#################################################
# Set the channel parameters and learning model parameters. To be moved to args file later on.
#################################################

         # If you want to load a model trained previously.
milestone_to_load       = args.load_epoch    # If loading a model, set the milestone to load. It is saved as epoch+1.
milestone_to_load_dec   = args.load_epoch
training_snr        = args.snr_points # To load models trained with training_snr
loss_ratio_c_to_r   = args.loss_ratio_c_to_r  # float in [0,1]. loss ratio between communication and sensing. 1 means training only for communication and 0 means training only for sensing.
if milestone_to_load > 0:
    LOADMODEL = True
else:
    LOADMODEL = False

auto_save = True            # If True, saving the trained models.

# Generating covariance based on Clakre's model
f_Dn = args.f_Dn            # In case Clarke's model used, the normalized Doppler frequency decides the covariance matrix for communication channel.
f_dopp_norm = float(f_Dn)   # Define the covariance matrix for communication channel

T = args.T                  # Symbol period
P = args.P                  # number of paths in sensing channel
tau_max = args.tau_max  # maximum delay of paths. Delay follows a uniform distribution U[0, tau_max].
if P > 1 and args.loss_ratio_c_to_r < 1:
    assert len(tau_max) == P, "The length of tau_max should be equal to P."

path = 'state_dict/'        # path to save and load models.

k = args.block_len * args.code_rate_k
M = 2 ** k
n_r = args.block_len * args.code_rate_n #+ 2
n = n_r // 2
rate = args.code_rate_k / args.code_rate_n  # only the channel coding part without considering modulation atm.

batch_size = args.batch_size
dataset_size = batch_size * 100
generate_dataset = False

noise_std = SNR_to_noise(args.snr_test_start)

def compute_mutual_coherence(vec1, vec2):
    return np.abs(np.vdot(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def compute_mutual_exponent_table(x0, table, noise_std):
    # expected value averaged over randomness.
    y = np.expand_dims(x0, axix=0).repeat(table.shape[0], axis=0)
    summand = (np.abs(table) ** 2 + 2 * noise_std ** 2)/ (np.abs(x0) ** 2 + 2 * noise_std ** 2)
    exponent = - np.sum(summand, axis=1)
    return exponent

def compute_mutual_exponent(x0, x, noise_std):
    # expected value averaged over randomness.
    summand = (np.abs(x) ** 2 + 2 * noise_std ** 2)/ (np.abs(x0) ** 2 + 2 * noise_std ** 2)
    exponent = - np.sum(summand)
    return exponent

def compute_log_coefficient(x, noise_std):
    N = x.shape[0]
    coeff = math.pi ** N * np.prod(np.abs(x)**2 + 2*noise_std**2)
    return -np.log(coeff)

def binary_sequences_torch(k, device):
    """Generate all 2^k binary sequences of length k as torch tensor on device"""
    binary = list(itertools.product([0, 1], repeat=k))  # list of tuples
    tensor = torch.tensor(binary, dtype=torch.float32, device=device)
    return tensor  # shape: (2^k, k)


def binary_sequences_numpy(k):
    """
    Generate all 2^k binary sequences of length k as a NumPy array
    Equivalent to binary_sequences_torch() but using NumPy
    """
    binary = list(itertools.product([0, 1], repeat=k))  # list of tuples
    array = np.array(binary, dtype=np.float32)  # shape: (2^k, k)
    return array


def encoder_module_cc(k, n, mod_type, M_mod, seed=42, batch_size=48):
    from src.utils_standard_codec import cc_encoder, modulator, zero_small_parts, differential_encode
    import random
    np.random.seed(seed)
    random.seed(seed)

    message_bits = binary_sequences_numpy(k)  # shape: (2^k, k)
    message_bits = np.reshape(message_bits, (-1, batch_size))
    num_it = message_bits.shape[0]
    encoded_list = []
    for i in range(num_it):
        encoded_bits, trellis = cc_encoder(message_bits[i,:].copy(), code_rate=args.code_rate_n / args.code_rate_k)
        encoded_list.append(encoded_bits[:batch_size*args.code_rate_n // args.code_rate_k])
    encoded_bits = np.reshape(np.array(encoded_list), (-1))

    modulated_symb = modulator(encoded_bits, M_mod, mod_type)
    modulated_symb = zero_small_parts(modulated_symb)

    if mod_type == 'DPSK':
        truncated_len = (modulated_symb.shape[0] // (n - 1)) * (n - 1)
    else:
        truncated_len = (modulated_symb.shape[0] // n) * n
    modulated_symb = modulated_symb[:truncated_len]

    if mod_type == 'DPSK':
        modulated_symb = differential_encode(modulated_symb, n)
    else:
        modulated_symb = np.reshape(modulated_symb, (-1, n))

    return modulated_symb


def encoder_module_ml(k, encoder, device='cuda', seed=42):
    import random

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    M = 2 ** k

    if TURBO:  # binary data
        inputs = binary_sequences_torch(k, device=device)  # shape: (2^k, k)
        inputs = inputs.reshape(M, k, args.code_rate_k).float()

    else:  # one hot encoded data
        inputs = np.arange(0, M)
        inputs = torch.from_numpy(inputs)

    # 2. 인코딩
    encoder.to(device)
    encoder.eval()
    with (torch.no_grad()):
        encoded = encoder(inputs)  # shape: (M, N), torch complex
        encoded = torch.view_as_complex(encoded.detach().cpu().view(M, -1, 2))
        encoded_np = encoded.numpy()
    encoded_np = np.array(encoded_np, dtype=np.complex128)

    return encoded_np


def analyze_likelihood(encoded_np, average_over_all=False, seed=42, noise_std=10.1):
    """
    encoded_np: np array of encoded codewords
    device: 'cuda' or 'cpu'
    average_over_all: if True, computes mean mutual likelihood over all reference vectors
    """
    import random

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    M = encoded_np.shape[0]

    # 3. Likelihood 계산
    if average_over_all:
        sorted_likelihoods = []
        self_likelihoods = []


        for ref_idx in range( M ):
            v_ref = encoded_np[ref_idx]
            exponents = [compute_mutual_exponent(v_ref, v, noise_std=noise_std) for v in encoded_np]

            self_likelihoods.append(exponents[ref_idx])

            # 자기 자신 제외하고 mutual likelihood만 정렬
            mutuals = [exponents[i] for i in range(M) if i != ref_idx]
            mutuals_sorted = sorted(mutuals)
            sorted_likelihoods.append(mutuals_sorted)

        # mutual likelihood는 등수별 평균
        sorted_mutuals = np.array(sorted_likelihoods)  # shape: (M, M-1)
        avg_mutuals = np.mean(sorted_mutuals, axis=0)

        # self likelihood 평균
        avg_self = np.mean(self_likelihoods)

        # 전체 시각화용 리스트: 앞에 self, 뒤에 mutual 등수별 평균
        all_exponents = list(avg_mutuals) + [avg_self]
        colors = ['blue'] * (M - 1) + ['red']
        title = "Averaged"

    else:
        idx_ref = random.randint(0, M - 1)
        v_ref = encoded_np[idx_ref]
        coherence_values = [(i, compute_mutual_exponent(v_ref, v, noise_std=noise_std)) for i, v in enumerate(encoded_np)]
        coherence_values.sort(key=lambda x: x[1])
        indices, values = zip(*coherence_values)
        all_exponents = values
        colors = ['red' if i == idx_ref else 'blue' for i in indices]
        title = f"Ref index = {idx_ref + 1}"

    # 4. 시각화

    plt.figure(figsize=(6, 3))
    plt.bar(range(len(all_exponents)), all_exponents, color=colors)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Other vectors'),
        Patch(facecolor='red', label='Self')
    ]
    plt.legend(handles=legend_elements)
    plt.title(title)
    plt.xlabel("Sorted Index")
    plt.ylabel("Mutual Exponents")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if average_over_all:
        return all_exponents  # list of [avg_self, avg_rank1, avg_rank2, ..., avg_rank(M-1)]
    else:
        return idx_ref, list(zip(indices, all_exponents))

import numpy as np
import matplotlib.pyplot as plt

def compute_papr(codeword):
    """
    Compute PAPR for a single complex-valued codeword
    """
    power = np.abs(codeword) ** 2
    return np.max(power) / np.mean(power)

def plot_sorted_papr(codewords):
    """
    Compute and plot sorted PAPR values for a set of codewords

    Parameters:
        codewords (np.ndarray): shape (M, N), complex-valued codewords
    """
    papr_list = [compute_papr(cw) for cw in codewords]
    papr_sorted = np.sort(papr_list)

    # Plot
    plt.figure(figsize=(5, 3))
    plt.plot(papr_sorted, linestyle='-')
    plt.title("Turbo-CNN")
    plt.xlabel("Codeword Index (sorted)")
    plt.ylabel("PAPR")
    plt.grid(True)
    plt.ylim(0, 10)

    plt.tight_layout()
    plt.show()

    return papr_list, papr_sorted

###############################################
# Channels are defined by the given parameters
###############################################

print("System Parameters")
print(f"Block Length: {n}, Bit Sequence Length: {k}, Rate: {k/n:.2} (bits/symbol)\n")

print("Communication Channel")
print(f"Normalized Doppler freqency: {f_dopp_norm}\n")

print("Sensing Channel")
print(f"Number of paths: {P}, Random Delay~U[0, tau_max*{T:.4e}], tau_max = {tau_max}.\n")

print(f"Training Strategy: {training_strategy}\n")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#encoder = DummyEncoder(k=5)

if ML:
    if TURBO:
        binary = True  # If true, encoder takes binary input. If false, encoder takes onehot encoded inputs.

        #################################################
        # Setup Channel AE: Encoder, Decoder, Channel
        #################################################
        # choose encoder and decoder.
        encoder_ML = import_enc(args)
        decoder_ML = import_dec(args)

        # setup interleaver.
        if args.is_interleave == 1:  # fixed interleaver.
            seed = np.random.randint(0, 1)
            rand_gen = mtrand.RandomState(seed)
            p_array1 = rand_gen.permutation(arange(args.block_len))
            p_array2 = rand_gen.permutation(arange(args.block_len))

        elif args.is_interleave == 0:
            p_array1 = range(args.block_len)  # no interleaver.
            p_array2 = range(args.block_len)  # no interleaver.
        else:
            seed = np.random.randint(0, args.is_interleave)
            rand_gen = mtrand.RandomState(seed)
            p_array1 = rand_gen.permutation(arange(args.block_len))
            seed = np.random.randint(0, args.is_interleave)
            rand_gen = mtrand.RandomState(seed)
            p_array2 = rand_gen.permutation(arange(args.block_len))

        print('using random interleaver', p_array1, p_array2)

        if args.encoder == 'turboae_2int' and args.decoder == 'turboae_2int':
            encoder = encoder_ML(args, p_array1, p_array2)
            decoder = decoder_ML(args, p_array1, p_array2)
        else:
            encoder = encoder_ML(args, p_array1)
            decoder = decoder_ML(args, p_array1)

        encoder = encoder.to(device)
        decoder = decoder.to(device)

    else:
        binary = False
        encoder = models.Encoder_1h_n2_deeper_ee(M, n_r).to(device)
        decoder = models.Decoder_1h(k, M, n_r).to(device)
        args.encoder = 'mlp'
        args.decoder = 'mlp'

    t_conf = trainer.TrainerConfig(M=M, k=k, n=n, n_r=n_r, rate=rate, f_dopp_norm=f_dopp_norm,  # ebn0=ebn0,
                                   snr=training_snr,
                                   P=P, T=T, tau_max=tau_max, epoch_loaded=milestone_to_load - 1,
                                   epoch_loaded_dec=milestone_to_load_dec,
                                   dataset_size=dataset_size,
                                   batch_size=batch_size, binary=binary,
                                   training_strategy=training_strategy, loss_c=loss_c,
                                   loss_ratio_c_to_r=loss_ratio_c_to_r, auto_save=auto_save,
                                   path=path, device=device, args=args, generate_dataset=generate_dataset)

    t_joint = trainer.Trainer(encoder, t_conf, dec=decoder, TURBO=TURBO)

    if LOADMODEL:  ## To load the model if necessary.
        if args.loss_ratio_c_to_r > 0:
            t_joint.load(milestone=milestone_to_load, milestone_dec=milestone_to_load_dec)
        else:
            t_joint.load(milestone=milestone_to_load, encoder_only=True)

    print(encoder)
    save_arguments_to_file(args, filename=t_joint.config.path_state_dict + "/arguments.txt")

    encoded_np_array = encoder_module_ml(k, encoder, device=device)
else:
    encoded_np_array =  encoder_module_cc(k, args.block_len * args.code_rate_n, args.mod_type, 2, seed=42)

# 하나의 기준 벡터로 mutual coherence 플롯
#analyze_likelihood(encoded_np_array, average_over_all=False, noise_std=noise_std)

# 평균 mutual coherence 플롯
#analyze_likelihood(encoded_np_array, average_over_all=True, noise_std=noise_std)
# PAPR plot

_, papr_sorted = plot_sorted_papr(encoded_np_array)
if ML:
    np.save(f"test_results/papr_sorted_TurboAE_rate3_cnn_K{k}N{n}.npy", papr_sorted)
else:
    np.save(f"test_results/papr_sorted_CC_{args.mod_type}_K{k}N{n_r}.npy", papr_sorted)
'''
coeffs = [compute_log_coefficient(v, noise_std) for v in encoded_np_array]
coeffs_sorted = sorted(coeffs)

plt.figure(figsize=(6, 3))
plt.bar(range(len(coeffs_sorted)), coeffs_sorted)

plt.xlabel("Sorted Index")
plt.ylabel("Log Coefficient")
plt.grid(True)
plt.tight_layout()
plt.show()


with open(f'test_results/{t_conf.file_tag}_{t_conf.timestamp}.txt', 'w') as f:
    f.write('SNR\t')
    f.write(','.join(f"{snr:.2f}" for snr in snr_range))
    f.write('\nBER\t')
    f.write(','.join(f"{num:.4e}" for num in BER_list))
'''

