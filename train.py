import sys


# Own libraries
sys.path.append('src')
import src.trainer as trainer
import src.models as models
from src.utils import EbNo_to_noise
from src.utils_plot import *
from src.encoders import import_enc
from src.get_args import get_args


args = get_args()
print(args)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#################################################
# Set the channel parameters and learning model parameters. To be moved to args file later on.
#################################################
f_Dn = 0.1
g = 1
ebn0 = 15

## If you want to load a model trained previously.
LOADMODEL = False
milestone_to_load = None

## Communication and coding parameters
k = 2  # math.log2(M)
M = 2 ** k  # Number of different bit sequences (after source coding)
n = 3 * k  # should be an even number to represent a complex channel
rate = k/n  # bits per symbol rate. average power is n for a symbol.
n_r = 2 * n  # dimension in real datatype. It beacomes 2 times.
f_dopp_norm = float(f_Dn) ## Define the covariance matrix for communication channel
P = 1 # number of paths
T = 1e-9 # Symbol period
tau_max = 2 * T

## Trainining Parameters.
binary = False # If true, encoder takes binary input. If false, encoder takes onehot encoded inputs.
auto_save = True
dataset_size = 2 ** 10  # ~10 ** 6
batch_size = 16
max_epoch = 20  # Set as 0 to turn off training the AE model.
lr_seq = [1e-3, 1e-4, 1e-5]
ep_seq = [20, 20, 20]
only_groundtruth=True

path = 'state_dict/'

## calculate the noise standard deviation
TRAINING_SNR = int(ebn0)
noise_std = EbNo_to_noise(TRAINING_SNR, rate)  # for a normalized signal (per symbol)


print("System Parameters")
print(f"Block Length: {n}, Bit Sequence Length: {k}, Rate: {k/n:.2} (bits/symbol)\n")

print("Communication Channel")
print(f"Normalized Doppler freqency: {f_dopp_norm}\n")

print("Sensing Channel")
print(f"Number of paths: {P}, Random Delay~U[0, {tau_max * 1e+9:.4}] ns.")


#dataset_m = np.random.randint(M, size=dataset_size)
loss_ratio_c_to_r = float(g) #input("Enter the value of gamma. It is the ratio of the communication loss to the sensing loss.") #0  # should be in (0,1)

## File tag for distinguishing the files from this simulation.
file_tag = f"_M{M:d}N{n:d}_{TRAINING_SNR}dB_f{f_dopp_norm:.3f}_gamma{loss_ratio_c_to_r:.1f}"

#################################################
# Setup Channel AE: Encoder, Decoder, Channel
#################################################
# choose encoder and decoder.

encoder = models.Encoder_1h_n2_deeper_ee(M, n_r).to(device)

'''
encoder_ML = import_enc(args)

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
else:
    encoder = encoder_ML(args, p_array1)

encoder = encoder.to(device)
'''



t_conf = trainer.TrainerConfig(M=M, k=k, n=n, n_r=n_r, rate=rate, f_dopp_norm=f_dopp_norm, noise_std_c=noise_std,
                               P=P, T=T, tau_max=tau_max, noise_std_r=noise_std, epoch_loaded = milestone_to_load,
                               max_epochs=ep_seq, dataset_size=dataset_size, is_conv=False,
                               batch_size=batch_size, lr_enc=lr_seq, binary=binary,
                               loss_ratio_c_to_r=loss_ratio_c_to_r, only_groundtruth=only_groundtruth, auto_save=auto_save,
                               file_tag=file_tag, path=path, device=device,
                               args=args)
t_joint = trainer.Trainer(encoder, t_conf, binary=binary)

if LOADMODEL:  ## To load the model if necessary.
    t_joint.load(milestone=milestone_to_load)
    #encoder.load_state_dict(torch.load(path + 'encoder' + file_tag + '.pt'))
    #encoder.eval()
print(encoder)


_, _ = t_joint.train()
