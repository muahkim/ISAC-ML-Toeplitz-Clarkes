import sys

from numpy import arange
from numpy.random import mtrand

# Own libraries
sys.path.append('src')
import src.trainer as trainer
import src.models as models
from src.utils import save_arguments_to_file
from src.utils_plot import *
from src.encoders import import_enc
from src.decoders import import_dec
from src.get_args import get_args


#########################################
# Turbo AE requires additional arguments and use arguments. All can be turned on by TURBO flag in the source code.
#########################################

args = get_args()
print(args)

## To turn on TurboAE-based source code
TURBO = args.TURBO

## Training algorithm
## Available choices: 'likelihood_training' # 'joint_training' # 'iterative_training',
training_strategy = args.training_strategy
#loss_c = 'likelihood' # 'likelihood' and 'decoder'

#if args.f_Dn == 0:
#    training_strategy = 'likelihood_training'
if training_strategy == 'likelihood_training':
    loss_c = 'likelihood'
elif training_strategy in {'iterative_training', 'joint_training'}:
    loss_c = 'decoder'
else:
    raise ValueError('Invalid training strategy. Valid options: likelihood_training, iterative_training, joint_training')

loss_r = args.loss_r


###############################################
# Set the device to train
###############################################
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#################################################
# Set the channel parameters and learning model parameters. To be moved to args file later on.
#################################################
loss_ratio_c_to_r = args.loss_ratio_c_to_r  # float in [0,1]. loss ratio between communication and sensing. 1 means training only for communication and 0 means training only for sensing.

LOADMODEL               = False     # If you want to load a model trained previously.
milestone_to_load       = 0    # If loading a model, set the milestone to load. It is saved as epoch+1.
milestone_to_load_dec   = 0

if LOADMODEL is False:
    milestone_to_load = 0
    milestone_to_load_dec = 0
if milestone_to_load is None:
    milestone_to_load = 1
if milestone_to_load_dec is None:
    milstone_to_load_dec = milestone_to_load
auto_save = True            # If True, saving the trained models.

# Generating covariance based on Clakre's model
f_Dn = args.f_Dn            # In case Clarke's model used, the normalized Doppler frequency decides the covariance matrix for communication channel.
f_dopp_norm = float(f_Dn)   # Define the covariance matrix for communication channel

T = args.T                  # Symbol period
P = args.P                  # number of paths in sensing channel
tau_max = args.tau_max  # maximum delay of paths. Delay follows a uniform distribution U[0, tau_max].
if P > 1:
    assert len(tau_max) == P, "The length of tau_max should be equal to P."

lr_seq = [1e-3, 1e-4, 1e-5] # list of learning rates for training.
ep_seq = [10, 20, 30]#, 20]#, 10]       # list of number of epochs for each learning rate.
assert len(lr_seq) == len(ep_seq), "the length of lr_seq and ep_seq should be the same."

path = 'state_dict/'        # path to save and load models.

k = args.block_len * args.code_rate_k
M = 2 ** k
n_r = args.block_len * args.code_rate_n * args.enc_num_output_dim # block length in real valued from
n = n_r // 2
rate = args.code_rate_k / args.code_rate_n  # only the channel coding part without considering modulation atm.

batch_size = args.batch_size
dataset_size = batch_size * args.num_block
generate_dataset = False


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

t_conf = trainer.TrainerConfig(M=M, k=k, n=n, n_r=n_r, rate=rate, f_dopp_norm=f_dopp_norm,
                            P=P, T=T, tau_max=tau_max, epoch_loaded = milestone_to_load-1,
                            epoch_loaded_dec = milestone_to_load_dec,
                            max_epochs=ep_seq, dataset_size=dataset_size,
                            batch_size=batch_size, lr_enc=lr_seq, binary=binary,
                            training_strategy=training_strategy, loss_c = loss_c, loss_r = loss_r,
                            loss_ratio_c_to_r=loss_ratio_c_to_r, auto_save=auto_save,
                            path=path, device=device, args=args, generate_dataset=generate_dataset)

t_joint = trainer.Trainer(encoder, t_conf, dec=decoder, TURBO=TURBO)

if LOADMODEL:  ## To load the model if necessary.
    t_joint.load(milestone=milestone_to_load, milestone_dec=milestone_to_load_dec)

print(encoder)
save_arguments_to_file(args, filename=t_joint.config.path_state_dict+"/arguments.txt")
_, _ = t_joint.train()

