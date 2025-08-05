import sys

sys.path.append('src')
from src.utils import SNR_to_noise, CRLB_mean, MCLB_mean

from src.utils_plot import *
from src.get_args import get_args
from src.utils_standard_codec import modulator, zero_small_parts, differential_encode, zadoff_chu_seq


#########################################

args = get_args()
print(args)

T = args.T                  # Symbol period
P = args.P                  # number of paths in sensing channel
tau_max = [1,2,3,5,7]  # maximum delay of paths. Delay follows a uniform distribution U[0, tau_max].

assert len(tau_max) == P, "The length of tau_max should be P"
print("Sensing Channel")
print(f"Number of paths: {P}, Random Delay~U[0, tau_max*{T:.4e}], tau_max = {tau_max}.\n")


k = args.block_len * args.code_rate_k
mod_type =args.mod_type
N = int(args.block_len * args.code_rate_n)
M_mod = 2

batch_size = args.batch_size
dataset_size = 10 ** 6


device = "cuda" if torch.cuda.is_available() else "cpu"

snr_range = np.linspace(args.snr_test_start, args.snr_test_end, args.snr_points)

if mod_type == "ZC":
    zadoff_chu_seq_0 = np.reshape( zadoff_chu_seq(1,N), (1,N) )

elif mod_type == "OOK":
    OOK_table = np.ones((N,N)) * N ** 0.5


def test_LB(N, M_mod, T, tau_max, N0, num_samples=10*6, batch_size=500, mod_type="PSK"):
    num_batches = num_samples // batch_size
    LBs = []

    for batch_idx in range(num_batches):
        num_message_bits = int(batch_size // N * N) # make it multiple of the block length,

        message_bits = np.random.randint(0, 2, num_message_bits)

        num_modulated_symb = num_message_bits // int(np.log2(M_mod))
        if mod_type == "DPSK":
            truncated_len = (num_modulated_symb // (N - 1)) * (N - 1)
        else:
            truncated_len = (num_modulated_symb // N) * N

        if mod_type not in {"ZC", "OOK"}:
            modulated_symb = modulator(message_bits, M_mod, mod_type)
            modulated_symb = zero_small_parts(modulated_symb)
            modulated_symb = modulated_symb[:truncated_len]
            if mod_type == 'DPSK':
                modulated_symb = differential_encode(modulated_symb, N)
            else:
                modulated_symb = np.reshape(modulated_symb, (-1, N))
        elif mod_type == "ZC":
            modulated_symb = np.repeat( zadoff_chu_seq_0, truncated_len // N, axis=0 )
        elif mod_type == "OOK":
            indices = np.random.choice(OOK_table.shape[0], size=truncated_len // N, replace=True)
            modulated_symb = OOK_table[indices]

        tau_vals = np.random.uniform(size=(modulated_symb.shape[0])) * tau_max
        modulated_symb = torch.from_numpy(modulated_symb).to(device)

        if args.loss_r == "MCLB":
            h_s_a = torch.ones(batch_size, device=device)
            LB = MCLB_mean(N0, h_s_a, modulated_symb, tau_vals, T)
        elif args.loss_r == "CRLB":
            h_s_a = torch.normal(0, (1 / 2) ** 0.5, size=((modulated_symb.shape[0])), device=device) \
                    + 1j * torch.normal(0, (1 / 2) ** 0.5, size=((modulated_symb.shape[0])), device=device)
            LB = CRLB_mean(N0, h_s_a, modulated_symb, tau_vals, T, eps=1e-6)
        else:
            ValueError("loss_r must be either MCLB or CRLB")
        if LB < 10 ** 6:
            LBs.append(LB.cpu().numpy())
    return np.mean(LBs)

LB_list = []; LB_worst_list=[]; LB_best_list=[]
for snr in snr_range:
    noise_std = SNR_to_noise(snr)
    N0 = 2 * noise_std ** 2  # Noise spectral density
    LB_P = []
    for p in range(P):
        LB = test_LB(N, M_mod, T, tau_max[p]*T, N0=2 * noise_std ** 2, num_samples=dataset_size, batch_size=batch_size, mod_type=mod_type)
        LB_P.append(LB)
    LB_P_np = np.array(LB_P)
    LB_worst = np.max(LB_P_np)
    LB_best  = np.min(LB_P_np)
    LB_mean = np.mean(LB_P_np)

    LB_list.append(LB_mean)
    LB_worst_list.append(LB_worst)
    LB_best_list.append(LB_best)

print(snr_range)
print(LB_list)

with open(f'test_results/N{N}_P{P}_taumax{tau_max}_{args.loss_r}_{M_mod}{mod_type}.txt', 'w') as f:
    f.write('SNR\t')
    f.write(','.join(f"{snr:.2f}" for snr in snr_range))
    f.write(f'\n{args.loss_r}\t')
    f.write(','.join(f"{num:.4e}" for num in LB_list))
    f.write(f'\nWorst path {args.loss_r}\t')
    f.write(','.join(f"{num:.4e}" for num in LB_worst_list))
    f.write(f'\nBest path {args.loss_r}\t')
    f.write(','.join(f"{num:.4e}" for num in LB_best_list))

