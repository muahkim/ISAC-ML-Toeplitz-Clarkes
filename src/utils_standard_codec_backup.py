import numpy as np
import sys
import scipy

assert sys.version_info >= (3, 5)
np.random.seed(42)

sys.path.append('src/')
from utils import *
import utils
from channel_models import *

import commpy.channelcoding.convcode as cc
import commpy.modulation as modulation

def cc_encoder(input_bits, code_rate=3, termination='term'):
    # Memory elements per input bit (K-1)
    if code_rate == 1.5:
        memory = np.array([2, 2])
        # Define Rate 2/3 Convolutional Code with Constraint Length K=3
        g_matrix = np.array([[0o7, 0o5, 0o0],  # input 1 contributes to output 1 and 2
                             [0o0, 0o6, 0o3]])  # input 2 contributes to output 2 and 3
    elif code_rate == 3:
        memory = np.array([2])
        # Define Rate 1/3 Convolutional Code with Constraint Length K=3
        g_matrix = np.array([[7, 5, 6]])  # Generator polynomials in octal
    elif code_rate == 4:
        memory = np.array([2])
        # Define Rate 1/4 Convolutional Code with Constraint Length K=3
        g_matrix = np.array([[7, 5, 6, 3]])  # Generator polynomials in octal
    else:
        raise ValueError

    # Create Trellis structure
    trellis = cc.Trellis(memory, g_matrix)

    # Encode using Convolutional Encoder
    encoded_bits = cc.conv_encode(input_bits, trellis, termination=termination)

    rate = g_matrix.shape[0] / g_matrix.shape[1]

    return encoded_bits, trellis

def random_interleaver(data, seed=None):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data))
    interleaved = data[perm]
    return interleaved, perm

def random_deinterleaver(data, perm):
    deinterleaved = np.empty_like(data)
    deinterleaved[perm] = data  # Reverse the permutation
    return deinterleaved

def fixed_permutation_interleave(bits, perm):
    return bits[perm]

def fixed_permutation_deinterleave(bits, perm):
    inv = np.argsort(perm)
    return bits[inv]

def differential_encode(modulated_symbols, n, prev_symbol = 1 + 0j):
    """Differentially encodes a sequence of already PSK-modulated symbols."""
    # modulated_symbols: (-1,) np complex array
    # prev_symbol: Initial reference symbol (can be 1+0j or any reference point)

    modulated_symbols = modulated_symbols.reshape(-1, n-1) # chunk into blocks
    encoded_symbols = np.zeros((modulated_symbols.shape[0], n), dtype=complex)
    encoded_symbols[:, 0] = prev_symbol

    for idx in range(modulated_symbols.shape[0]):
        block = modulated_symbols[idx, :]
        prev_symbol_block = prev_symbol
        for i, symbol in enumerate(block):
            encoded_symbols[idx, i+1] = prev_symbol_block * symbol  # Multiply with previous symbol
            prev_symbol_block = encoded_symbols[idx, i+1]  # Update for next iteration

    return encoded_symbols


def hard_viterbi_tail_biting(encoded_bits, trellis, traceback_depth=15):
    """
    Tail-biting Viterbi decoder for hard-decision input with multi-bit input symbols.

    Parameters:
        encoded_bits (np.ndarray): Encoded bits (1D array)
        trellis (commpy.Trellis): Trellis object
        traceback_depth (int): Traceback depth (currently unused)

    Returns:
        decoded_bits (np.ndarray): Reconstructed input bit stream
    """
    n_states = trellis.number_states
    k = trellis.k  # number of input bits per input symbol
    n = trellis.n  # number of output bits per input symbol
    n_input_symbols = 2 ** k

    num_symbols = len(encoded_bits) // n
    encoded_reshaped = encoded_bits.reshape((-1, n))

    best_path_metric = np.inf
    best_decoded_input_symbols = None

    for start_state in range(n_states):
        path_metrics = np.full((num_symbols + 1, n_states), np.inf)
        path_metrics[0, start_state] = 0

        prev_states = np.zeros((num_symbols, n_states), dtype=int)
        prev_inputs = np.zeros((num_symbols, n_states), dtype=int)

        for t in range(num_symbols):
            for curr_state in range(n_states):
                for input_sym in range(n_input_symbols):
                    for prev_state in range(n_states):
                        if trellis.next_state_table[prev_state, input_sym] == curr_state:
                            expected_output = trellis.output_table[prev_state, input_sym]
                            expected_bits = np.array(
                                list(np.binary_repr(expected_output, width=n)), dtype=int
                            )
                            metric = np.sum(expected_bits != encoded_reshaped[t])
                            new_metric = path_metrics[t, prev_state] + metric

                            if new_metric < path_metrics[t + 1, curr_state]:
                                path_metrics[t + 1, curr_state] = new_metric
                                prev_states[t, curr_state] = prev_state
                                prev_inputs[t, curr_state] = input_sym

        # Tail-biting 조건: 마지막 상태 == 시작 상태
        if path_metrics[-1, start_state] < best_path_metric:
            best_path_metric = path_metrics[-1, start_state]
            input_symbols = np.zeros(num_symbols, dtype=int)
            state = start_state

            for t in reversed(range(num_symbols)):
                input_symbols[t] = prev_inputs[t, state]
                state = prev_states[t, state]

            best_decoded_input_symbols = input_symbols

    if best_decoded_input_symbols is None:
        raise RuntimeError("No valid tail-biting path found.")

    # 각 input symbol을 k비트로 풀어내기
    decoded_bits = np.zeros(num_symbols * k, dtype=int)
    for i, symbol in enumerate(best_decoded_input_symbols):
        bits = np.array(list(np.binary_repr(symbol, width=k)), dtype=int)
        decoded_bits[i * k : (i + 1) * k] = bits

    return decoded_bits


def differential_decode_mpsk(received_symb, M, n, modulation_table):
    """Differentially decodes a sequence of M-PSK demodulated symbols."""
    received_symb = np.reshape(received_symb, (-1, n))
    decoded_symbols = np.zeros((received_symb.shape[0], n-1), dtype=int)

    for idx in range(received_symb.shape[0]):
        for i in range(1, n):
            # Compute phase difference
            vector_diff = received_symb[idx, i] * np.conj(received_symb[idx, i-1])
            vector_diff_norm = vector_diff / np.linalg.norm(vector_diff)

            # Map phase difference to closest M-PSK symbol
            phase_diff_diff = np.abs(np.angle(modulation_table * np.conj(vector_diff_norm)))
            symbol_dec = np.argmin(phase_diff_diff)
            decoded_symbols[idx, i-1] = symbol_dec

    decoded_symbols = np.reshape(decoded_symbols, (-1,))
    decoded_symbols =  utils.dec2bin_digits(decoded_symbols, bits=int(math.log2(M)))
    return decoded_symbols


def zero_small_parts(x: np.ndarray, threshold: float = 1e-10) -> np.ndarray:
    # Suppress arbitrarly small values to zero.
    real = np.real(x).copy()
    imag = np.imag(x).copy()

    # Set small values to zero
    real[np.abs(real) < threshold] = 0
    imag[np.abs(imag) < threshold] = 0

    return real + 1j * imag


def compute_gaussian_thresholds(variances):
    """
    Compute intersection thresholds between adjacent zero-mean Gaussians.
    This is used for ML detection when ASK is used.

    Parameters:
    variances: list or array-like of floats
        List of variances in ascending order.

    Returns:
    thresholds: list of floats
        List of positive thresholds where PDFs intersect.
        Each threshold lies between variances[i] and variances[i+1].
    """
    variance = np.array(variances)
    if not np.all(np.diff(variance) > 0):
        raise ValueError("Variances must be in strictly ascending order.")

    thresholds = []
    for i in range(len(variances) - 1):
        var1 = variances[i]
        var2 = variances[i + 1]
        numerator = np.log(var2 / var1)
        denominator = (1 / var1) - (1 / var2)
        x_squared = numerator / denominator
        if x_squared < 0:
            raise ValueError(f"Negative x^2 encountered between σ^2={variances[i]} and σ^2={variances[i + 1]}")
        x = np.sqrt(x_squared)
        thresholds.append(x)  # Only need one side, since distributions are symmetric

    return thresholds


class ASKModem:
    def __init__(self, M, noise_std=None):
        assert M >= 2 and ((M & (M - 1)) == 0), "M must be a power of 2"
        self.M = M
        self.k = int(np.log2(M))  # bits per symbol

        # ASK constellation: [0, 1, ..., M-1]
        self.constellation = np.arange(M, dtype=np.float64)
        # Normalize average energy to 1
        self.constellation /= np.sqrt(np.mean(self.constellation**2))
        self.A = self.constellation[1]
        if noise_std is not None:
            variances = [(m**2 * self.A **2 + noise_std**2) for m in range(M)] # noise variance of in ascending order
            self.thresholds = compute_gaussian_thresholds(variances)

    def modulate(self, bits):
        symbols = []
        for i in range(0, len(bits), self.k):
            b = bits[i:i+self.k]
            idx = int("".join(str(x) for x in b), 2)
            symbols.append(self.constellation[idx])
        return np.array(symbols)

    def demodulate(self, rx_signal, thresholds):
        """
        Demodulate a complex-valued received signal using real part and given thresholds.
        It is ML detector when the channel is Rayleigh fading channel with AWGN.

        Parameters:
        rx_signal: array-like of complex numbers
            The received noisy symbols from the channel.
        thresholds: list of floats
            Decision thresholds (positive values), assumed symmetric about 0.
            Should be in ascending order.

        Returns:
        symbols: np.ndarray of ints
            Demodulated symbol indices (e.g., 0, 1, 2, ..., M-1)
        """
        thresholds = np.array(self.thresholds)
        abs_part = np.abs(rx_signal)

        # Use np.digitize to assign region indexes based on thresholds
        # Bins go from -0 to first threshold, first to second, ..., last to +inf
        symbol_indices = np.digitize(abs_part, thresholds)

        return symbol_indices

    def demodulate_NN(self, symbols, threshold=None, mode='hard'):
        # Minimum distance decoding / Nearest Neighbor Rule
        indices = np.argmin(np.abs(symbols[:, None] - self.constellation[None, :])**2, axis=1)
        bits = [list(np.binary_repr(i, width=self.k)) for i in indices]
        return np.array(bits, dtype=int).flatten()


def modulator(bit_seq, M_mod, mod_type):
    ## Define modulator
    if mod_type in {'PSK', 'DPSK'}:
        modem = modulation.PSKModem(M_mod)  # M-PSK modem initialization
    elif mod_type == 'QAM':
        modem = modulation.QAMModem(M_mod)
    elif mod_type == 'ASK':
        modem = ASKModem(M_mod)
    else:
        assert False

    modulated_symb = modem.modulate(bit_seq)

    return modulated_symb

def demodulator(received_seq, M_mod, mod_type, noise_std=None):
    ## Define demodulator
    if mod_type in {'PSK', 'DPSK'} :
        modem = modulation.PSKModem(M_mod)  # M-PSK modem initialization
    elif mod_type == 'QAM':
        modem = modulation.QAMModem(M_mod)
    elif mod_type == 'ASK':
        modem = ASKModem(M_mod, noise_std)
    else:
        assert False

    return modem.demodulate(received_seq, 'hard')

def test_ER_CC_Mod_batched(n, M_mod, channel_model, snr_range, k, code_rate=3, num_samples=10 ** 8, batch_size=10 ** 4, mod_type='PSK', uncoded=False):
    num_batches = num_samples // batch_size
    BER_per_SNR = []

    if mod_type == 'DPSK':
        all_bit_seq = dec2bin_digits(np.arange(M_mod), int(math.log2(M_mod)))
        all_bit_seq = all_bit_seq.reshape(-1, 1)
        modulation_table = modulator(all_bit_seq, M_mod, mod_type)
        modulation_table = zero_small_parts(modulation_table)
    else:
        modulation_table = None
    for snr in tqdm(snr_range, desc="testing"):

        noise_std = utils.SNR_to_noise(snr)
        BER_list = []
        #pbar = tqdm(range(num_batches), desc="Batches", miniters=num_batches//5)
        for it in range(num_batches):
            if uncoded:
                BER_batch = test_ER_Mod(n, M_mod, channel_model, noise_std, k, num_samples=batch_size, mod_type=mod_type, modulation_table=modulation_table)
            else:
                BER_batch = test_ER_CC_Mod(n, M_mod, channel_model, noise_std, k, code_rate=code_rate,
                                       num_samples=batch_size, mod_type=mod_type, modulation_table=modulation_table)

            BER_list.append(BER_batch)
            BER_array = np.array(BER_list)

            if (it+1) % (num_batches//100) == 0:
                print(f"Iteration {it} BER {np.mean(BER_array, axis=0):.4e}")
            if (it+1) * batch_size * np.mean(BER_array, axis=0) > 10000 :
                break

        BER_array = np.array(BER_list)
        BER_per_SNR.append(np.mean(BER_array, axis=0))

        print(f"\nSNR: {snr}, BER: {BER_per_SNR[-1]:.3e}")


    return BER_per_SNR

def test_ER_CC_Mod(n, M_mod, channel_model, noise_std, k, code_rate=3, num_samples=10 ** 6, mod_type='PSK', modulation_table=None, interleaving=False, tail_bite=False, datashape='chunk', termination='term'):
    # Generate a random input sequence
    np.random.seed(43)
    rng = np.random.default_rng(43)

    k_threshold = 24
    if k < k_threshold:
        message_length = k_threshold
        code_length = int(k_threshold * code_rate)
    else:
        message_length = k
        code_length = n

    if interleaving:
        perm = rng.permutation(message_length)

    num_message_bits = int(num_samples // message_length * message_length) # make it multiple of the block length,

    message_bits = np.random.randint(0, 2, num_message_bits)

    # Create a trellis diagram for encoding and decoding
    _, trellis = cc_encoder(message_bits.copy(), code_rate=code_rate)

    if datashape=='stream':
        if interleaving:
            message_bits = fixed_permutation_interleave(message_bits.copy(), perm)
        encoded_bits, trellis = cc_encoder(message_bits.copy(), code_rate=code_rate, termination=termination)
    else:
        # encode chunk-wise
        encoded_bits_all = []

        num_chunks = len(message_bits) // message_length

        for i in range(num_chunks):
            start = i * message_length
            end = start + message_length
            chunk = message_bits[start:end]
            if interleaving:
                assert chunk.shape[0] == message_length
                chunk = fixed_permutation_interleave(chunk.copy(), perm)

            encoded_chunk, _ = cc_encoder(chunk.copy(), code_rate=code_rate, termination=termination)
            #encoded_chunk = encoded_chunk[:code_length]
            encoded_bits_all.append(encoded_chunk)

        encoded_bits = np.concatenate(encoded_bits_all)

    code_length = encoded_chunk.shape[0]

    # pad zeros to match the shape to be multiple of the block lengths
    if mod_type == 'DPSK':
        encoded_len = encoded_bits.shape[0]
        padded_len = (encoded_len // n + 1) * n
        pad_length = padded_len - encoded_len
    else:
        encoded_len = encoded_bits.shape[0]
        padded_len = (encoded_len // n + 1) * n
        pad_length = padded_len - encoded_len
    encoded_bits = np.concatenate([encoded_bits, np.zeros((pad_length,), dtype=int)])

    # Modulate
    modulated_symb = modulator(encoded_bits, M_mod, mod_type)
    modulated_symb = zero_small_parts(modulated_symb)

    if mod_type == 'DPSK':
        modulated_symb = differential_encode(modulated_symb, n + 1)
    else:
        modulated_symb = np.reshape(modulated_symb, (-1, n))

    modulated_symb_torch = torch.from_numpy(modulated_symb)

    # Channel
    received_symb, h = channel_model(modulated_symb_torch, noise_std=noise_std)
    received_symb_scale_back = received_symb.cpu().numpy() #/ sqrt(n) * np.reshape(sigma, (-1,1))
    received_symb_scale_back = np.reshape(received_symb_scale_back, (-1,))

    if mod_type == 'DPSK':
        demodulated_bits = differential_decode_mpsk(received_symb_scale_back, M_mod, n + 1, modulation_table)
    else:
        demodulated_bits = demodulator(received_symb_scale_back, M_mod, mod_type, noise_std=noise_std)
    demodulated_bits = np.reshape(demodulated_bits, (-1,))
    demodulated_bits = demodulated_bits[:encoded_len] # Cut out the zero paddings.


    if datashape == 'stream':
        #demod_padded = np.concatenate([demodulated_bits.copy(), np.zeros(trellis.total_memory, dtype=int)])
        decoded_bits = cc.viterbi_decode(demodulated_bits.copy(), trellis, tb_depth=15)
        decoded_bits = decoded_bits[:message_length]
        if interleaving:
            decoded_bits = fixed_permutation_deinterleave(decoded_bits.copy(), perm)
    else:
        decoded_bits_all = []

        #Decoding per each chunk
        num_chunks = len(demodulated_bits) // code_length

        for i in range(num_chunks):
            start = i * code_length
            end = start + code_length
            chunk = demodulated_bits[start:end]

            # Viterbi decoding
            if tail_bite:
                decoded_chunk = hard_viterbi_tail_biting(chunk.copy(), trellis)
            else:
                # zero-padding for decoding
                #demod_padded = np.concatenate([chunk.copy(), np.zeros(trellis.total_memory, dtype=int)])
                decoded_chunk = cc.viterbi_decode(chunk.copy(), trellis, tb_depth=15)
                decoded_chunk = decoded_chunk[:message_length]
                if interleaving:
                    assert decoded_chunk.shape[0] == message_length
                    decoded_chunk = fixed_permutation_deinterleave(decoded_chunk.copy(), perm)
            decoded_bits_all.append(decoded_chunk)

        # Combining all the chunks
        decoded_bits = np.concatenate(decoded_bits_all)

    decoded_bits = np.reshape( decoded_bits, (-1,))
    message_bits = np.reshape( message_bits, (-1,))
    demodulated_bits = np.reshape(demodulated_bits, (-1,))

    try:
        ne = np.not_equal(decoded_bits[:], message_bits[:decoded_bits.shape[0]])
    except:
        ne = np.not_equal(decoded_bits[:message_bits.shape[0]], message_bits[:])
    ne1 = np.not_equal(demodulated_bits, encoded_bits[:demodulated_bits.shape[0]])

    BER_tmp = np.mean(ne.astype(float))

    '''
    ### To test the CC
    decoded_bits_CC = cc.viterbi_decode(encoded_bits.copy(), trellis)
    ne = np.not_equal(decoded_bits_CC[:message_bits.shape[0]], message_bits)
    BER_only_CC = np.mean(ne.astype(float))
    print(f"BER only encoding and decoding of CC: {BER_only_CC:.3e}")
    
    
    decoded_bits_all_CC = []

    # 각 청크에 대해 디코딩 수행
    num_chunks = len(encoded_bits) // code_length

    for i in range(num_chunks):
        start = i * code_length
        end = start + code_length
        chunk = encoded_bits[start:end]

        # 디코딩을 위한 zero-padding
        demod_padded = np.concatenate([chunk.copy(), np.zeros(trellis.total_memory, dtype=int)])

        # Viterbi 디코딩
        decoded_chunk = cc.viterbi_decode(demod_padded.copy(), trellis, tb_depth=15)
        if interleaving:
            decoded_chunk = fixed_permutation_deinterleave(decoded_chunk.copy(), perm)
        decoded_bits_all_CC.append(decoded_chunk[:message_length])

    # 디코딩된 청크들을 하나로 합칩니다
    decoded_bits_CC = np.concatenate(decoded_bits_all_CC)
    ne = np.not_equal(decoded_bits_CC, message_bits[:decoded_bits.shape[0]])
    BER_only_CC = np.mean(ne.astype(float))
    print(f"BER only encoding and decoding of CC: {BER_only_CC:.3e}")
    

    
    ###
    
    ### To test modulation
    modulated_symb_torch_test = modulated_symb_torch.view(-1,)
    if mod_type == 'DPSK':
        demodulated_bits_test = differential_decode_mpsk(modulated_symb_torch_test.cpu().numpy(), M_mod, n+1, modulation_table)
    else:
        demodulated_bits_test = demodulator(modulated_symb_torch_test.cpu().numpy(), M_mod, mod_type, noise_std=noise_std)
    demodulated_bits_test = np.reshape(demodulated_bits_test, (-1,))
    ne_mod = np.not_equal(demodulated_bits_test, encoded_bits[:demodulated_bits_test.shape[0]])
    BER_mod = np.mean(ne_mod.astype(float))
    print(f"BER from modulation without channel: {BER_mod:.3e}." )
    '''

    return BER_tmp


def test_ER_Mod(n, M_mod, channel_model, noise_std, k, num_samples=10 ** 6, mod_type='PSK',
                   modulation_table=None):
    # Generate a random input sequence
    np.random.seed(42)

    message_bits = np.random.randint(0, 2, num_samples)

    modulated_symb = modulator(message_bits, M_mod, mod_type)
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

    modulated_symb_torch = torch.from_numpy(modulated_symb)

    received_symb, h = channel_model(modulated_symb_torch, noise_std=noise_std)
    received_symb_scale_back = received_symb.cpu().numpy()  # / sqrt(n) * np.reshape(sigma, (-1,1))
    received_symb_scale_back = np.reshape(received_symb_scale_back, (-1,))

    if mod_type == 'DPSK':
        demodulated_bits = differential_decode_mpsk(received_symb_scale_back, M_mod, n, modulation_table)
    else:
        demodulated_bits = demodulator(received_symb_scale_back, M_mod, mod_type, noise_std=noise_std)
    demodulated_bits = np.reshape(demodulated_bits, (-1,))

    message_bits = np.reshape(message_bits, (-1,))
    demodulated_bits = np.reshape(demodulated_bits, (-1,))

    ne = np.not_equal(demodulated_bits, message_bits[:demodulated_bits.shape[0]])

    BER_tmp = np.mean(ne.astype(float))

    '''
    ### To test modulation
    modulated_symb_torch_test = modulated_symb_torch.view(-1,)
    if mod_type == 'DPSK':
        demodulated_bits_test = differential_decode_mpsk(modulated_symb_torch_test.cpu().numpy(), M_mod, n, modulation_table)
    else:
        demodulated_bits_test = demodulator(modulated_symb_torch_test.cpu().numpy(), M_mod, mod_type, noise_std=noise_std)
    demodulated_bits_test = np.reshape(demodulated_bits_test, (-1,))
    ne_mod = np.not_equal(demodulated_bits_test, encoded_bits[:demodulated_bits_test.shape[0]])
    BER_mod = np.mean(ne_mod.astype(float))
    print(f"BER from modulation without channel: {BER_mod:.3e}." )
    '''

    return BER_tmp


def Rayleigh_covariance_generator(N, f_dopp_norm, dtype_cov=torch.cfloat):
    # parameter scaling
    # the zeroth Bessel function
    # put the function output in the N x N matrix correctly.
    assert f_dopp_norm <= 1 and f_dopp_norm >= 0, "The normalized doppler frequency out of range. It should be in (0,1)."
    arg = 2 * math.pi * f_dopp_norm

    cov = torch.empty(N, N, dtype=dtype_cov)
    for i in range(N):
        for j in range(N):
            cov[i,j] = scipy.special.j0(arg * abs(j-i))
            if i is not j:
                cov[i,j] = cov[i,j] #- 2**(-10)


    #cov += 1e-6 * torch.eye(N, dtype=dtype_cov)

    #cov = (cov + cov.T.conj()) / 2  # Ensure Hermitian symmetry

    return cov



def zadoff_chu_seq(u, N_zc):
    """
    Generate a Zadoff-Chu sequence.

    Parameters:
        u     : root index (must be relatively prime to N_zc)
        N_zc  : sequence length (typically a prime number)

    Returns:
        zc_seq : complex Zadoff-Chu sequence of length N_zc
    """
    if np.gcd(u, N_zc) != 1:
        raise ValueError("u must be relatively prime to N_zc")

    n = np.arange(N_zc)
    if N_zc % 2 == 0:
        zc_seq = np.exp(-1j * np.pi * u * n ** 2 / N_zc)
    else:
        zc_seq = np.exp(-1j * np.pi * u * n * (n + 1) / N_zc)

    return zc_seq


