import numpy as np
import commpy.channelcoding.convcode as cc
import sys
import math

sys.path.append('src')
from src.utils_standard_codec import fixed_permutation_interleave, fixed_permutation_deinterleave, differential_encode, differential_decode_mpsk, modulator, demodulator, zero_small_parts
from src.utils import dec2bin_digits


import numpy as np
from commpy.channelcoding import convcode

def tail_biting_encode(input_bits, trellis):
    """
    Perform tail-biting convolutional encoding by testing all initial states.

    Returns:
        encoded_bits: Encoded sequence
        init_state: The starting state used
    """
    n_states = trellis.number_states
    n_input_bits = trellis.k
    n_output_bits = trellis.n

    num_steps = len(input_bits) // n_input_bits
    assert len(input_bits) % n_input_bits == 0, "Input length must be divisible by k."

    for init_state in range(n_states):
        state = init_state
        encoded = []

        for t in range(num_steps):
            input_sym = input_bits[t * n_input_bits : (t + 1) * n_input_bits]
            input_sym_val = int("".join(str(b) for b in input_sym), 2)

            output_val = trellis.output_table[state, input_sym_val]
            next_state = trellis.next_state_table[state, input_sym_val]

            out_bits = np.array([int(b) for b in np.binary_repr(output_val, width=n_output_bits)])
            encoded.append(out_bits)

            state = next_state

        if state == init_state:
            # Success: tail-biting achieved
            return np.concatenate(encoded), init_state

    raise RuntimeError("Tail-biting condition not met for any initial state. Try modifying the input or use longer blocks.")

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
#    decoded_chunk = hard_viterbi_tail_biting(chunk, trellis)


interleaving = False
M_mod = 2
mod_type = 'DPSK'
MOD = False

np.random.seed(1)
rng = np.random.default_rng(1)

# 트렐리스 정의 (예: rate 2/3)
memory = np.array([2, 2])
g_matrix = np.array([[0o7, 0o5, 0o0],  # input 1 contributes to output 1 and 2
                     [0o0, 0o6, 0o3]])  # input 2 contributes to output 2 and 3
trellis = cc.Trellis(memory, g_matrix)

# 전체 메시지 스트림 (예시)
message_stream = np.random.randint(0, 2, 1000)
k = 24#56  # 입력 청크 길이
n = 36#84  # 출력 청크 길이 (rate 2/3)

if interleaving:
    perm = rng.permutation(k)

# === ENCODING ===
encoded_chunks = []
if MOD:
    mod_chunks = []
    if mod_type == 'DPSK':
        all_bit_seq = dec2bin_digits(np.arange(M_mod), int(math.log2(M_mod)))
        all_bit_seq = all_bit_seq.reshape(-1, 1)
        modulation_table = modulator(all_bit_seq, M_mod, mod_type)
        modulation_table = zero_small_parts(modulation_table)
    else:
        modulation_table = None
decoded_chunks = []
num_chunks = len(message_stream) // k

for i in range(num_chunks):
    start = i * k
    end = start + k
    chunk = message_stream[start:end]

    if interleaving:
        #print(chunk)
        assert chunk.shape[0] == k, f"{chunk.shape[0]}"
        chunk = fixed_permutation_interleave(chunk.copy(), perm)
        #print(fixed_permutation_deinterleave(chunk.copy(), perm))
    trellis = cc.Trellis(memory, g_matrix)
    encoded_chunk = cc.conv_encode(chunk.copy(), trellis, termination='term')  # output: 9 bits


    decoded_chunk = cc.viterbi_decode(encoded_chunk.copy(), trellis, tb_depth=15, decoding_type='hard')
    decoded_chunk = decoded_chunk[:k]

    #BER_tmp = np.mean(decoded_chunk != chunk)
    #print(f"BER: {BER_tmp:.3e}")
    decoded_chunks.append(decoded_chunk.copy())

    #encoded_chunk, init_state = tail_biting_encode(chunk.copy(), trellis)
    #encoded_chunk = encoded_chunk[:n]
    '''
    error = np.zeros_like(encoded_chunk)
    error_pos = np.random.randint(0, len(error))
    error[error_pos] = 1
    encoded_chunk ^= error  # XOR
    '''
    encoded_chunks.append(encoded_chunk.copy())

    n = encoded_chunk.shape[0]

    if MOD:
        modulated_symb = modulator(encoded_chunk, M_mod, mod_type)
        modulated_symb = zero_small_parts(modulated_symb)

        if mod_type == 'DPSK':
            modulated_symb = differential_encode(modulated_symb, n + 1)
        else:
            modulated_symb = np.reshape(modulated_symb, (-1, n))
        mod_chunks.append(modulated_symb.copy())

encoded_stream = np.concatenate(encoded_chunks)


error = np.zeros_like(encoded_stream)
error_pos = np.random.randint(0, len(error))
error[error_pos] = 1
encoded_stream ^= error  # XOR


#decoded_chunks = []
num_encoded_chunks = len(encoded_stream) // encoded_chunk.shape[0]

encoded_chunks_check = []
for i in range(num_encoded_chunks):
    start = i * encoded_chunk.shape[0]
    end = start + encoded_chunk.shape[0]
    chunk = encoded_stream[start:end]

    encoded_chunks_check.append(chunk.copy())
    if MOD:
        chunk = mod_chunks[i]
        if mod_type == 'DPSK':
            demodulated_bits = differential_decode_mpsk(chunk, M_mod, n + 1, modulation_table)
        else:
            demodulated_bits = demodulator(chunk, M_mod, mod_type)#, noise_std=0.0)
        chunk = np.reshape(demodulated_bits, (-1,))

    #demod_padded = np.concatenate([chunk.copy(), np.zeros(trellis.total_memory, dtype=int)])
    decoded_chunk = cc.viterbi_decode(chunk.copy(), trellis, tb_depth=15, decoding_type='hard')
    #decoded_chunk = hard_viterbi_tail_biting(chunk.copy(), trellis, traceback_depth=40)

    decoded_chunk = decoded_chunk[:k]

    if interleaving:
        assert decoded_chunk.shape[0] == k, {decoded_chunk.shape[0]}
        decoded_chunk = fixed_permutation_deinterleave(decoded_chunk.copy(), perm)

 #   decoded_chunks.append(decoded_chunk.copy())

decoded_stream = np.concatenate(decoded_chunks.copy())
encoded_stream_check = np.concatenate(encoded_chunks_check)

BER_enc = np.mean(encoded_stream != encoded_stream_check)
print(f"BER_enc: {BER_enc:.3e}")

# === 결과 확인 ===
message_stream_truncated = message_stream[:len(decoded_stream)]
'''
np.savetxt('message_stream.txt', np.reshape(message_stream_truncated, (-1, k)), fmt='%d')
np.savetxt('encoded_stream.txt', np.reshape(encoded_stream, (-1, n)), fmt='%d')
np.savetxt('decoded_stream.txt', np.reshape(decoded_stream, (-1, k)), fmt='%d')
'''

BER = np.mean(decoded_stream != message_stream_truncated)
print(f"Chunked BER: {BER:.3e}")


# === (2) 전체 메시지를 한 번에 인코딩 + 디코딩 ===
encoded_full = cc.conv_encode(message_stream, trellis, termination='term')
decoded_full = cc.viterbi_decode(encoded_full, trellis, tb_depth=15)
decoded_full = decoded_full[:len(message_stream)]
BER_full = np.mean(decoded_full != message_stream)
print(f"[Full-sequence] BER: {BER_full:.3e}")


print("Chunk-wise encoding and decoding work as expected, but encoding full chunks and decoding full chunks break the decoders. Reasons unclear.")