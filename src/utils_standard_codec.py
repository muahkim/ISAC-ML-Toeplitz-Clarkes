# Loss Function for Diffusion Model
# Original Source: https://github.com/acids-ircam/diffusion_models

import torch
import numpy as np
import sys
import scipy
import math

assert sys.version_info >= (3, 5)
np.random.seed(42)

sys.path.append('src/')
from utils import *
import utils
from channel_models import *

import commpy.channelcoding.convcode as cc
import commpy.modulation as modulation


# Approximate 16 QAM Error

def SIXT_QAM_sim(ebnodb):
    return (3.0 / 2) * special.erfc(np.sqrt((4.0 / 10) * 10. ** (ebnodb / 10)))


def BPSK_rayleigh(ebnodb, n, k):
    ebn0 = 10. ** (ebnodb / 10)
    return 0.5 * (1 - np.sqrt((n / k) * ebn0 / ((n / k) * ebn0 + 1)))


def MQAM_rayleigh_approx(M, ebnodb):
    ebno = 10. ** (ebnodb / 10)
    esno = 4 * ebno
    # Goldsmith, p.185, 6.3.2, Eqn 6.61, alphaM=4, betaM=3/(M-1)
    a = 3.25
    b = 3 / (M - 1)
    e = b * esno
    return (a / 2) * (1 - np.sqrt(0.5 * e / (1 + 0.5 * e))), a / (2 * b * esno)


def PSK(m, M, avg_power, device="cpu"):  # m is the message index

    x = torch.zeros([m.shape[0], 2])  # complex channel
    rad = torch.linspace(0, 2 * math.pi, steps=M + 1, device=device)[:M]  # phase

    x[:, 0] = math.sqrt(avg_power) * torch.cos(rad[m])
    x[:, 1] = math.sqrt(avg_power) * torch.sin(rad[m])
    return x


def PSK_1h(m, avg_power):  # m is one-hot encdoed tensor
    M = m.size()[1]  # message cardinality
    label = torch.argmax(m, dim=1)

    x = torch.zeros([m.size()[0], 2])  # complex channel
    rad = torch.linspace(0, 2 * math.pi, steps=M + 1)[:M]  # phase

    x[:, 0] = math.sqrt(avg_power) * torch.cos(rad[label])
    x[:, 1] = math.sqrt(avg_power) * torch.sin(rad[label])
    return x


def hamming(bits):
    # Return given 4 bits plus parity bits for bits (1,2,3), (2,3,4) and (1,3,4)
    t1 = parity(bits, [0, 1, 3])
    t2 = parity(bits, [0, 2, 3])
    t3 = parity(bits, [1, 2, 3])
    return t1 + t2 + bits[0] + t3 + bits[1:]  # again saying, works only for 7,4


def parity(s, indicies):
    # Compute the parity bit for the given string s and indices
    sub = ""
    for i in indicies:
        sub += s[i]
    return str(str.count(sub, "1") % 2)


def qam16_mapper_n(m):
    # m takes in a vector of messages
    x = np.linspace(-3, 3, 4)
    y = np.meshgrid(x, x)
    z = np.array(y).reshape(2, 16)
    z = z / np.std(z)

    return np.array([[z[0][i], z[1][i]] for i in m])


def qam_mapper_n(m, M):
    # m takes in a vector of messages
    # M is the cardinality of message. Should be a power of 2.
    x = np.linspace(-3, 3, int(M ** 0.5))
    y = np.meshgrid(x, x)
    z = np.array(y).reshape(2, M)
    z = z / np.std(z) / (2 ** 0.5)
    return np.array([[z[0][i], z[1][i]] for i in m])


def cc_mapper(k, M_qam, rep=2, mod_type='PSK'):
    # m is the integer messages
    # k is the input bit sequence length
    M = int(2 ** k)
    m = np.arange(M, dtype=int)
    ##  binary representation of the messages.
    m_ = np.zeros((M, k * rep), dtype=int)
    for i in m:
        str_tmp = np.binary_repr(i, width=int(k)) * rep
        for j in range(k * rep):
            m_[i, j] = int(str_tmp[j])
    ##  repetition.
    np.random.seed(42)
    message_bits = np.zeros_like(m_)
    coded_bits = np.zeros((M, 2 * rep * k), dtype=int)
    modulated = np.zeros((M, int(2 * rep * k / math.log2(M_qam))), dtype=complex)
    generator_matrix = np.array([[5, 7]])  # generator branches

    trellis = cc.Trellis(np.array([M_qam]), generator_matrix)  # Trellis structure
    if mod_type == 'PSK':
        modem = modulation.PSKModem(M_qam)  # M-PSK modem initialization
    elif mod_type == 'QAM':
        modem = modulation.QAMModem(M_qam)
    else:
        assert False
    for i in range(M):
        np.random.seed(42) # this is to fix the interleaving randomness
        m_tmp = m_[i, :]
        ##   encoding with a convolutonal code and interleaving randomly
        coded_bits_tmp =  cc.conv_encode(m_tmp, trellis, termination='cont')
        np.random.shuffle(coded_bits_tmp)
        coded_bits[i, :] = coded_bits_tmp  # encoding and interleaving
        
        ##   QAM
        modulated[i, :] = modem.modulate(coded_bits[i, :])

    if mod_type == 'QAM':
        modulated = modulated / (10 ** 0.5)  # modulation
    return modulated


def test_ER_CC(M, n, M_qam, snr_range, cov, device, rep=2, num_samples=10 ** 5,
               batch_size=100, mod_type='PSK'):  # with repetition code.
    k = int(math.log2(M))
    rate = k / n
    BER_AE_seq = []
    SER_AE_seq = []
    table = torch.tensor(cc_mapper(k, M_qam, rep, mod_type), dtype=torch.cfloat)
    # for checking the power constraint
    print(torch.sum(torch.view_as_real(table).view(M, -1) ** 2, dim=1))

    def run_epoch():
        m = np.random.randint(low=0, high=M, size=[num_samples])
        m_batched = np.split(m, num_samples // batch_size)
        ber_batch = []
        ser_batch = []
        for m in m_batched:
            m_ = [np.binary_repr(i, width=int(k)) for i in m] # repetition coding
            x_qam_c = table[m, :] # modulation
            y_c, h_c = ch_corr_Rayleigh_by_cov(x_qam_c, cov, AWGN_std_c, device)

            prob, m_hat = utils.ML_det_Rayleigh_ncd(y_c, table, cov, AWGN_std_c)
            bool_sym = torch.ne(torch.tensor(m, dtype=int), m_hat.int().view(-1))
            SER_tmp = torch.sum(bool_sym) / m_hat.shape[0]

            labels_bin = torch.squeeze(utils.dec2bin(torch.tensor(m, dtype=int), k))
            m_d_bin = torch.squeeze(utils.dec2bin(m_hat.int(), k))
            ne = torch.ne(labels_bin, m_d_bin)
            BER_tmp = torch.mean(ne.float())

            ber_batch.append(BER_tmp.item())
            ser_batch.append(SER_tmp.item())
        return np.mean(ber_batch), np.mean(ser_batch)

    for epoch, snr in enumerate(snr_range):
        AWGN_std_c = utils.EbNo_to_noise(snr, rate)  # for a normalized signal (per symbol)

        tmp_ser = 0
        tmp_ber = 0

        it_in_while = 0
        while tmp_ber * num_samples * (
                it_in_while) < 1000:  # To guarantee we have enough samples for the Monte-Carlo method.
            if it_in_while > 1:
                print(
                    f"epoch {epoch} The number of samples is not enough. One more epoch is running. Total # of samples used: ",
                    num_samples * it_in_while)
            tmp_ber2, tmp_ser2 = run_epoch()
            tmp_ser = (it_in_while * tmp_ser + tmp_ser2) / (
                    it_in_while + 1)  # taking average of the error probability.
            tmp_ber = (it_in_while * tmp_ber + tmp_ber2) / (it_in_while + 1)
            it_in_while += 1

        # ser.append(tmp_ser)
        BER_AE_seq.append(tmp_ber)
        SER_AE_seq.append(tmp_ser)
    return BER_AE_seq, SER_AE_seq


def test_MSE_CC(M, n, M_qam, snr_range, cov, device, rep=2, mod_type='PSK', num_samples=10 ** 5,
                batch_size=100):  # with repetition code.
    MSE_AE_seq_qam = []
    k = int(math.log2(M))

    table = torch.tensor(cc_mapper(k, M_qam, rep, mod_type), dtype=torch.cfloat, device=device)
    #print(torch.mean(torch.sum(torch.view_as_real(table).view(M, -1) ** 2, dim=1)))

    def run_epoch():
        m = np.random.randint(low=0, high=M, size=[num_samples])
        m_batched = np.split(m, num_samples // batch_size)
        MSE_batch = []
        for m in m_batched:
            m_ = [np.binary_repr(i, width=int(k)) for i in m]
            x_qam_c = table[m, :]
            # _, y_r, _, h_r, _ = channels_by_cov_2mat_nb(x_psk_c, cov, 0, AWGN_std_r, device)
            y_r, h_r = ch_corr_Rayleigh_by_cov(x_qam_c, cov, AWGN_std_r, device)
            # H_r_hat = MMSE_H_estimator_iid(x_psk_c, y_r, AWGN_std_r)
            h_r_hat = utils.MMSE_H_estimator_corr(x_qam_c, y_r, AWGN_std_r, cov)
            ## BER
            MSE_tmp = utils.MSE_c(h_r, h_r_hat)
            MSE_batch.append(MSE_tmp.detach().cpu().numpy())
        return np.mean(MSE_batch)

    for epoch, snr in enumerate(snr_range):
        AWGN_std_r = utils.EbNo_to_noise(snr, k / n)  # for a normalized signal (per symbol)
        tmp_mse = 0
        it_in_while = 0
        while tmp_mse * num_samples * (
                it_in_while) < 1000:  # To guarantee we have enough samples for the Monte-Carlo method.
            if it_in_while > 1:
                print(
                    f"epoch {epoch} The number of samples is not enough. One more epoch is running. Total # of samples used: ",
                    num_samples * it_in_while)
            tmp_mse2 = run_epoch()
            tmp_mse = (it_in_while * tmp_mse + tmp_mse2) / (
                    it_in_while + 1)  # taking average of the error probability.
            it_in_while += 1
        # ser.append(tmp_ser)
        MSE_AE_seq_qam.append(tmp_mse)
    return MSE_AE_seq_qam


def test_MSE_OOK(M, n, snr_range, cov, device, num_samples=10 ** 5, batch_size=100):
    MSE_AE_seq_ook = []
    k = int(math.log2(M))

    def run_epoch():
        m = np.random.randint(low=0, high=M, size=[num_samples])
        m_batched = np.split(m, num_samples // batch_size)
        MSE_batch = []
        for m in m_batched:
            x_ook = torch.zeros(batch_size, n, dtype=torch.cfloat, device=device)
            x_ook[:, 0] = n ** 0.5 + 0j
            y_r, h_r = ch_corr_Rayleigh_by_cov(x_ook, cov, AWGN_std_r, device)
            # H_r_hat = MMSE_H_estimator_iid(x_psk_c, y_r, AWGN_std_r)
            h_r_hat = utils.MMSE_H_estimator_corr(x_ook, y_r, AWGN_std_r, cov).to(device)
            MSE_tmp = utils.MSE_c(h_r, h_r_hat)
            MSE_batch.append(MSE_tmp.detach().cpu().numpy())
        return np.mean(MSE_batch)

    for epoch, snr in enumerate(snr_range):
        AWGN_std_r = utils.EbNo_to_noise(snr, k / n)  # for a normalized signal (per symbol)

        tmp_mse = 0

        it_in_while = 0
        while tmp_mse * num_samples * (
                it_in_while) < 1000:  # To guarantee we have enough samples for the Monte-Carlo method.
            if it_in_while > 1:
                print(
                    f"epoch {epoch} The number of samples is not enough. One more epoch is running. Total # of samples used: ",
                    num_samples * it_in_while)
            tmp_mse2 = run_epoch()
            tmp_mse = (it_in_while * tmp_mse + tmp_mse2) / (
                    it_in_while + 1)  # taking average of the error probability.
            it_in_while += 1

        # ser.append(tmp_ser)
        MSE_AE_seq_ook.append(tmp_mse)
    return MSE_AE_seq_ook

def Rayleigh_covariance_generator(N, f_dopp_norm, dtype_cov=torch.cfloat):
    # parameter scaling
    # the zeroth Bessel function
    # put the function output in the N x N matrix correctly.
    assert f_dopp_norm < 1 and f_dopp_norm > 0, "The normalized doppler frequency out of range. It should be in (0,1)."
    arg = 2 * math.pi * f_dopp_norm

    cov = torch.empty(N, N, dtype=dtype_cov)
    for i in range(N):
        for j in range(N):
            cov[i,j] = scipy.special.jv(0, arg * (j-i))
            if i is not j:
                cov[i,j] = cov[i,j] - 2**(-10)


    #cov += 1e-6 * torch.eye(N, dtype=dtype_cov)

    #cov = (cov + cov.T.conj()) / 2  # Ensure Hermitian symmetry

    return cov