# Loss Function for Diffusion Model
# Original Source: https://github.com/acids-ircam/diffusion_models

import torch
import numpy as np
import sys
import math

assert sys.version_info >= (3, 5)
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

np.random.seed(42)
from tqdm import tqdm
from utils_standard_codec import *

sys.path.append('src/')
from channel_models import *


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(torch.tensor(np.sqrt(2.0 / np.pi)) * (x + 0.044715 * torch.pow(x, 3))))


def EbNo_to_noise(ebnodb, rate):
    '''Transform EbNo[dB]/snr to noise power'''
    ebno = 10 ** (ebnodb / 10)
    noise_std = 1 / np.sqrt(2 * rate * ebno)
    return noise_std


def SNR_to_noise(snrdb):
    '''Transform EbNo[dB]/snr to noise power'''
    snr = 10 ** (snrdb / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std


def noise_to_SNR(noise_std):
    '''Transform noise power to SNR'''
    snr = 1 / 2 / noise_std ** 2
    snrdb = 10 * math.log10(snr)
    return snrdb


def B_Ber_m(input_msg, msg):
    '''Calculate the Batch Symbol Error Rate'''
    batch, bits = msg.size()

    pred_error = torch.ne(input_msg, torch.round(msg))
    bber = torch.sum(pred_error) / (batch * bits)
    return bber


def Interleaver_set(dataset):
    '''Interleave the whole codeword set at once, set length needs to be a mupltiple of 255'''
    output = [dataset[i:i + 255].transpose() for i in range(0, len(dataset), 255)]
    return np.vstack(output)


def prediction_errors(input_msg, msg):
    pred_error = torch.ne(torch.flatten(input_msg), msg.argmax(dim=1))
    return pred_error


def block_error(prediction_matrix, erasures_matrix, erasure=False):
    if erasure:
        errors_wo_erasures = torch.logical_and(prediction_matrix, torch.logical_not(erasures_matrix))
        errors = torch.sum(errors_wo_erasures, dim=1)
        erasures = torch.sum(erasures_matrix, dim=1)
        # if torch.sum(erasures>1)>1: print(torch.sum(erasures>1))
        errors_tot = torch.sum(2 * errors + erasures > 32)
        block_error = errors_tot / 255

    else:
        errors = torch.sum(prediction_matrix, dim=1)
        block_error = torch.sum(errors > 16) / 255

    return block_error


def SER(input_msg, msg, erasure_bound=0.9):
    '''Calculate the Symbol Error Rate'''
    batch, _ = msg.size()
    msg = msg.float().view(-1, 1)
    # count erasures and give indices without erasures
    smax = torch.nn.Softmax(dim=1)
    out, _ = torch.topk(smax(msg), 1, axis=1)
    indices_erasures = (out <= erasure_bound).flatten()  # .nonzero()
    # erasures = out.size()[0]-indices_non_erasures.size()[0]

    # erasures = torch.count_nonzero(out < erasure_bound).item()

    # cut erasures from prediction
    # msg_wo_er = torch.index_select(msg, 0, indices_non_erasures.flatten())
    # input_msg_wo_er = torch.index_select(torch.flatten(input_msg), 0, indices_non_erasures.flatten())

    # print("input_msg", input_msg)
    # print("msg", msg.argmax(dim=1))

    input_msg_wo_er = torch.flatten(input_msg)
    msg_wo_er = msg.view(input_msg.size(0), -1)
    pred_error = torch.ne(input_msg_wo_er, msg_wo_er.argmax(dim=1))
    block_er = torch.sum(pred_error)
    ser = block_er / batch

    return ser, pred_error, indices_erasures


def Block_ER(input_msg, msg):
    '''Calculate the Batch Symbol Error Rate'''
    batch, bits = msg.size()

    pred_error = torch.ne(input_msg, torch.round(msg))
    ber, _ = torch.max(pred_error, dim=1)
    bber = torch.sum(ber)
    if bber < 17:
        bler = 0.
    else:
        bler = 1.
    return bler


def random_sample(batch_size=32, M=16):
    msg = torch.randint(0, 2, size=(batch_size, 1))
    return msg


def test_encoding(encoder, M=16, n=1):
    inp = np.arange(0, M)
    coding = encoder(inp).detach()
    fig = plt.figure(figsize=(4, 4))
    plt.plot(coding[:, 0], coding[:, 1], "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18, rotation=0)
    plt.grid(True)
    plt.gca().set_ylim(-2, 2)
    plt.gca().set_xlim(-2, 2)
    plt.show()


def test_noisy_codeword(data):
    rcvd_word = data[1:2000]
    fig = plt.figure(figsize=(4, 4))
    plt.plot(rcvd_word[:, 0], rcvd_word[:, 1], "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18, rotation=0)
    plt.grid(True)
    plt.gca().set_ylim(-2, 2)
    plt.gca().set_xlim(-2, 2)
    plt.show()


def minimum_distance(encoder, M, n):
    inp = np.arange(0, M)
    code = encoder(inp).detach()
    code_flat = np.reshape(code, (M, n))
    distmat = distance.cdist(code_flat, code_flat, 'euclidean') + np.identity(M) * np.exp(100)
    return np.min(distmat)


def plot_loss(encoder, M, n, step, epoch, mean_loss, X_batch, y_pred, plot_encoding):
    template = 'Iteration: {}, Epoch: {}, Loss: {:.5f}, Batch_BER: {:.5f}'
    if step % 500 == 0:
        print(template.format(step, epoch, mean_loss, B_Ber_m(X_batch, y_pred)))
        if plot_encoding:
            test_encoding(encoder, M=16, n=1)


def plot_batch_loss(epoch, mean_loss, X_batch, y_pred):
    template_outer_loop = 'Interim result for Epoch: {}, Loss: {:.5f}, Batch_BER: {:.5f}'
    print(template_outer_loop.format(epoch, mean_loss, B_Ber_m(X_batch, y_pred)))


# Approximate 16 QAM Error

def rayleigh_error(ebnodb):
    esno = 4 * 10. ** (ebnodb / 10)

    def awgn_ser(esno):
        return 1.5 * special.erfc(np.sqrt(0.1 * esno))

    def weighted_ser(x):
        weight = 2 * x * np.exp(-x * np.conj(x))
        return weight * awgn_ser(esno * x * np.conj(x))

    r, e = integrate.quad(weighted_ser, 0, np.inf)
    return r


def NN_det_batched(r_seq, table):
    # r_seq: real Tensor (bs, 1, 2n)
    # table: real Tensor (bs, M, 2n)
    dist = torch.squeeze(torch.cdist(r_seq, table, p=2))  # (bs, 1, M)
    indices = torch.argmin(dist, dim=1)
    return indices


def normpdf_zm_torch(x, det):
    # x = [batch_size, M, n] torch.Tensor, repeated for M times.
    # det = [M] torch.Tensor
    bs, M, n = x.shape
    det = det.repeat(bs)[:, None, None]
    x = x.reshape(-1, 1, n)
    denom = (2 * math.pi * det) ** .5
    num = torch.exp(-torch.bmm(x, torch.transpose(torch.conj(x), 1, 2)) / (2 * det))
    return torch.div(num, denom).reshape(-1, M)  # [batch_size, M]


def ML_det_Rayleigh(r_seq, table, std_h, std_z):
    # r_seq: [batch_size, n]
    # table: list of possible codewords. shape: [M, n]
    # indices: [batch_size, 1], telling the indices of detected symbols in the table.
    M, n = table.shape
    I = torch.ones(M, n, dtype=torch.cfloat)
    I.imag = 0.
    var_x = table * torch.conj(table)
    var_x = var_x.real
    var = var_x * std_h ** 2 + std_z ** 2 * I.real  # [M, n], per codeword, per dimension
    det = torch.prod(var, dim=1)  # [M] # determinant of a diagonal matrix
    r_seq = r_seq[:, None, :].repeat(1, M, 1)  # [batch_size, M, n]
    prob = torch.reshape(normpdf_zm_torch(r_seq, det), (-1, M))
    prob = prob.real
    indices = torch.argmax(prob, dim=1)
    return indices.view(-1, 1)


def complex_multivariate_Gaussian(x, cov):  ## This function is giving a complex probability.
    # x = [batch_size, M, n] torch.Tensor, repeated for M times.
    # cov = covariance matrix. [M, n, n]
    bs, M, n = x.shape
    det = torch.det(cov)  # [M] # determinant of a diagonal matrix
    cov_inv = torch.inverse(cov)[None, :, :, :].repeat(bs, 1, 1, 1).view(-1, n, n)  # [bs * M, n, n]
    det = det[None, :].repeat(bs, 1).view(-1, 1)
    x = x.view(-1, n, 1)
    denom = det * (math.pi ** n)
    quad = torch.bmm(torch.bmm(torch.transpose(x.conj(), 1, 2), cov_inv), x)  # [bs * M, 1, 1]
    num = torch.exp(-quad).view(-1, 1)
    return torch.div(num, denom).reshape(-1, M)  # [batch_size, M]


def complex_multivariate_Gaussian2(x, cov):
    # x = [batch_size, M, n] torch.Tensor, repeated for M times.
    # cov = covariance matrix. [M, n, n]
    bs, M, n = x.shape

    cov_real = torch.empty(M, 2 * n, 2 * n, dtype=torch.float, device=x.device)
    cov_real[:, :n, :n] = cov.real
    cov_real[:, n:, n:] = cov.real
    cov_real[:, :n, n:] = -cov.imag
    cov_real[:, n:, :n] = cov.imag
    cov_real = cov_real / 2

    x_real = torch.empty(bs, M, 2 * n, device=x.device, dtype=torch.float)
    x_real[:, :, :n] = x.real
    x_real[:, :, n:] = x.imag

    det = torch.det(cov_real)  # [M] # determinant of a diagonal matrix
    cov_inv = torch.inverse(cov_real)[None, :, :, :].repeat(bs, 1, 1, 1).view(-1, 2 * n, 2 * n)  # [bs * M, 2*n, 2*n]
    det = det[None, :].repeat(bs, 1).view(-1, 1)
    x_real = x_real.view(-1, 2 * n, 1)

    denom = (det ** 0.5) * (math.pi ** n) * (2 ** n)
    quad = torch.bmm(torch.bmm(torch.transpose(x_real, 1, 2), cov_inv), x_real) / 2  # [bs * M, 1, 1]
    num = torch.exp(-quad).view(-1, 1)
    return torch.div(num, denom).reshape(-1, M)  # [batch_size, M]


def complex_multivariate_Gaussian_loglikelihood(x, cov, threshold=1e-4):
    # x = [batch_size, M, n] torch.Tensor, repeated for M times.
    # cov = covariance matrix. [M, n, n]
    bs, M, n = x.shape

    cov_real = torch.empty(M, 2 * n, 2 * n, dtype=torch.float, device=x.device)
    cov_real[:, :n, :n] = cov.real
    cov_real[:, n:, n:] = cov.real
    cov_real[:, :n, n:] = -cov.imag
    cov_real[:, n:, :n] = cov.imag
    cov_real = cov_real / 2

    x_real = torch.empty(bs, M, 2 * n, device=x.device, dtype=torch.float)
    x_real[:, :, :n] = x.real
    x_real[:, :, n:] = x.imag

    eigen_values, eigen_vector = torch.linalg.eig(cov_real)
    eigen_values = eigen_values.real
    idx_bool = eigen_values > threshold

    log_eigen_v = torch.log(eigen_values) * idx_bool
    det = torch.sum(log_eigen_v, dim=1)

    cov_inv = torch.inverse(cov_real)
    cov_inv = cov_inv[None, :, :, :].repeat(bs, 1, 1, 1).view(-1, 2 * n, 2 * n)  # [bs * M, 2*n, 2*n]
    det = det[None, :].repeat(bs, 1).view(-1, 1)
    x_real = x_real.view(-1, 2 * n, 1)

    denom = det * 0.5 + math.log((math.pi ** n) * (2 ** n))
    quad = torch.bmm(torch.bmm(torch.transpose(x_real, 1, 2), cov_inv), x_real) / 2  # [bs * M, 1, 1]
    num = -quad.view(-1, 1)  # torch.exp(-quad).view(-1,1)
    return (-denom + num).reshape(-1, M)  # torch.div(num, denom).reshape(-1, M) # [batch_size, M]


def ML_det_Rayleigh_ncd(r_seq, table, cov_h, std_z):
    # r_seq: [batch_size, n]
    # table: list of possible codewords. shape: [M, n]
    # indices: [batch_size, 1], telling the indices of detected symbols in the table.
    bs, _ = r_seq.shape
    M, n = table.shape
    device = r_seq.device
    cov_h = cov_h.to(device)
    table = table.to(device)
    I = torch.eye(n, dtype=torch.cfloat).to(device)  # [None, None, :,:].repeat(bs, M, 1, 1)
    I.imag = 0.
    table = table[:, :, None]
    xx = torch.bmm(table, torch.transpose(torch.conj(table), 1, 2))  # [M,n,1] * [M,1,n] => [M, n, n]
    # ang_xx = xx / (xx.abs()+10 ** (-10))
    cov = xx * cov_h[None, :, :].repeat(M, 1, 1) + 2* std_z ** 2 * I[None, :, :].repeat(M, 1, 1)  # [M, n, n]

    # prob = complex_multivariate_Gaussian2(r_seq[:,None,:].repeat(1,M,1), cov)
    prob = complex_multivariate_Gaussian_loglikelihood(r_seq[:, None, :].repeat(1, M, 1), cov)
    indices = torch.argmax(prob, dim=1)
    return prob, indices.view(-1, 1)


def MMSE_H_estimator_iid(x, y, sigma_n):
    # x: shape = [batch_size, n], dtype = torch.cfloat
    # y: shape = [batch_size, n], dtype = torch.cfloat
    # sigma_n: noise standard deviation of the sensing channel. float number.

    n = x.size(1)
    assert (x.shape == y.shape)

    x_conj = torch.conj(x)
    h_hat = x_conj / (x * x_conj + sigma_n ** 2) * y

    return h_hat  # shape = [batch_size, n], dtype = torch.cfloat


def MMSE_estimator_seq(encoder, M, cov, labels, noise_std_r, device):
    MSE_batch = []
    SNR = noise_to_SNR(noise_std_r)
    loader = DataLoader(labels, shuffle=True, pin_memory=False,
                        batch_size=1000, drop_last=True)
    pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

    for it, m in pbar:
        label = m.reshape((-1, 1)).to(device)
        bs = m.size(0)

        # Forward Pass:
        # Encoding
        m = F.one_hot(label.long(), num_classes=M)
        m = m.float().reshape(-1, M)
        x = encoder(m)  # each sample is a 2n-dimensional real vector
        n = x.size(1) // 2
        x_ = x.view(-1, n, 2)  # each sample is a n x 2 tensor.
        x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

        # Communication and Radar Channel
        _, y_r, _, h_r, _ = channels_by_cov_2mat_nb(x_, cov, 0, noise_std_r, device)

        # Sensing by MMSE 
        H_r_hat = MMSE_H_estimator_iid(x_, y_r, noise_std_r)
        MSE_tmp = MSE_c(h_r, H_r_hat)

        MSE_batch.append(MSE_tmp.detach().cpu().numpy())
        pbar.set_description(
            f"SNR {round(SNR):}: MSE: {np.mean(MSE_batch):.2e}")

    return np.mean(MSE_batch)


def MMSE_estimator_seq_n(encoder, M, cov, labels, noise_std_r, device):
    MSE_batch = []
    SNR = noise_to_SNR(noise_std_r)
    loader = DataLoader(labels, shuffle=True, pin_memory=False,
                        batch_size=1000, drop_last=True)
    pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

    for it, m in pbar:
        label = m.reshape((-1, 1)).to(device)
        bs = m.size(0)

        # Forward Pass:
        # Encoding
        m = F.one_hot(label.long(), num_classes=M)
        m = m.float().reshape(-1, M)
        x = encoder(m)  # each sample is a 2n-dimensional real vector
        n = x.size(1) // 2
        x_ = x.view(-1, n, 2)  # each sample is a n x 2 tensor.
        x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

        # Communication and Radar Channel
        _, y_r, _, h_r, _ = channels_by_cov_2mat_nb(x_, cov, 0, noise_std_r, device)

        # Sensing by MMSE
        H_r_hat = MMSE_H_estimator_iid(x_, y_r, noise_std_r)
        MSE_tmp = nMSE(h_r, H_r_hat)

        MSE_batch.append(MSE_tmp.detach().cpu().numpy())
        pbar.set_description(
            f"SNR {round(SNR):}: NMSE: {np.mean(MSE_batch):.2e}")

    return np.mean(MSE_batch)


def MMSE_H_estimator_iid_w_priori(x, y, sigma_n, rho, mean_hc, sigma_hc):  # hist_Hc_samples):

    # rho is the correlation, i.e., E{h_c.h_r*}
    # hist_Hc_samples are used to calculate the statistics from realizations of H_c

    assert (x.shape == y.shape)
    x_conj = torch.conj(x)
    sigma2_hc = sigma_hc ** 2
    # sigma2_hc = torch.var(hist_Hc_samples, dim=0)
    # mean_hc = torch.mean(hist_Hc_samples, dim=0)
    sigma2_hr = 1 - rho ** 2 + (rho ** 2) * sigma2_hc
    h_hat = (sigma2_hr * x_conj / (sigma2_hr * x * x_conj + sigma_n ** 2)) * (y - rho * mean_hc * x) + rho * mean_hc

    return h_hat


def MMSE_H_estimator_corr(x, y, sigma_n, cov):  # hist_Hc_samples):

    # sigma_n is the standard deviation of the additive noise
    # cov is the covariance matrix of the communication channel.
    # x, y: [bs, n] complex tensors
    # cov: [n, n] real tensor
    assert (x.shape == y.shape)
    cov = cov.to(x.device)
    bs, n = x.shape
    cov = cov[None, :, :].repeat(bs, 1, 1)
    X = torch.zeros_like(cov, dtype=torch.cfloat)
    X[:, range(n), range(n)] = x
    y = y[:, :, None].to(x.device) # vector form
    CXh = torch.bmm(cov, X.conj())
    denom = torch.bmm(X, CXh)
    denom[:, range(n), range(n)] = denom[:, range(n), range(n)] + 2 * sigma_n ** 2
    W = torch.bmm(CXh, torch.inverse(denom))
    h_hat = torch.bmm(W, y)
    h_hat = torch.squeeze(h_hat)
    return h_hat


def MMSE_estimator_seq_corr(encoder, M, cov, labels, noise_std_r, device):
    MSE_batch = []
    SNR = noise_to_SNR(noise_std_r)
    loader = DataLoader(labels, shuffle=True, pin_memory=False,
                        batch_size=1000, drop_last=True)
    pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

    for it, m in pbar:
        label = m.reshape((-1, 1)).to(device)
        bs = m.size(0)

        # Forward Pass:
        # Encoding
        m = F.one_hot(label.long(), num_classes=M)
        m = m.float().reshape(-1, M)
        x = encoder(m)  # each sample is a 2n-dimensional real vector
        n = x.size(1) // 2
        x_ = x.view(-1, n, 2)  # each sample is a n x 2 tensor.
        x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

        # Communication and Radar Channel
        y_r, h_r = ch_corr_Rayleigh_by_cov(x_, cov, noise_std_r, device)
        # Sensing by MMSE
        H_r_hat = MMSE_H_estimator_corr(x_, y_r, noise_std_r, cov)
        MSE_tmp = MSE_c(h_r, H_r_hat)

        MSE_batch.append(MSE_tmp.detach().cpu().numpy())
        pbar.set_description(
            f"SNR {round(SNR):}: MSE: {np.mean(MSE_batch):.2e}")

    return np.mean(MSE_batch)


def MMSE_estimator_seq_corr_(M, cov, labels, noise_std_r, device):
    MSE_batch = []
    SNR = noise_to_SNR(noise_std_r)
    loader = DataLoader(labels, shuffle=True, pin_memory=False,
                        batch_size=1000, drop_last=True)
    pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

    for it, m in pbar:
        label = m.reshape((-1, 1)).to(device)
        bs = m.size(0)

        # Forward Pass:
        # Encoding
        m = F.one_hot(label.long(), num_classes=M)
        m = m.float().reshape(-1, M)
        x = qam_mapper_n(m, M)
        # x = encoder(m)  # each sample is a 2n-dimensional real vector
        n = x.size(1) // 2
        x_ = x.view(-1, n, 2)  # each sample is a n x 2 tensor.
        x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

        # Communication and Radar Channel
        y_r, h_r = ch_corr_Rayleigh_by_cov(x_, cov, noise_std_r, device)
        # Sensing by MMSE
        H_r_hat = MMSE_H_estimator_corr(x_, y_r, noise_std_r, cov)
        MSE_tmp = MSE_c(h_r, H_r_hat)

        MSE_batch.append(MSE_tmp.detach().cpu().numpy())
        pbar.set_description(
            f"SNR {round(SNR):}: MSE: {np.mean(MSE_batch):.2e}")

    return np.mean(MSE_batch)


def LS_estimator_seq(encoder, M, cov, labels, noise_std_r, device):
    MSE_batch = []
    SNR = noise_to_SNR(noise_std_r)
    loader = DataLoader(labels, shuffle=True, pin_memory=False,
                        batch_size=1000, drop_last=True)
    pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

    for it, m in pbar:
        label = m.reshape((-1, 1)).to(device)
        bs = m.size(0)

        # Forward Pass:
        # Encoding
        m = F.one_hot(label.long(), num_classes=M)
        m = m.float().reshape(-1, M)
        x = encoder(m)  # each sample is a 2n-dimensional real vector
        n = x.size(1) // 2
        x_ = x.view(-1, n, 2)  # each sample is a n x 2 tensor.
        x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

        # Communication and Radar Channel
        _, y_r, _, h_r, _ = channels_by_cov_2mat_nb(x_, cov, 0, noise_std_r, device)

        # Sensing by MMSE 
        H_r_hat = y_r / x_
        MSE_tmp = nMSE(h_r, H_r_hat)

        MSE_batch.append(MSE_tmp.detach().cpu().numpy())
        pbar.set_description(
            f"SNR {round(SNR):}: MSE: {np.mean(MSE_batch):.2e}")

    return np.mean(MSE_batch)


def onehot2bin(m_1h):
    bs = m_1h.size(0)
    M = m_1h.size(1)

    k = int(math.log2(M))

    m_bin = torch.zeros(bs, k)
    m = torch.argmax(m_1h, dim=1)
    m = 2 ** k - m - 1

    for i in range(k):
        m_bin[:, i] = m // (2 ** (k - i - 1))
        m = m % (2 ** (k - i - 1))

    return m_bin.int()


def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


def MSE_c(a, a_hat):
    assert a.shape == a_hat.shape
    diff = a - a_hat
    sq = diff * torch.conj(diff)
    sq = sq.real
    return torch.mean(sq)


def RSE_c(a, a_hat):  # Relative squared error. Since mean squared error weighting the errors of large values.
    assert a.shape == a_hat.shape
    bs = a.shape[0]
    diff = a - a_hat
    sq = diff * torch.conj(diff)
    a_bar = torch.mean(a_hat, dim=0)
    diff_2 = a - a_bar[None, :].repeat(bs, 1)
    sq2 = diff_2 * torch.conj(diff_2)
    sq = sq.real / (sq2.real + 2 ** (-10))

    return torch.mean(sq)


def nMSE(a, a_hat):  # normalized MSE
    assert a.shape == a_hat.shape
    diff = a - a_hat
    sq = diff * torch.conj(diff) / (a * torch.conj(a))
    sq = sq.real
    return torch.mean(sq)


def MSE_diag(a, a_hat):
    n = a.size(1)
    assert a.size(1) == a.size(2)
    assert a_hat.size(1) == a_hat.size(2)
    a_diag = a[:, np.arange(n), np.arange(n)]
    a_hat_diag = a_hat[:, np.arange(n), np.arange(n)]
    diff = a_diag - a_hat_diag
    sq = diff * torch.conj(diff)
    sq = sq.real

    return torch.mean(sq)


def BER(a, b):  # Bit error rates of onehot-encoded sequence and its prediction
    M = b.size(1)
    b = torch.argmax(b, dim=1)
    b = F.one_hot(b, num_classes=M)

    a = onehot2bin(a)
    b = onehot2bin(b)

    ne = torch.ne(a, b)
    ber = torch.mean(ne.float())
    return ber

import torch

def sinc_derivative(x):
    """
    Computes the derivative of the sinc function for batched x using PyTorch.
    sinc(x) = sin(πx) / (πx)

    Parameters:
    x (torch.Tensor): Input tensor of shape (batch_size, N)

    Returns:
    torch.Tensor: Derivative of sinc(x) of shape (batch_size, N)
    """
    # Handle division by zero with L'Hôpital's rule
    derivative = torch.where(
        torch.abs(x) < 1e-6,  # Small threshold to avoid instability
        torch.tensor(0.0, dtype=x.dtype, device=x.device),
        (torch.cos(torch.pi * x) / x) - (torch.sin(torch.pi * x) / (torch.pi * x**2))
    )
    return derivative


def FIM_diag_Toeplitz_delay(N0, h_s_a, x, tau_s_a, T):
    """
    Computes the Fisher Information Matrix (FIM) diagonal element Ja,a(θ) in a batched manner using PyTorch.

    Parameters:
    N0 (float): Noise power spectral density
    h_s_a (torch.Tensor): Complex channel coefficient (batch_size,)
    x (torch.Tensor): Input signal sequence (batch_size, N)
    tau_s_a (torch.Tensor): Time delay parameter (batch_size,)
    T (float): Symbol duration

    Returns:
    torch.Tensor: Computed Ja,a(θ) value (batch_size,)
    """
    h_s_a = torch.from_numpy(h_s_a).to(x.device)
    tau_s_a = torch.from_numpy(tau_s_a).to(x.device)

    batch_size, N = x.shape  # Get batch size and sequence length
    coeff = (2 / N0) * (torch.abs(h_s_a) ** 2)  # Compute coefficient (batch_size,)

    # Create indices for m and n (shape: (N, N))
    m_idx = torch.arange(1, N + 1, device=x.device).view(N, 1)  # Column vector (N, 1)
    n_idx = torch.arange(1, N + 1, device=x.device).view(1, N)  # Row vector (1, N)

    # Compute sinc derivative argument: (m - n - tau_s_a / T) for batch
    tau_s_a_expanded = tau_s_a.view(batch_size, 1, 1)  # (batch_size, 1, 1)
    sinc_arg = m_idx - n_idx - tau_s_a_expanded / T  # (batch_size, N, N)

    # Compute sinc derivative
    sinc_deriv = sinc_derivative(sinc_arg)  # (batch_size, N, N)

    # Compute inner summation (sum over n dimension)
    inner_sum = torch.sum(x.unsqueeze(1) * sinc_deriv, dim=-1)  # (batch_size, N)

    # Compute final summation (sum over m dimension) and square
    summation = torch.sum(inner_sum * torch.conj(inner_sum), dim=-1).real  # (batch_size,)

    return coeff * summation  # (batch_size,)

def CRLB(N0, h_s_a, x, tau_s_a, T, eps=1e-6):
    J_aa = 1 / ( FIM_diag_Toeplitz_delay(N0, h_s_a, x, tau_s_a, T) + eps )
    CRLB = torch.sum(J_aa)
    return CRLB

