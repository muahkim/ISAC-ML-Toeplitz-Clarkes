# Loss Function for Diffusion Model
# Original Source: https://github.com/acids-ircam/diffusion_models

import numpy as np
import sys

assert sys.version_info >= (3, 5)
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader

np.random.seed(42)
from tqdm import tqdm
from utils_standard_codec import *

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(torch.tensor(np.sqrt(2.0 / np.pi)) * (x + 0.044715 * torch.pow(x, 3))))


def EbNo_to_noise(ebnodb, rate):
    '''Transform EbNo[dB] to noise power'''
    ebno = 10 ** (ebnodb / 10)
    noise_std = 1 / np.sqrt(2 * rate * ebno)
    return noise_std


def SNR_to_noise(snrdb):
    '''Transform snr[dB] to noise power'''
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


def complex_multivariate_Gaussian_loglikelihood(x, cov, threshold=1e-3, only_groundtruth=False):
    # x = [batch_size, M, n] torch.Tensor in torch.cfloat, repeated for M times. If only_groundtruth, M=1.
    # cov = covariance matrix. if only_groundtruth, [bs, n, n]. Otherwise, [M, n, n].
    bs, _, n = x.shape
    num_codewords = cov.shape[0]

    x = x.view(-1, n, 1)

    # Eigen decomposition
    eigen_values, eigen_vector = torch.linalg.eig(cov)
    eigen_values = eigen_values.real  # Only real parts are needed
    eigen_values = torch.clamp(eigen_values, min=threshold)  # Avoid numerical issues

    log_det = torch.sum(torch.log(eigen_values), dim=1)

    # Use pseudo-inverse to avoid singularity issues
    cov_inv = torch.linalg.pinv(cov)
    # Forcing the inverse matrix Hermitian
    cov_inv = (cov_inv + cov_inv.mH) / 2

    if not only_groundtruth:
        cov_inv = cov_inv[None, :, :, :].repeat(bs, 1, 1, 1).view(-1, n, n)  # [bs * M, n, n]
        log_det = log_det[None, :].repeat(bs, 1).view(-1, 1)

    # Compute likelihood terms
    denom = n * math.log(math.pi) + log_det
    denom = denom.view(bs, -1)

    # Correct Hermitian transposition
    quad = x.mH @ cov_inv @ x  # [bs * M, 1, 1]
    exponent = -quad.view(bs, -1)

    log_likelihood = (-denom + exponent).reshape(bs, -1)

    return log_likelihood.real  # Return only real part

def complex_gaussian_logpdf_via_real_equivalent(x, cov, threshold=1e-6, only_groundtruth=False):
    """
    Computes the log-likelihood of a complex Gaussian random vector using its
    real-valued equivalent representation.

    Args:
        x (torch.Tensor): Complex input vector of shape [batch_size, n].
        cov (torch.Tensor): Complex covariance matrix of shape [n, n].
        threshold (float): Regularization to ensure numerical stability.

    Returns:
        log_likelihood (torch.Tensor): Log-likelihood values of shape [batch_size].
    """
    # x = [batch_size, M, n] torch.Tensor in torch.cfloat, repeated for M times. If only_groundtruth, M=1.
    # cov = covariance matrix. if only_groundtruth, [bs, n, n]. Otherwise, [M, n, n].
    bs, _, n = x.shape
    num_codewords = cov.shape[0]

    # Construct real-valued equivalent vector
    x_real = torch.view_as_real(x)  # Shape: [batch_size, n, 2]
    x_real = x_real.reshape(bs, -1, 2 * n)  # Shape: [batch_size, 2n]

    # Construct real-valued equivalent covariance matrix
    cov_real = torch.view_as_real(cov)  # Shape: [n, n, 2]
    cov_r = cov_real[:, :, :, 0]  # Real part
    cov_i = cov_real[:, :, :, 1]  # Imaginary part

    # Construct full real-valued covariance matrix
    cov_tilde = torch.cat([
        torch.cat([cov_r, -cov_i], dim=2),
        torch.cat([cov_i, cov_r], dim=2)
    ], dim=1)  # Shape: [2n, 2n]

    # Regularization to ensure positive definiteness
    cov_tilde += threshold * torch.eye(2 * n, dtype=torch.float32, device=cov_tilde.device)[None, :, :].repeat(num_codewords, 1, 1)

    # Compute determinant and inverse of the covariance matrix
    sign, logdet = torch.linalg.slogdet(cov_tilde)  # Log determinant for stability
    cov_tilde_inv = torch.linalg.pinv(cov_tilde)  # More stable than inverse

    x_real = x_real.reshape(-1, 2*n)

    if not only_groundtruth:
        logdet = logdet[None, :].repeat(bs, 1).view(-1, 1)
        cov_tilde_inv = cov_tilde_inv[None, :, :, :].repeat(bs, 1, 1, 1).view(-1, 2 * n, 2 * n)

    # Compute quadratic form: (x^T * Sigma^-1 * x)
    quad_form = torch.einsum('bi, bij, bj -> b', x_real, cov_tilde_inv, x_real)

    quad_form = quad_form.view(-1)
    logdet = logdet.view(-1)

    # Compute log-likelihood
    log_likelihood = -0.5 * (2 * n * torch.log(torch.tensor(2 * torch.pi)) + logdet + quad_form)

    return log_likelihood.reshape(bs, -1)


def complex_multivariate_Gaussian_loglikelihood_old(x, cov, threshold=1e-4, only_groundtruth=False):
    # x = [batch_size, M, n] torch.Tensor in torch.cfloat, repeated for M times.
    # cov = covariance matrix. if only_groundtruth, [bs, n, n]. Otherwise, [M, n, n].
    bs, _, n = x.shape
    num_codewords = cov.shape[0]

    x = x.view(-1, n, 1)

    eigen_values, eigen_vector = torch.linalg.eig(cov)
    eigen_values = eigen_values.real
    idx_bool = eigen_values > threshold

    log_eigen_v = torch.log(eigen_values) * idx_bool
    log_det = torch.sum(log_eigen_v, dim=1)

    cov_inv = torch.inverse(cov)

    if only_groundtruth is False:
        cov_inv = cov_inv[None, :, :, :].repeat(bs, 1, 1, 1).view(-1, n, n)  # [bs * M, 2*n, 2*n]
        log_det = log_det[None, :].repeat(bs, 1).view(-1, 1)

    denom = n * math.log(math.pi) + log_det
    denom = denom.view(bs, -1)
    quad = torch.bmm(torch.bmm(torch.transpose(x.conj(), 1, 2), cov_inv), x)   # [bs * M, 1, 1]
    exponent = -quad.view(bs, -1)  # torch.exp(-quad).view(-1,1)
    log_likelihood = (-denom + exponent).reshape(bs, -1)

    return log_likelihood.real



def ML_det_Rayleigh_ncd(r_seq, table, cov_h, std_z, only_groundtruth=False):
    # r_seq: [batch_size, n]
    # table: list of possible codewords. shape: if only_groundtruth [bs, n]. Otherwise, [M, n].
    # cov_h: [n, n] torch.cfloat matrix. The covariance matrix of the channel vector.
    # std_z: float The additive noise standard deviation.
    # only_groundtruth: bool. If true, only the likelihood of the groundtruth is calculated without detection.

    bs, _ = r_seq.shape
    device = r_seq.device
    cov_h = cov_h.to(device)
    table = table.to(device)

    num_codewords, n = table.shape
    table = table[:, :, None] # [bs, n, 1] / [M, n, 1]

    # identity matrix
    I = torch.eye(n, dtype=torch.cfloat).to(device)  # [None, None, :,:].repeat(bs, M, 1, 1)
    I.imag = 0.

    X_cov = torch.diag_embed(table.view(-1, n)) @ cov_h[None, :, :].repeat(num_codewords, 1, 1)
    X_cov_XH = X_cov @ torch.diag_embed(table.view(-1,n)).mH
    cov_x_tilde = X_cov_XH + 2 * std_z ** 2 * I[None, :, :].repeat(num_codewords, 1, 1)

    #print(is_valid_covariance_matrix(cov_x_tilde))

    if only_groundtruth:
        log_prob = complex_gaussian_logpdf_via_real_equivalent(r_seq[:, None, :], cov_x_tilde, only_groundtruth=only_groundtruth)
    else:
        #log_prob = complex_multivariate_Gaussian_loglikelihood(r_seq[:, None, :].repeat(1, num_codewords, 1), cov_x_tilde, only_groundtruth=only_groundtruth)

        log_prob = complex_gaussian_logpdf_via_real_equivalent(r_seq[:, None, :].repeat(1, num_codewords, 1), cov_x_tilde, only_groundtruth=only_groundtruth)


    if only_groundtruth:
        return log_prob, None
    else:
        indices = torch.argmax(log_prob, dim=1)
        return log_prob, indices.view(-1, 1)



def is_valid_covariance_matrix(cov, threshold=1e-6):
    """
    Checks if a given covariance matrix is Hermitian and positive definite.

    Args:
        cov (torch.Tensor): A square complex matrix of shape (n, n).
        threshold (float): Minimum allowable eigenvalue to be considered positive definite.

    Returns:
        bool: True if the matrix is valid, False otherwise.
    """
    # Check Hermitian property: cov should be equal to its conjugate transpose
    is_hermitian = torch.allclose(cov, cov.mH, atol=1e-6)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(cov).real  # Only real parts are relevant

    # Check positive definiteness: all eigenvalues must be > threshold
    is_positive_definite = torch.all(eigenvalues > threshold)

    return is_hermitian and is_positive_definite



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

def dec2bin_np(x, bits):
    x = np.asarray(x)  # Ensure input is an array
    mask = 2 ** np.arange(bits - 1, -1, -1)
    return ((x[..., None] & mask) > 0).astype(float)

def dec2bin_digits(arr, bits=None):
    """Convert a NumPy array of integers to binary (as digit arrays)."""
    bin_arr = np.array([list(format(x, f'0{bits}b')) for x in arr], dtype=int)
    return bin_arr

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
    tau_s_a (torch.Tensor): Time delay parameter (batch_size, )
    T (float): Symbol duration

    Returns:
    torch.Tensor: Computed Ja,a(θ) value (batch_size,)
    """
    h_s_a = torch.from_numpy(h_s_a).to(x.device)
    tau_s_a = torch.from_numpy(tau_s_a).to(x.device)

    batch_size, N = x.shape  # Get batch size and sequence length
    coeff = (2 / N0) * (torch.abs(h_s_a).view(-1,1) ** 2)  # Compute coefficient (batch_size,)
    coeff = coeff.squeeze()
    # Create indices for m and n (shape: (N, N))
    m_idx = torch.arange(1, N + 1, device=x.device).view(N, 1)  # Column vector (N, 1)
    n_idx = torch.arange(1, N + 1, device=x.device).view(1, N)  # Row vector (1, N)

    # Compute sinc derivative argument: (m - n - tau_s_a / T) for batch
    tau_s_a_expanded = tau_s_a.view(batch_size, 1, 1)  # (batch_size, 1, 1)
    sinc_arg = m_idx - n_idx - tau_s_a_expanded / T  # (batch_size, N, N)

    # Compute sinc derivative
    sinc_deriv = sinc_derivative(sinc_arg) * (-1.0) # (batch_size, N, N)

    # Compute inner summation (sum over n dimension)
    inner_sum = torch.sum(x.unsqueeze(1) * sinc_deriv, dim=-1)  # (batch_size, N)

    # Compute final summation (sum over m dimension) and square
    summation = torch.sum(inner_sum * torch.conj(inner_sum), dim=-1).real  # (batch_size,)

    return coeff * summation  # (batch_size,)


def CRLB_mean(N0, h_s_a, x, tau_s_a, T, eps=1e-6):
    J_aa = 1 / ( FIM_diag_Toeplitz_delay(N0, h_s_a, x, tau_s_a, T) + eps )
    CRLB = torch.mean(J_aa) # Miller Chang bound # normalization added.
    return CRLB

def MCRB_mean(N0, dummy, x, tau_s_a, T, eps=1e-6):
    num_samples = x.size(0)
    h_s_a = np.ones(num_samples)
    J_aa = 1 / ( FIM_diag_Toeplitz_delay(N0, h_s_a, x, tau_s_a, T) + eps )
    MCRB = torch.mean(J_aa) # Miller Chang bound # normalization added.
    return MCRB



def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def snr_sigma2db(train_sigma):
    try:
        return -20.0 * math.log(train_sigma, 10)
    except:
        return -20.0 * torch.log10(train_sigma)

def Likelihood_corr_fading_ncd(y, table, noise_std):
    dist = torch.squeeze(torch.cdist(y, table, p=2) ** 2 / (noise_std ** 2))  # (bs, 1, M)
    prob = torch.exp(-dist)
    prob = prob / torch.sum(prob, dim=1)[:, None]  # normalization of the probability
    return prob

def save_arguments_to_file(args, filename="arguments.txt"):
    with open(filename, "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")