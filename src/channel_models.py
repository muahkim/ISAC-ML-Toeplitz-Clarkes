# The channel models are defined here.
from __future__ import division, print_function  # Python 2 compatibility

from functools import partial

import torch
import math
import numpy as np

from numpy import abs, sqrt, sum, zeros, identity, hstack, einsum, trace, kron, absolute, fromiter, array, exp, \
    pi, cos
from numpy.random import randn, random, standard_normal
from scipy.linalg import sqrtm, cholesky
from scipy.integrate import quad_vec
import scipy.integrate as spi


def ch_corr_Rayleigh_by_cov(x, cov, noise_std, device):
    x = x.to(device)
    bs = x.size(0)
    n = x.size(1)
    assert cov.shape == (n, n)

    # generate normally distributed matrices
    h = torch.normal(mean=0., std=1., size=(bs, n), device=device, dtype=torch.cfloat)
    h = h[:, :, None]
    # Compute a matrix A such that A*A^H = Cov.

    # Compute the Cholesky decomposition.
    A = torch.linalg.cholesky(cov, upper=False)  # torch.cholesky(Cov, upper=False)
    # print(torch.mm(A, A.conj().T))
    # print(cov)

    A = A[None, :, :].repeat(bs, 1, 1).to(device)

    # Correlated channel matrices h = Ah,  A[n,n] h[n, 1]
    h = torch.bmm(A, h)
    h = h.view(bs, n)

    y_c = h * x
    #    x_r = torch.view_as_real(h)
    #    print(torch.std(x_r[:,:,0]) ** 2)
    #    print(torch.std(x_r[:,:,1]) ** 2)

    y_c = y_c + torch.normal(mean=0, std=2 **0.5 * noise_std, size=x.size(), dtype=torch.cfloat).to(device)

    return y_c, h


def ch_Rayleigh(ch_input, h_std, noise_std, device):
    # y = hx + z
    # The diagonal element of H follows a Rayleigh distribution.

    ch_input = ch_input.to(device)
    h = torch.normal(mean=0, std=h_std, size=ch_input.size(), dtype=torch.cfloat).to(device)  # fading complex Gaussian
    z = torch.normal(mean=0, std=2 ** 0.5 * noise_std, size=ch_input.size(), dtype=torch.cfloat).to(device)  # AWGN

    ch_output = torch.zeros_like(ch_input).to(device)
    ch_output = torch.mul(h, ch_input) + z

    return ch_output, h


def sinc_pulse(t, T=1.0):
    """
    Computes the sinc pulse function.

    Parameters:
    t : float or numpy array - Time variable
    T : float - Scaling parameter (default T=1)

    Returns:
    sinc(T * t) pulse value
    """
    return np.sinc(t / T)  # np.sinc(x) in NumPy is sin(πx)/(πx)


def g_m_minus_n(P, h, tau, T, delta, p=sinc_pulse):
    """
    Computes g_{m-n} for a given difference delta = (m-n).

    Parameters:
    P     : int - Number of paths
    h     : array of cfloats - Channel coefficients of all symbols of all paths (batchsize x P)
    tau   : list of floats - Delays (tau_c^(i) or tau_s^(i))
    T     : float - Symbol period
    delta : int - Difference (m-n)

    Returns:
    g_{m-n} value (scalar)
    """

    result = 0
    for j in range(h.shape[0]):
        for i in range(P):
            h_i = h[j,i]
            tau_i = tau[j,i]
            integral, _ = spi.quad(lambda t: p(t - delta * T - tau_i, T) * np.conj(p(t, T)), -np.inf, np.inf)

            result += h_i * integral#[:,None]

    return result


def construct_Toeplitz_matrix(N, P, h, tau, T):
    """
    Constructs an batchsize x N x N Toeplitz matrix where (m, n)-th element is g_{m-n}.

    Parameters:
    N   : int - Size of the matrix (N x N)
    P   : int - Number of paths
    h   : array of cfloats - Channel coefficients for all symbols of all paths, (batchsize x P)
    tau : list of floats - Delays (batchsize x N)
    T   : float - Symbol period

    Returns:
    batchsize x N x N numpy Toeplitz matrix
    """
    batchsize = h.shape[0]
    # Compute g_{m-n} only for unique values of (m-n)
    unique_g = {delta: g_m_minus_n(P, h, tau, T, delta) for delta in range(-(N - 1), N)}

    # Construct the Toeplitz matrix using these precomputed values
    G = np.zeros((batchsize, N, N), dtype=np.complex64)
    for m in range(N):
        for n in range(N):
            G[:, m, n] = unique_g[m - n]  # Assign based on (m-n)

    return G


def construct_Toeplitz_matrices(N, P, h_batch, tau_batch, T):
    """
    Constructs a sequence of N x N Toeplitz matrices for batched (h, tau) pairs.

    Parameters:
    N        : int - Size of the matrix (N x N)
    P        : int - Number of paths (Pc or Ps)
    h_batch  : list of lists - Channel coefficients for each batch
    tau_batch: list of lists - Delays for each batch
    T        : float - Symbol period

    Returns:
    List of N x N numpy Toeplitz matrices (one for each batch entry)
    """

    def g_m_minus_n(P, h, tau, T, delta):
        # Define the function that computes g_{m-n} given P, h, tau, T, and delta
        # (Implementation assumed to be available)
        pass

    batch_size = len(h_batch)
    toeplitz_matrices = []

    for i in range(batch_size):
        h = h_batch[i]
        tau = tau_batch[i]

        # Compute g_{m-n} only for unique values of (m-n)
        unique_g = {delta: g_m_minus_n(P, h, tau, T, delta) for delta in range(-(N - 1), N)}

        # Construct the Toeplitz matrix using these precomputed values
        G = np.zeros((N, N), dtype=complex)
        for m in range(N):
            for n in range(N):
                G[m, n] = unique_g[m - n]  # Assign based on (m-n)

        toeplitz_matrices.append(G)

    return toeplitz_matrices


def Toeplitz_channel(input, P, T, h_mean=0+0j, h_var=1, N0=1, tau_max=10):
    """
    Gives the channel output of the given input for defined Toeplitz channel model.

    Parameters:
    input   : torch tensor - Channel input (bs, N)
    P       : int - Number of paths
    T       : float - Symbol period
    h_mean  : complex float - mean value of the channel coefficient
    h_var   : float - variance of the channel coefficient
    tau_max : float - max value of the path delay


    Returns:
    (bs, N) torch tensor - Channel output
    """
    bs, N = input.shape

    # Construct the channel matrix
    #h = torch.randn(bs, P, dtype=torch.cfloat, device=input.device) * ( (h_var/2) ** 0.5 ) + h_mean
    #tau = torch.empty(bs, P, device=input.device).uniform_(0, tau_max)

    # Complex-valued random numbers in NumPy (equivalent to PyTorch's `torch.randn` with `cfloat`)
    h_real = np.random.randn(bs, P) * ((h_var / 2) ** 0.5) + h_mean
    h_imag = np.random.randn(bs, P) * ((h_var / 2) ** 0.5)  # Imaginary part
    h = h_real + 1j * h_imag  # Combine real and imaginary parts

    # Uniformly distributed random values in [0, tau_max] (equivalent to `torch.empty.uniform_`)
    tau = np.random.uniform(0, tau_max, (bs, P))

    H = torch.from_numpy(construct_Toeplitz_matrix(N, P, h, tau, T)).to(input.device)
    w = torch.randn_like(input, dtype=torch.cfloat, device=input.device) * ( (N0/2) ** 0.5 )

    y = torch.bmm(H, input[:,:,None]).reshape(bs, N) + w

    return y, h, tau



class _FlatChannel(object):

    def __init__(self):
        self.noises = None
        self.channel_gains = None
        self.unnoisy_output = None

    def generate_noises(self, dims):

        """
        Generates the white gaussian noise with the right standard deviation and saves it.

        Parameters
        ----------
        dims : int or tuple of ints
                Shape of the generated noise.
        """

        # Check channel state
        assert self.noise_std is not None, "Noise standard deviation must be set before propagation."

        # Generate noises
        if self.isComplex:
            self.noises = (standard_normal(dims) + 1j * standard_normal(dims)) * self.noise_std * 0.5
        else:
            self.noises = standard_normal(dims) * self.noise_std

    def set_SNR_dB(self, SNR_dB, code_rate: float = 1., Es=1):

        """
        Sets the noise standard deviation based on SNR expressed in dB.

        Parameters
        ----------
        SNR_dB      : float
                        Signal to Noise Ratio expressed in dB.

        code_rate   : float in (0,1]
                        Rate of the used code.

        Es          : positive float
                        Average symbol energy
        """

        self.noise_std = sqrt((self.isComplex + 1) * self.nb_tx * Es / (code_rate * 10 ** (SNR_dB / 10)))

    def set_SNR_lin(self, SNR_lin, code_rate=1, Es=1):

        """
        Sets the noise standard deviation based on SNR expressed in its linear form.

        Parameters
        ----------
        SNR_lin     : float
                        Signal to Noise Ratio as a linear ratio.

        code_rate   : float in (0,1]
                        Rate of the used code.

        Es          : positive float
                        Average symbol energy
        """

        self.noise_std = sqrt((self.isComplex + 1) * self.nb_tx * Es / (code_rate * SNR_lin))

    @property
    def isComplex(self):
        """ Read-only - True if the channel is complex, False if not."""
        return self._isComplex


class CorrRayleighChannel(_FlatChannel):
    """
    Constructs a SISO channel with a flat fading.
    The channel coefficient are normalized i.e. the mean magnitude is 1.

    Parameters
    ----------
    noise_std    : float, optional
                   Noise standard deviation.
                   *Default* value is None and then the value must set later.

    fading_param : tuple of 2 floats, optional
                   Parameters of the fading (see attribute for details).
                   *Default* value is (1,0) i.e. no fading.

    Attributes
    ----------
    fading_param : tuple of 2 floats
                   Parameters of the fading. The complete tuple must be set each time.
                   Raise ValueError when sets with value that would lead to a non-normalized channel.

                        * fading_param[0] refers to the mean of the channel gain (Line Of Sight component).

                        * fading_param[1] refers to the variance of the channel gain (Non Line Of Sight component).

                   Classical fadings:

                        * (1, 0): no fading.

                        * (0, 1): Rayleigh fading.

                        * Others: rician fading.

    noise_std       : float
                       Noise standard deviation. None is the value has not been set yet.

    isComplex       : Boolean, Read-only
                        True if the channel is complex, False if not.
                        The value is set together with fading_param based on the type of fading_param[0].

    k_factor        : positive float, Read-only
                        Fading k-factor, the power ratio between LOS and NLOS.

    nb_tx           : int = 1, Read-only
                        Number of Tx antennas.

    nb_rx           : int = 1, Read-only
                        Number of Rx antennas.

    noises          : 1D ndarray
                        Last noise generated. None if no noise has been generated yet.

    channel_gains   : 1D ndarray
                        Last channels gains generated. None if no channels has been generated yet.

    unnoisy_output  : 1D ndarray
                        Last transmitted message without noise. None if no message has been propagated yet.

    Raises
    ------
    ValueError
                    If the fading parameters would lead to a non-normalized channel.
                    The condition is :math:`|param[1]| + |param[0]|^2 = 1`
    """

    @property
    def nb_tx(self):
        """ Read-only - Number of Tx antennas, set to 1 for SISO channel."""
        return 1

    @property
    def nb_rx(self):
        """ Read-only - Number of Rx antennas, set to 1 for SISO channel."""
        return 1

    def __init__(self, noise_std=None, cov=None):
        super(CorrRayleighChannel, self).__init__()
        self.noise_std = noise_std
        self.cov = cov

    def propagate(self, msg):

        """
        Propagates a message through the channel.

        Parameters
        ----------
        msg : 1D ndarray
                Message to propagate.

        Returns
        -------
        channel_output : 1D ndarray
                            Message after application of the fading and addition of noise.

        Raises
        ------
        TypeError
                        If the input message is complex but the channel is real.

        AssertionError
                        If the noise standard deviation as not been set yet.
        """

        if isinstance(msg[0], complex) and not self.isComplex:
            raise TypeError('Trying to propagate a complex message in a real channel.')
        nb_symb = len(msg)
        n = np.shape(self.cov)[0]
        nb_frame = nb_symb // n
        nb_symb = nb_frame * n # to truncate
        # Generate noise
        self.generate_noises(nb_symb)



        # cholesky decomposition of covariance matrix
        A = cholesky(self.cov, lower=True) # n by n matrix

        # Generate channel
        # generate random h
        h_T = 0.5*np.random.normal(0, 1, (nb_frame, 1, n)) \
            +0.5j * np.random.normal(0, 1, (nb_frame, 1, n)) # h^T

        # correlate the h
        h_H = h_T.conj()
        A_H = A.conj().T
        self.channel_gains = np.transpose(np.matmul(h_H, A_H).conj(), (0, 2, 1)).flatten() # Ah=(h^HA^H)^H for every frame.

        # Generate outputs
        self.unnoisy_output = self.channel_gains * msg

        return self.unnoisy_output + self.noises
    '''
    @property
    def fading_param(self):
        """ Parameters of the fading (see class attribute for details). """
        return self._fading_param

    @fading_param.setter
    def fading_param(self, fading_param):
        if fading_param[1] + absolute(fading_param[0]) ** 2 != 1:
            raise ValueError("With this parameters, the channel would add or remove energy.")

        self._fading_param = fading_param
        self._isComplex = isinstance(fading_param[0], complex)

    @property
    def k_factor(self):
        """ Read-only - Fading k-factor, the power ratio between LOS and NLOS """
        return absolute(self.fading_param[0]) ** 2 / absolute(self.fading_param[1])
    '''