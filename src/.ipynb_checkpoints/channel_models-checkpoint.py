# The channel models are defined here.
from __future__ import division, print_function  # Python 2 compatibility

import torch
import math
import numpy as np

from numpy import abs, sqrt, sum, zeros, identity, hstack, einsum, trace, kron, absolute, fromiter, array, exp, \
    pi, cos
from numpy.random import randn, random, standard_normal
from scipy.linalg import sqrtm, cholesky


def ch_corr_Rayleigh_by_cov(x, Cov_h, AWGN_std, device):
    x = x.to(device)
    bs = x.size(0)
    n = x.size(1)
    assert Cov_h.shape == (n, n)

    # generate normally distributed matrices
    h = torch.normal(mean=0., std=1., size=(bs, n), device=device, dtype=torch.cfloat)
    h = h[:, :, None]
    # Compute a matrix A such that A*A^H = Cov.

    # Compute the Cholesky decomposition.
    A = torch.linalg.cholesky(Cov_h, upper=False)  # torch.cholesky(Cov, upper=False)
    # print(torch.mm(A, A.conj().T))
    # print(Cov_h)

    A = A[None, :, :].repeat(bs, 1, 1).to(device)

    # Correlated channel matrices h = Ah,  A[n,n] h[n, 1]
    h = torch.bmm(A, h)
    h = h.view(bs, n)

    y_c = h * x
    #    x_r = torch.view_as_real(h)
    #    print(torch.std(x_r[:,:,0]) ** 2)
    #    print(torch.std(x_r[:,:,1]) ** 2)

    y_c = y_c + torch.normal(mean=0, std=2 **0.5 * AWGN_std, size=x.size(), dtype=torch.cfloat).to(device)

    return y_c, h


def ch_Rayleigh(ch_input, h_std, AWGN_std, device):
    # y = hx + z
    # The diagonal element of H follows a Rayleigh distribution.

    ch_input = ch_input.to(device)
    h = torch.normal(mean=0, std=h_std, size=ch_input.size(), dtype=torch.cfloat).to(device)  # fading complex Gaussian
    z = torch.normal(mean=0, std=2 ** 0.5 * AWGN_std, size=ch_input.size(), dtype=torch.cfloat).to(device)  # AWGN

    ch_output = torch.zeros_like(ch_input).to(device)
    ch_output = torch.mul(h, ch_input) + z

    return ch_output, h

def Toeplitz(h, delay, n):
    H = torch.zeros([n, n], dtype = torch.cfloat)
    ## define the nonzero elements of H by using h and delay. 
    return H

def ISAC_channel(ch_input, h_std, delay, AWGN_std, device):
    ch_input = ch_input.to(device)
    n = ch_input.size[1]
    h = torch.normal(mean=0, std=h_std, size=ch_input.size[0], dtype=torch.cfloat, device=device)  # fading complex Gaussian
    H = Toeplitz(h, delay, n).to(device)
    z = torch.normal(mean=0, std=2**0.5*AWGN_std, size=ch_input.size(), dtype=torch.cfloat, device=device)  # AWGN

    ch_output = H * ch_input + z
    
    return ch_output



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