import torch
import scipy
import math


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
                cov[i,j] = cov[i,j] + 2**(-10)

    #cov += 1e-6 * torch.eye(N, dtype=dtype_cov)

    #cov = (cov + cov.T.conj()) / 2  # Ensure Hermitian symmetry

    return cov



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

n = 300
cov = Rayleigh_covariance_generator(n, 0.1)

x = torch.randn(10, n, device=torch.device('cpu'), dtype=torch.cfloat)
y = ch_corr_Rayleigh_by_cov(x, cov, 0, torch.device('cpu'))