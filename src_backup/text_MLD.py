import torch

from src.utils import complex_multivariate_Gaussian_loglikelihood

# Set random seed for reproducibility
torch.manual_seed(0)

# Example parameters
batch_size = 10
M = 5
n = 4  # Dimension

# Random complex input x
x_real = torch.randn(batch_size, M, n)
x_imag = torch.randn(batch_size, M, n)
x = torch.complex(x_real, x_imag)

# Random valid covariance matrices (Hermitian positive definite)
cov = torch.randn(M, n, n, dtype=torch.cfloat)
cov = cov @ cov.mH  # Make it Hermitian and positive definite

# Compute log-likelihood
log_likelihood = complex_multivariate_Gaussian_loglikelihood(x, cov)

# Check if all values are ≤ 0
print("Log-likelihood values:\n", log_likelihood)
print("All values ≤ 0:", torch.all(log_likelihood <= 0).item())
