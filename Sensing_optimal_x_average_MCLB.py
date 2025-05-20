import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
N = 300               # Blocklength: Number of symbols
T = 1.0              # Sampling interval
N0 = 1.0             # Noise spectral density
device = "cuda" if torch.cuda.is_available() else "cpu"

# Learnable signal vector x
x_raw = nn.Parameter(torch.randn(2*N, device=device))
#x_raw = nn.Parameter(torch.randn(N, dtype=torch.cfloat))

def get_power_normalized_x(x_raw, P):
    norm = torch.sum(torch.abs(x_raw)**2)
    return x_raw * (P / norm).sqrt()

# Define derivative of sinc using PyTorch
def sinc_derivative(argument):
    pi_arg = torch.pi * argument
    numerator = pi_arg * torch.cos(pi_arg) - torch.sin(pi_arg)
    denominator = torch.pi * argument ** 2
    # avoid division by zero
    derivative = torch.where(argument == 0, torch.tensor(0.0, device=device), numerator / denominator)
    return derivative

# Compute MCLB over a batch of tau values
def compute_mclb(x_raw, tau_vals, T, N0):
    N = x_raw.shape[0] // 2
    n = torch.arange(1, N + 1, device=device).view(1, -1)
    m = torch.arange(1, N + 1, device=device).view(-1, 1)

    x = get_power_normalized_x(x_raw, N)
    tau_vals = tau_vals.view(-1, 1, 1)  # [batch, 1, 1]
    argument = m - n - tau_vals / T     # [batch, N, N]
    derivative = sinc_derivative(argument) * (-1.0 / T)  # [batch, N, N]
    x = x.view(N, 2)
    x_c = torch.view_as_complex(x)
    x_expanded = x_c.view(1, 1, -1)       # [1, 1, N]
    product = derivative * x_expanded  # [batch, N, N]
    sum_inner = product.sum(dim=2)     # sum over n -> [batch, N]
    result = (sum_inner.abs() ** 2).sum(dim=1)  # sum over m -> [batch]
    mclb = (2.0 / N0 * result) ** -1    # final MCLB
    return mclb

# Optimization setup
tau_vals = torch.linspace(0, 2 * T, 100, device=device)
optimizer = optim.Adam([x_raw], lr=0.01)
num_epochs = 2000
loss_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    mclb_vals = compute_mclb(x_raw, tau_vals, T, N0)
    loss = torch.mean(mclb_vals)  # approximate integral
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4e}")

# Plotting results
x_opt = x_raw.detach()
x_opt = get_power_normalized_x(x_opt, N)
mclb_vals_opt = compute_mclb(x_opt, tau_vals, T, N0).detach().cpu().numpy()
x_opt = x_opt.detach().cpu().numpy()

plt.figure(figsize=(22, 5))
plt.subplot(1, 4, 1)
plt.plot(tau_vals.cpu().numpy(), mclb_vals_opt)
plt.title("Optimized MCLB vs $\\tau$")
plt.xlabel("$\\tau$")
plt.ylabel("MCLB")

plt.subplot(1, 4, 2)
plt.plot(loss_history)
plt.title("Loss over Epochs (Integral of MCLB)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()

x_opt = np.reshape(x_opt, (N, 2))
plt.subplot(1, 4, 3)
plt.plot(np.arange(1, N+1), x_opt[:, 0], c="red", label="real")
plt.plot(np.arange(1, N+1), x_opt[:, 1], c="blue", label="imag")
plt.plot(np.arange(1, N+1), np.sum(x_opt**2, axis=1)**0.5, c="green", label="magnitude")
plt.plot(np.arange(1, N+1), np.arctan(x_opt[:,1]/x_opt[:,0]), c="black", label="phase")
plt.legend()
plt.title("Optimal X")

plt.subplot(1, 4, 4)
#plt.scatter(x_opt[:, 0], x_opt[:, 1])
plt.hist2d(x_opt[:,0], x_opt[:,1], bins=100)
plt.title("Optimal X")
#plt.xlim(-1.2, 1.2)
#plt.ylim(-1.2, 1.2)

plt.show()
