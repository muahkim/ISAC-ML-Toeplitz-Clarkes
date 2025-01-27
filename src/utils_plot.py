import torch
import torch.nn.functional as F
import numpy as np
import sys
import math

assert sys.version_info >= (3, 5)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

from sklearn.manifold import TSNE
from matplotlib.lines import Line2D


from channel_models import *

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = "serif"
mpl.rcParams['mathtext.fontset'] = "dejavuserif"  # cm, stix, stixsans, dejavuserif, dejavusans.
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
np.random.seed(42)


def myplot(x1, x2, s, max_amp, bins=1000):
    # This is a filter to draw a heatmap.        
    heatmap, xedges, yedges = np.histogram2d(x1, x2, bins=bins, range=[[-max_amp, max_amp], [-max_amp, max_amp]])
    heatmap[heatmap <1 ] -= 30
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def Constellation_Enc(enc, M, path=None, title=None, max_amp=None, is_emb=True):
    # Show the constellation of the learned encoder
    labels = torch.arange(0, M, dtype=int)
    if is_emb:
        m = labels.squeeze()
    else:
        m = F.one_hot(labels, num_classes=M)
        m = m.float().reshape(-1, M)
    x = enc(m).detach().cpu()
    n = x.shape[1]
    x = x.view(-1, n // 2, 2).numpy()
    # x = np.reshape(x, [-1,n//2,2])
    if max_amp == None:
        max_re = np.sqrt(np.max(x[:, :, 0] ** 2)) * 1.2
        max_im = np.sqrt(np.max(x[:, :, 1] ** 2)) * 1.2
        if max_re > max_im:
            max_amp = max_re
        else:
            max_amp = max_im
    else:
        max_re, max_im = max_amp, max_amp

    # print("Constellation of learned encoder", flush=True)
    if n // 2 == 1:
        fig = plt.figure(figsize=(4, 4))
        for j in range(M):
            plt.scatter(x[j, 0, 0], x[j, 0, 1], s=size, )
            plt.xlabel('Re($x$)')
            plt.ylabel('Im($x$)')
    else:
        fig, axes = plt.subplots(1, n // 2, figsize=((n // 2) * 2.8, 3))
        for i in range(n // 2):
            for j in range(M):
                axes[i].scatter(x[j, i, 0], x[j, i, 1], )  # , s=size)
            # axes[i].set_xlabel('Re($x_' + str(i+1) + '$)')
            # axes[i].set_ylabel('Im($x_'+ str(i+1) +')$')
            axes[i].set_title(r'$x_{' + str(i + 1) + '}$', fontsize=14)  # ("#"+ str(i+1))
            axes[i].set_xlim([-max_re, max_re])
            axes[i].set_ylim([-max_im, max_im])
    fig.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
    plt.show()

    return


def Constellation_Enc_both(enc, M, is_heatmap=False, s=5, size=1, bins=1000, path=None, title=None, max_amp=None,
                           colormap=cm.coolwarm):
    # Show the constellation of the learned encoder and the heatmap
    # Arrays for plotting
    labels = torch.arange(0, M, dtype=int)
    m = F.one_hot(labels, num_classes=M)
    m = m.float().reshape(-1, M)
    x = enc(m).detach().cpu()
    n = x.shape[1]
    x = x.view(-1, n // 2, 2).numpy()
    # x = np.reshape(x, [-1,n//2,2])
    if max_amp == None:
        max_re = np.sqrt(np.max(x[:, :, 0] ** 2)) * 1.2
        max_im = np.sqrt(np.max(x[:, :, 1] ** 2)) * 1.2
        if max_re > max_im:
            max_amp = max_re
        else:
            max_amp = max_im
    else:
        max_re, max_im = max_amp, max_amp

    # print("Constellation of learned encoder", flush=True)
    if n // 2 == 1:
        fig = plt.figure(figsize=(4, 4))
        for j in range(M):
            plt.scatter(x[j, 0, 0], x[j, 0, 1], s=size, )
            plt.xlabel('Re($x$)')
            plt.ylabel('Im($x$)')
    else:
        fig, axes = plt.subplots(1, n // 2 + 1, figsize=((n // 2 + 1) * 2.8, 3))
        for i in range(n // 2):
            for j in range(M):
                axes[i].scatter(x[j, i, 0], x[j, i, 1], )  # , s=size)
            # axes[i].set_xlabel('Re($x_' + str(i+1) + '$)')
            # axes[i].set_ylabel('Im($x_'+ str(i+1) +')$')
            axes[i].set_title(r'$x_{' + str(i + 1) + '}$', fontsize=14)  # ("#"+ str(i+1))
            axes[i].set_xlim([-max_re, max_re])
            axes[i].set_ylim([-max_im, max_im])

        if is_heatmap:
            x = np.reshape(x, (-1, 2))
            img, extent = myplot(x[:, 0], x[:, 1], s, max_amp, bins)
            max = np.max(img)
            im = axes[-1].imshow(img, extent=extent, aspect='auto', origin='lower', cmap=colormap)
            axes[-1].figure.colorbar(im, fraction=0.046, pad=0.04)
        else:
            axes[-1].scatter(x[:, :, 0], x[:, :, 1], s=size, )
        # axes[-1].set_xlabel('Re($x_i$)')
        # axes[-1].set_ylabel('Im($x_i$)')
        axes[-1].set_title("All in one", fontsize=16)  # , fontdict=font1)
        axes[-1].set_xlim([-max_re, max_re])
        axes[-1].set_ylim([-max_im, max_im])

    fig.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
    plt.show()

    return


def Constellation_Enc_m(enc, M, m=0, path=None, title=None, max_amp=None):
    # Show the constellation of the learned encoder
    mpl.rcParams.update(mpl.rcParamsDefault)
    # Arrays for plotting

    labels = torch.tensor(m, dtype=int)  # torch.arange(0,M, dtype=int)
    m = F.one_hot(labels, num_classes=M)
    m = m.float().reshape(-1, M)
    x = enc(m).detach().cpu()
    n = x.shape[1]
    x = x.view(-1, n // 2, 2).numpy()
    # x = np.reshape(x, [-1,n//2,2])
    if max_amp == None:
        max_re = np.sqrt(np.max(x[:, :, 0] ** 2))
        max_im = np.sqrt(np.max(x[:, :, 1] ** 2))
    else:
        max_re, max_im = max_amp, max_amp

    print("Constellation of learned encoder", flush=True)
    if n // 2 == 1:
        fig = plt.figure(figsize=(4, 4))
        for j in range(M):
            plt.scatter(x[j, 0, 0], x[j, 0, 1])
        plt.xlabel('$Re(x)$')
        plt.ylabel('$Im(x)$')
    #    axes[i].set_xlim([-1, 1])
    else:
        fig, axes = plt.subplots(1, n // 2, figsize=(n // 2 * 3, 3))
        for i in range(n // 2):
            axes[i].scatter(x[0, i, 0], x[0, i, 1])
            axes[i].set_xlabel('$Re(x_' + str(i + 1) + ')$')
            axes[i].set_ylabel('$Im(x_' + str(i + 1) + ')$')
            axes[i].set_title("channel #" + str(i + 1))
            axes[i].set_xlim([-max_re, max_re])
            axes[i].set_ylim([-max_im, max_im])
    if title is not None:
        fig.tight_layout()
        fig.suptitle(title)
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
    plt.show()

    return


def Constellation_Enc_all(enc, M, is_emb=True, is_heatmap=True, s=10, bins=200, path=None, title=None, max_amp=None,
                          colormap=cm.coolwarm):
    # Show the constellation of the learned encoder by a heat map across over n dimensions.
    mpl.rcParams.update(mpl.rcParamsDefault)

    labels = torch.arange(0, M, dtype=int)
    if is_emb:
        m = labels.squeeze()
    else:
        m = F.one_hot(labels, num_classes=M)
        m = m.float().reshape(-1, M)
    x = enc(m).detach()
    bs = x.shape[0]
    n_r = x.shape[1]

    n = n_r // 2
    x = x.view(bs * n, 2).detach().cpu().numpy()
    if max_amp == None:
        max_amp = np.sqrt(np.max(x ** 2)) * 1.2

    fig, ax = plt.subplots(figsize=(3, 3))

    print("Constellation of learned encoder by a heat map", flush=True)

    if is_heatmap:
        img, extent = myplot(x[:, 0], x[:, 1], s, max_amp, bins=bins)
        im = ax.imshow(img, extent=extent, aspect='auto', origin='lower', cmap=colormap)
        cbar = ax.figure.colorbar(im, fraction=0.05, pad=0.04)
        #cbar = fig.colorbar(im)
        cbar.set_ticks([])
    else:
        ax.scatter(x[:, 0], x[:, 1])

    ax.set_xlabel('$Re$')
    ax.set_ylabel('$Im$')
    if title is not None:
        ax.set_title(title)
    ax.set_xlim([-max_amp, max_amp])
    ax.set_ylim([-max_amp, max_amp])
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
    plt.show()
    return


def Constellation_Enc_all_m(enc, M, is_heatmap=True, m=0, s=10, path=None, title=None, max_amp=None):
    # Show the constellation of the learned encoder by a heat map across over n dimensions.
    mpl.rcParams.update(mpl.rcParamsDefault)

    labels = torch.tensor([m])  # torch.arange(0, M, dtype=int)
    m = F.one_hot(labels, num_classes=M)
    m = m.float().reshape(-1, M)
    x = enc(m).detach()
    bs = x.shape[0]
    n_r = x.shape[1]

    n = n_r // 2
    x = x.view(bs * n, 2).detach().cpu().numpy()

    if max_amp == None:
        max_amp = np.sqrt(np.max(x ** 2)) * 1.2

    print("Constellation of learned encoder by a heat map", flush=True)

    def myplot(x1, x2, s, bins=1000):

        heatmap, xedges, yedges = np.histogram2d(x1, x2, bins=bins, range=[[-max_amp, max_amp], [-max_amp, max_amp]])
        heatmap = gaussian_filter(heatmap, sigma=s)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        return heatmap.T, extent

    fig, ax = plt.subplots(figsize=(3, 3))

    if is_heatmap:
        img, extent = myplot(x[:, 0], x[:, 1], s)
        im = ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
        ax.figure.colorbar(im)
    else:
        ax.scatter(x[:, 0], x[:, 1], marker='o', s=5)

    ax.set_xlabel('$Re$')
    ax.set_ylabel('$Im$')
    if title is not None:
        ax.set_title(title)
    ax.set_xlim([-max_amp, max_amp])
    ax.set_ylim([-max_amp, max_amp])
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
    plt.grid(True)
    plt.show()
    return


def Constellation_InOut(enc, M, cov, noise_std, dev, num_samples=500, bs=100, path=None):
    # Show the constellation of the learned channel output with a comparison to the channel model's outputs.
    mpl.rcParams.update(mpl.rcParamsDefault)
    colors = 100 // M * np.arange(M) / 100 + 0.07  # np.arange(1, 100, 100//M, dtype=int)[:M]
    cmap = plt.cm.get_cmap('gist_ncar')
    colors_rgba = cmap(colors) * 0.8
    # Arrays for plotting
    labels = torch.arange(0, M, dtype=int)
    m = F.one_hot(labels, num_classes=M)
    m = m.float().reshape(-1, M)
    x = enc(m)  # codeword 2n-dimensional real, n-dimensional complex
    n = x.shape[1] // 2

    x_r = x.repeat(num_samples, 1)
    x_r = x_r.view(-1, n, 2)
    x_c = torch.view_as_complex(x_r)

    y = torch.empty(num_samples * M, n, dtype=torch.cfloat).to(dev)
    h_c = torch.empty(num_samples * M, n, dtype=torch.cfloat).to(dev)
    x_c_batched = torch.split(x_c, bs)
    for count, x_c_b in enumerate(x_c_batched):
        y[count * bs: (count + 1) * bs, :], h_c[count * bs: (count + 1) * bs, :] = ch_corr_Rayleigh_by_cov(x_c_b, cov,
                                                                                                           noise_std,
                                                                                                           dev)
    # move tensors to cpu and convert to numpy array for plotting.

    y = torch.view_as_real(y)
    max_re = torch.sqrt(torch.max(y[:, :, 0] ** 2)).item()
    max_im = torch.sqrt(torch.max(y[:, :, 1] ** 2)).item()

    y = y.detach().cpu().numpy()
    x = x_r.detach().cpu().numpy()

    print("Constellation of channel output", flush=True)
    if n == 1:
        for j in range(M):
            fig = plt.figure(figsize=(4, 4))
            plt.scatter(y[num_samples * j: num_samples * (j + 1), 0],
                        y[num_samples * j: num_samples * (j + 1), 1], marker='x', s=1)
            plt.scatter(x_r[j, 0], x_r[j, 1], color=colors_rgba[j], marker='o')
            plt.xlim([-max_re, max_re])
            plt.ylim([-max_im, max_im])
            plt.xlabel('$Re(x)$')
            plt.ylabel('$Im(x)$')
    else:
        for j in range(M):
            fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
            for i in range(n):
                # for j in range(M):
                axes[i].scatter(y[num_samples * j: num_samples * (j + 1), i, 0],
                                y[num_samples * j: num_samples * (j + 1), i, 1], marker='x', s=1)
                axes[i].scatter(x[j, i, 0], x[j, i, 1], marker='o')
                axes[i].set_xlabel('$Re(x_' + str(i + 1) + ')$')
                axes[i].set_ylabel('$Im(x_' + str(i + 1) + ')$')
                axes[i].set_title("channel #" + str(i + 1) + " message #" + str(j + 1))

                axes[i].set_xlim([-max_re, max_re])
                axes[i].set_ylim([-max_im, max_im])
            if path is not None:
                fig.savefig(path + '_' + str(j + 1) + '.pdf', bbox_inches='tight')
    plt.show()

    return


def tSNE_plot(encoder, M, device):

    num_samples = 100

    m_sample = torch.eye(M)
    m = m_sample.repeat_interleave(num_samples, dim=0).to(device)

    x = encoder(m)
    samples = x.detach().cpu().numpy()  # channel_model(x, noise_std, device).detach().cpu().numpy()
    samples_emb = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(samples)

    '''
    x_ = x.view(-1, n, 2)  # each sample is a n x 2 tensor.
    x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector
    # Communication and Radar Channel
    y_c, _, h_c, _, _ = channels_by_cov_2mat_nb(x_, cov, noise_std,
                                                noise_std, device)
    y_c_r = torch.view_as_real(y_c).reshape(-1,2 * n)
    samples_y = y_c_r.detach().cpu().numpy() #channel_model(x, noise_std, device).detach().cpu().numpy() 
    samples_emb = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(samples_y)
    '''



    unfilled_markers = [m for m, func in Line2D.markers.items()
                        if func != 'nothing' and m not in Line2D.filled_markers]

    colors = 100 // M * np.arange(M) / 100 + 0.07  # np.arange(1, 100, 100//M, dtype=int)[:M]
    cmap = plt.cm.get_cmap('gist_ncar')
    colors_rgba = cmap(colors) * 0.8

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    for i in range(M):
        axs[0].scatter(samples[i * num_samples:(i + 1) * num_samples, 0],
                       samples[i * num_samples:(i + 1) * num_samples, 1]
                       , color=colors_rgba[i], s=10, marker=unfilled_markers[i], label="m=" + str(i + 1))

    # for i in range(M):
    #    axs[1].scatter(samples_y[i*num_samples:(i+1)*num_samples,0], samples_y[i*num_samples:(i+1)*num_samples,1]
    #                , color =colors_rgba[i], s=3, marker=unfilled_markers[i], label="m="+str(i+1))

    # plt.legend()
    axs[0].set_ylim(-1.8, 1.8)
    axs[0].set_xlim(-1.8, 1.8)
    # ax.set_xticks([],[])
    # ax.set_title('Channel Output')
    plt.savefig('fig/t-SNR-cd-3-encoders.pdf', bbox_inches='tight')
    plt.show()

def load_and_plot_enc(gamma_list, f_dopp_norm, tf_base):
    for gamma in gamma_list:
        ft_tmp = ft_base + str(f'_f{f_dopp_norm:.3f}_gamma{gamma:.1f}') 
        ## Load the encoder
        encoder = models.Encoder_1h_n2_deeper_ee(M, n_r).to(device)
        encoder.load_state_dict(torch.load('state_dict/encoder'+ft_tmp+'.pt'))
        encoder.eval()
        ## Plotting the Constellation
        Constellation_Enc(encoder, M ,path='fig/Constellation'+ft_tmp+'.pdf', max_amp=4) 
        Constellation_Enc_all(encoder, M, is_heatmap=True, s=2, bins=200, path='fig/ConstellationA'+ft_tmp+'.pdf', title=r'$\gamma=$'+str(gamma), max_amp=4)

