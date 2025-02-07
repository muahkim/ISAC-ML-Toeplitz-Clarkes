
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F

import utils
from channel_models import *
import utils_standard_codec
from utils_plot import Constellation_Enc


class TrainerConfig:
    # optimization parameters
    M = 16
    n = 7
    n_r = n*2
    rate = math.log2(M) / n
    cov = torch.eye(n, n, dtype=torch.cfloat)
    f_dopp_norm = 0.1
    noise_std_c = 1
    P = 10
    T = 1e-9
    tau_max = 10*T
    noise_std_r = 1

    dataset_size = 10 ** 6
    max_epochs = 10
    batch_size = 512
    num_workers = 0  # for DataLoader
    lr = 1e-3
    loss_ratio_c_to_r = 1.
    only_groundtruth = False

    auto_save = True
    file_tag = "test"
    path = os.getcwd()
    path = path + '/state_dict'
    #correlation = 0.0

    is_conv = False
    is_emb = False

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        # self.model = torch.nn.DataParallel(self.model).to(self.device)
    else:
        device = "cpu"

    writer = None
    args = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.cov = utils_standard_codec.Rayleigh_covariance_generator(self.n, self.f_dopp_norm, dtype_cov=torch.cfloat)
        plt.imshow(self.cov.real, cmap='hot')  # , interpolation='nearest')
        plt.title(r'Covariance, $\bar{f}_D=' + str(self.f_dopp_norm) + '$')
        plt.colorbar()
        plt.show()

        # Generate a random number for the folder name
        #folder_name = str(np.random.randint(100000, 999999))  # Adjust range as needed
        self.timestamp = datetime.now().strftime("%m%d%H%M%S")

        # Define the folder path (creates in the current directory)
        self.path_state_dict = os.path.join(os.getcwd() + '/state_dict', self.file_tag[1:] + '_' + self.timestamp)
        self.path_figures = os.path.join(os.getcwd() + '/figures', self.file_tag[1:] + '_' + self.timestamp)

        # Create the directory if it doesn't exist
        os.makedirs(self.path_state_dict, exist_ok=True)
        os.makedirs(self.path_figures , exist_ok=True)
        print(f"Folder created: {self.path_state_dict}")
        print(f"Folder created: {self.path_figures }")

def Likelihood_corr_fading_ncd(y, table, noise_std):
    dist = torch.squeeze(torch.cdist(y, table, p=2) ** 2 / (noise_std ** 2))  # (bs, 1, M)
    prob = torch.exp(-dist)
    prob = prob / torch.sum(prob, dim=1)[:, None]  # normalization of the probability
    return prob


def forward_pass(encoder, channel, decoder, labels, config):
    # This is for a neural decoder.
    # Forward Pass:
    # Encoding
    m = F.one_hot(labels.long(), num_classes=config.M)
    m = m.float().reshape(-1, config.M)
    x = encoder(m)  # each sample is a 2n-dimensional real vector
    x_ = x.view(-1, config.n, 2)  # each sample is a n x 2 tensor.
    x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector
    # Communication and Radar Channel
    y_c, h_c = channel(x_, config.cov, config.noise_std_c, config.device)
    y_r = channel(x_, config.noise_std_r, config.device)

    # Decodingtmp[:, None, :])
    y_c_r = torch.view_as_real(y_c).view(-1, 2 * config.n)  # each sample is a n x 2 tensor.
    h_c_r = torch.view_as_real(h_c).view(-1, 2 * config.n)
    m_d = decoder(y_c_r, h_c_r)  # coherent detection: full access to h_c.

    return m_d, x_, y_c, y_r, h_c


class Trainer:
    def __init__(self, enc, config, binary=False):
        self.enc = enc

        if binary:
            self.train_dataset = 2 *( np.random.randint( 2, size=(config.dataset_size, config.k, 1) ) - 0.5)
        else: # one hot encoded data
            self.train_dataset = np.random.randint(config.M, size=config.dataset_size)

        self.config = config
        self.binary = binary
        self.is_conv = config.is_conv
        self.is_emb = config.is_emb

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        self.channel_c = partial(ch_corr_Rayleigh_by_cov, cov=config.cov,
                                 noise_std=config.noise_std_c, device=self.device, decomposition_method='eigenvector')
        self.channel_s = partial(Toeplitz_channel, P=config.P, T=config.T,
                                 N0= 2 * config.noise_std_r ** 2, tau_max=config.tau_max)
        self.MLDetector = partial(utils.ML_det_Rayleigh_ncd, cov_h=config.cov, std_z=config.noise_std_c)

    def train(self):
        enc, config, device = self.enc, self.config, self.device
        loss_CE = nn.CrossEntropyLoss()


        optimizer_enc = torch.optim.NAdam(enc.parameters(), lr=config.lr)

        M = config.M
        k = int(math.log2(M))
        n = config.n
        n_r = config.n_r

        BER_ep, SER_ep = [], []

        def run_epoch():
            loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=False, batch_size=config.batch_size)
            loss_batch, loss_batch_c, loss_batch_r, batch_BER, batch_SER = [], [], [], [], []

            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

            for it, m in pbar:
                bs = m.shape[0]
                if self.binary:
                    m = m.float().to(device)
                else: # one-hot encoded
                    labels = m.reshape((-1, 1)).to(device)
                    if self.is_conv:
                        m_tmp = torch.eye(M).to(device)
                        table = torch.view_as_complex(enc(m_tmp[:, None, :]).view(M, n, 2)).to(device).detach()
                    elif self.is_emb:
                        m_tmp = torch.arange(M).to(device)
                        table = torch.view_as_complex(enc(m_tmp).view(M, n, 2)).to(device).detach()
                    else:
                        m_tmp = torch.eye(M).to(device)
                        table = torch.view_as_complex(enc(m_tmp).view(M, n, 2)).to(device).detach()

                    # Forward Pass:
                    m = F.one_hot(labels.long(), num_classes=M)
                    m = m.float().reshape(-1, M)

                enc.zero_grad()

                if self.is_conv:
                    x = enc(m[:, None, :])
                elif self.is_emb:
                    x = enc(labels.squeeze())
                else:
                    x = enc(m)  # each sample is a 2n-dimensional real vector

                x_ = x.view(-1, n, 2)  # each sample is a n x 2 tensor.
                x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

                # Communication Channel
                y_c, h_c = self.channel_c(x_)

                # Maximum-Likelihood Decoding
                if config.only_groundtruth:
                    loglikelihood, _ = self.MLDetector(y_c, x_, only_groundtruth = config.only_groundtruth) # likelihood of the correct message.
                    loss_c = - loglikelihood # equivalent to the binary cross entropy.
                else:
                    table = table.view(M, n)
                    loglikelihood, m_hat = self.MLDetector(y_c, table, only_groundtruth = config.only_groundtruth)

                    # Backward pass for decoder
                    loss_c = loss_CE(loglikelihood, labels.long().squeeze(1))
                loss_c = loss_c.mean()


                # Sensing
                y_r, h_s_a, tau_s_a = self.channel_s(x_)

                # Backward pass for estimator
                loss_r = utils.CRLB(2 * config.noise_std_r ** 2, h_s_a, x_, tau_s_a, config.T)

                loss = config.loss_ratio_c_to_r * loss_c + (
                        1 - config.loss_ratio_c_to_r) * loss_r  # normalization and good scaling needed.
                loss.backward()

                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1)
                optimizer_enc.step()

                loss_batch_c.append(loss_c.item())
                loss_batch_r.append(loss_r.item())
                loss_batch.append(loss.item())

                '''
                bool_sym = torch.ne(torch.squeeze(labels).int(), torch.squeeze(m_hat).int())
                SER_tmp = torch.sum(bool_sym) / m_hat.shape[0]

                labels_bin = torch.squeeze(utils.dec2bin(labels.int(), k))
                m_d_bin = utils.dec2bin(torch.squeeze(m_hat).int(), k)

                ne = torch.ne(labels_bin, m_d_bin)
                BER_tmp = torch.mean(ne.float())

                batch_BER.append(BER_tmp.item())
                batch_SER.append(SER_tmp.item())
                
                if it % 100 == 0:
                    pbar.set_description(
                        f"epoch {epoch + 1}: loss c, loss r, sum: {np.mean(loss_batch_c):.2e}, {np.mean(loss_batch_r):.2e}, {np.mean(loss_batch):.2e}")

                '''

                if it % 100 == 0:
                    pbar.set_description(
                        f"epoch {epoch + 1}: loss c, loss r, sum: {np.mean(loss_batch_c):.2e}, {np.mean(loss_batch_r):.2e}, {np.mean(loss_batch):.2e}")
            #print(f"epoch {epoch + 1}: BER {np.mean(batch_BER):.2e} FER {np.mean(batch_SER):.2e} CRLB:{np.mean(loss_batch_r):.2e}")


            if config.auto_save:
                torch.save(enc.state_dict(), config.path_state_dict + '/encoder' + config.file_tag + f'-{epoch+1}.pt')
            if config.n <= 10:
                Constellation_Enc(enc, M, path=f"{config.path_figures}/Constellation{config.file_tag}-{epoch+1}.pdf", title=f"{epoch+1} epochs",is_emb=False)

            return #np.mean(batch_BER), np.mean(batch_SER)

        for epoch in range(config.max_epochs):
            run_epoch()
            #BER_ep.append(ber_tmp)
            #SER_ep.append(ser_tmp)

        return None, None # BER_ep, SER_ep

    def train_ro(self, loss_type='MSE'): # training for radar-only, with the MMSE estimator.
        enc, config, device = self.enc, self.config, self.device
        cov_r = config.cov_r.to(device)
        optimizer_enc = torch.optim.NAdam(enc.parameters(), lr=config.lr)

        M = config.M
        n = config.n

        BER_ep, SER_ep = [], []

        def run_epoch():
            loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=False, batch_size=config.batch_size)
            loss_batch, loss_batch_c, loss_batch_r, batch_BER, batch_SER = [], [], [], [], []

            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

            for it, m in pbar:
                bs = m.size(0)
                labels = m.reshape((-1, 1)).to(device)
                m_tmp = torch.eye(M).to(device)
                table = torch.view_as_complex(enc(m_tmp).view(M, n, 2)).to(device).detach()

                enc.zero_grad()

                # Forward Pass:
                m = F.one_hot(labels.long(), num_classes=M)
                m = m.float().reshape(-1, M)

                x = enc(m)  # each sample is a 2n-dimensional real vector
                x_ = x.view(-1, n, 2)  # each sample is a n x 2 tensor.
                x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

                # Sensing
                y_r, h_r = ch_corr_Rayleigh_by_cov(x_, cov_r, config.noise_std_r, device)
                h_r_hat = utils.MMSE_H_estimator_corr(x_, y_r, config.noise_std_r, cov_r)

                # Backward pass for estimator
                assert h_r.shape == h_r_hat.shape
                if loss_type == 'MSE':
                    loss_r = utils.MSE_c(h_r, h_r_hat)
                elif loss_type == 'RSE':
                    loss_r = utils.RSE_c(h_r, h_r_hat)
                else:
                    assert False
                loss_r.backward()

                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.)
                optimizer_enc.step()
                loss_batch_r.append(loss_r.item())

                if it % 100 == 0:
                    pbar.set_description(
                        f"epoch {epoch + 1}: MSE: {np.mean(loss_batch_r):.2e}")

            if config.auto_save:
                torch.save(enc.state_dict(), config.path + '/encoder_ro' + config.file_tag + '.pt')

            return

        for epoch in range(config.max_epochs):
            run_epoch()

        return

    def train_co(self): # training for communication-only, with the ML detection.
        enc, config, device = self.enc, self.config, self.device
        cov = config.cov.to(device)
        loss_CE = nn.CrossEntropyLoss()
        optimizer_enc = torch.optim.NAdam(enc.parameters(), lr=config.lr)

        M = config.M
        k = int(math.log2(M))
        n = config.n

        BER_ep, SER_ep = [], []

        def run_epoch():
            loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=False, batch_size=config.batch_size)
            loss_batch, batch_BER, batch_SER = [], [], []

            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

            for it, m in pbar:
                bs = m.size(0)
                labels = m.reshape((-1, 1)).to(device)
                m_tmp = torch.eye(M).to(device)
                table = torch.view_as_complex(enc(m_tmp).view(M, n, 2)).to(device).detach()

                enc.zero_grad()

                # Forward Pass:
                m = F.one_hot(labels.long(), num_classes=M)
                m = m.float().reshape(-1, M)

                x = enc(m)  # each sample is a 2n-dimensional real vector
                x_ = x.view(-1, n, 2)  # each sample is a n x 2 tensor.
                x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

                # Communication and Radar Channel
                y_c, h_c = ch_corr_Rayleigh_by_cov(x_, cov, config.noise_std_c, device)

                # Maximum-Likelihood Decoding
                prob, m_hat = utils.ML_det_Rayleigh_ncd(y_c, table, cov, config.noise_std_c)

                # Backward pass for decoder
                loss_c = loss_CE(prob, labels.long().squeeze(1))
                loss_c = loss_c.mean()

                loss_c.backward()

                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.)
                optimizer_enc.step()

                loss_batch.append(loss_c.item())

                bool_sym = torch.ne(torch.squeeze(torch.Tensor(labels)).int(), torch.squeeze(m_hat).int())
                SER_tmp = torch.sum(bool_sym) / m_hat.shape[0]

                labels_bin = torch.squeeze(utils.dec2bin(labels.int(), k))
                m_d_bin = utils.dec2bin(torch.squeeze(m_hat).int(), k)

                ne = torch.ne(labels_bin, m_d_bin)
                BER_tmp = torch.mean(ne.float())

                batch_BER.append(BER_tmp.item())
                batch_SER.append(SER_tmp.item())

                if it % 100 == 0:
                    pbar.set_description(
                        f"epoch {epoch + 1}: loss: {np.mean(loss_batch):.2e}")
            print(f"epoch {epoch + 1}: BER {np.mean(batch_BER):.2e} SER {np.mean(batch_SER):.2e}")

            if config.auto_save:
                torch.save(enc.state_dict(), config.path + '/encoder_co' + config.file_tag + '.pt')

            return np.mean(batch_BER), np.mean(batch_SER)

        for epoch in range(config.max_epochs):
            ber_tmp, ser_tmp = run_epoch()
            BER_ep.append(ber_tmp)
            SER_ep.append(ser_tmp)

        return BER_ep, SER_ep
    '''
    def train_nndec_co(self): # training for communication-only, with neural decoder.
        enc, dec, config, device = self.enc, self.dec, self.config, self.device
        cov = config.cov.to(device)
        loss_CE = nn.CrossEntropyLoss()
        optimizer_codec = torch.optim.NAdam(list(enc.parameters()) + list(dec.parameters()), lr=config.lr)

        M = config.M
        n = config.n

        BER_ep, SER_ep = [], []

        def run_epoch():
            loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=False, batch_size=config.batch_size)
            loss_batch, batch_BER, batch_SER = [], [], []

            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

            for it, m in pbar:
                labels = m.reshape((-1, 1)).to(device)
                m_tmp = torch.eye(M).to(device)
                table = torch.view_as_complex(enc(m_tmp).view(M, n, 2)).to(device).detach()

                enc.zero_grad()

                # Forward Pass:
                m = F.one_hot(labels.long(), num_classes=M)
                m = m.float().reshape(-1, M)

                x = enc(m)  # each sample is a 2n-dimensional real vector
                x_ = x.view(-1, n, 2)  # each sample is a n x 2 tensor.
                x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

                # Communication and Radar Channel
                y_c, h_c = ch_corr_Rayleigh_by_cov(x_, cov, config.noise_std_c, device)

                # Decoding
                y_c_r = torch.view_as_real(y_c).view(-1, 2 * n)
                m_d = dec(y_c_r)

                loss_c = loss_CE(m_d, labels.long().squeeze(1))
                loss_c = loss_c.mean()

                # Compute the loss and the gradient
                loss_c.backward()

                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.)
                optimizer_codec.step()

                loss_batch.append(loss_c.item())

                labels = torch.squeeze(labels)

                SER_tmp, _, _ = utils.SER(labels.long(), m_d, 0.9)
                BER_tmp = utils.BER(m, m_d)

                batch_BER.append(BER_tmp.item())
                batch_SER.append(SER_tmp.item())

                if it % 100 == 0:
                    pbar.set_description(
                        f"epoch {epoch + 1}: loss: {np.mean(loss_batch):.2e}")
            print(f"epoch {epoch + 1}: BER {np.mean(batch_BER):.2e} SER {np.mean(batch_SER):.2e}")

            if config.auto_save:
                torch.save(enc.state_dict(), config.path + '/encoder_co_nndec' + config.file_tag + '.pt')
                torch.save(dec.state_dict(), config.path + '/decoder_co_nndec' + config.file_tag + '.pt')

            return np.mean(batch_BER), np.mean(batch_SER)

        for epoch in range(config.max_epochs):
            ber_tmp, ser_tmp = run_epoch()
            BER_ep.append(ber_tmp)
            SER_ep.append(ser_tmp)

        return BER_ep, SER_ep
    
    def train_nndec(self):
        enc, dec, config, device = self.enc, self.dec, self.config, self.device
        cov = config.cov.to(device)

        loss_CE = nn.CrossEntropyLoss()
        optimizer_enc = torch.optim.NAdam(enc.parameters(), lr=config.lr)
        optimizer_dec = torch.optim.NAdam(dec.parameters(), lr=config.lr)

        M = config.M
        n = config.n

        BER_ep, SER_ep = [], []

        def run_epoch():
            loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=False, batch_size=config.batch_size)
            loss_batch, loss_batch_c, loss_batch_r, batch_BER, batch_SER = [], [], [], [], []

            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

            for it, m in pbar:
                labels = m.reshape((-1, 1)).to(self.device)
                dec.zero_grad()
                enc.zero_grad()

                bs = m.size(0)

                # Forward Pass:
                # Encoding
                m = F.one_hot(labels.long(), num_classes=M)
                m = m.float().reshape(-1, M)
                x = enc(m)  # each sample is a 2n-dimensional real vector
                x_ = x.view(-1, n, 2)  # each sample is a n x 2 tensor.
                x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

                # Communication and Radar Channel
                y_c, h_c = ch_corr_Rayleigh_by_cov(x_, cov, config.noise_std_c, device)

                # channel_model(x, cov, noise_std, device)
                y_c = torch.view_as_real(y_c)  # each sample is a n x 2 tensor.
                y_c = y_c.view(-1, 2 * n)  # each sample is a 2n-dimensional real vector
                # Decoding
                m_d = dec(y_c)

                # Backward pass for decoder
                loss_c = loss_CE(m_d, labels.long().squeeze(1))
                loss_c = loss_c.mean()
                loss_c.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.)
                optimizer_dec.step()

                # Free the gradients used for back propagation.
                dec.zero_grad()
                enc.zero_grad()

                # Sensing
                y_r = ch_corr_Rayleigh_by_cov(x_, config.noise_std_r, device)
                h_r_hat = utils.MMSE_H_estimator_corr(x_, y_r, config.noise_std_r)

                # Backward pass for estimator
                assert h_r.shape == h_r_hat.shape
                loss_r = utils.MSE_c(h_r, h_r_hat)

                # Backward pass for encoder with the updated decoder and the estimator.
                m_d = dec(y_c)
                loss_c = loss_CE(m_d, labels.long().squeeze(1))
                loss_c = loss_c.mean()

                # Compute the loss and the gradient
                loss = config.loss_ratio_c_to_r * loss_c + (
                        1 - config.loss_ratio_c_to_r) * loss_r  # normalization and good scaling needed.
                loss.backward()

                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.)
                optimizer_enc.step()

                loss_batch_c.append(loss_c.item())
                loss_batch_r.append(loss_r.item())
                loss_batch.append(loss.item())
                SER_tmp, _, _ = utils.SER(labels.long(), m_d, 0.9)
                BER_tmp = utils.BER(m, m_d)

                batch_BER.append(BER_tmp.item())
                batch_SER.append(SER_tmp.item())

                if it % 100 == 0:
                    pbar.set_description(
                        f"epoch {epoch + 1}: loss c, r, sum: {np.mean(loss_batch_c):.2e}, {np.mean(loss_batch_r):.2e}, {np.mean(loss_batch):.2e}")

            print(f"epoch {epoch + 1}: BER {np.mean(batch_BER):.2e} SER {np.mean(batch_SER):.2e}")
            # Save epoch loss
            # loss_ep_1.append(np.mean(loss_batch))

            if config.auto_save:
                torch.save(enc.state_dict(), config.path + '/encoder_nndec' + config.file_tag + '.pt')
                torch.save(dec.state_dict(), config.path + '/decoder_nndec' + config.file_tag + '.pt')

            return np.mean(batch_BER), np.mean(batch_SER)

        for epoch in range(config.max_epochs):
            ber_tmp, ser_tmp = run_epoch()
            BER_ep.append(ber_tmp)
            SER_ep.append(ser_tmp)

        return BER_ep, SER_ep

    def train_nndec_j(self):
        enc, dec, config, device = self.enc, self.dec, self.config, self.device
        cov = config.cov.to(device)

        loss_CE = nn.CrossEntropyLoss()
        optimizer_enc = torch.optim.NAdam(enc.parameters(), lr=config.lr)
        optimizer_dec = torch.optim.NAdam(dec.parameters(), lr=config.lr)

        M = config.M
        n = config.n

        BER_ep, SER_ep = [], []

        def run_epoch():
            loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=False, batch_size=config.batch_size)
            loss_batch, loss_batch_c, loss_batch_r, batch_BER, batch_SER = [], [], [], [], []

            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

            for it, m in pbar:
                labels = m.reshape((-1, 1)).to(self.device)
                dec.zero_grad()
                enc.zero_grad()

                bs = m.size(0)

                # Forward Pass:
                # Encoding
                m = F.one_hot(labels.long(), num_classes=M)
                m = m.float().reshape(-1, M)
                x = enc(m)  # each sample is a 2n-dimensional real vector
                x_ = x.view(-1, n, 2)  # each sample is a n x 2 tensor.
                x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

                # Communication and Radar Channel
                y_c, h_c = ch_corr_Rayleigh_by_cov(x_, cov, config.noise_std_c, device)

                # channel_model(x, cov, noise_std, device)
                y_c = torch.view_as_real(y_c)  # each sample is a n x 2 tensor.
                y_c = y_c.view(-1, 2 * n)  # each sample is a 2n-dimensional real vector
                # Decoding
                m_d = dec(y_c)

                # Backward pass for decoder
                loss_c = loss_CE(m_d, labels.long().squeeze(1))
                loss_c = loss_c.mean()
                loss_c.backward(retain_graph=True)
                grad_c = []  # Saving the gradient for enc_c
                for param in enc.parameters():
                    grad_c.append(param.grad.clone())
                torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.)
                optimizer_dec.step()

                # Free the gradients used for back propagation.
                dec.zero_grad()
                enc.zero_grad()

                # Sensing
                y_r, h_r = ch_corr_Rayleigh_by_cov(x_, config.noise_std_r, device)
                h_r_hat = utils.MMSE_H_estimator_corr(x_, y_r, config.noise_std_r)
                # h_r_hat =  utils.MMSE_H_estimator_iid_w_priori(x_, y_r, config.noise_std_r, config.correlation, 0., 1.)

                # Backward pass for estimator
                assert h_r.shape == h_r_hat.shape
                loss_r = utils.MSE_c(h_r, h_r_hat)
                loss_r.backward()

                grad_j = []

                for count, param in enumerate(
                        enc.parameters()):  # Combining the gradient of loss_c and loss_r w.r.t. mixer.
                    grad_j.append(
                        config.loss_ratio_c_to_r * grad_c[count] + (1 - config.loss_ratio_c_to_r) * param.grad.clone())

                for param in enc.parameters():  # Updating enc_c
                    param.grad = grad_j.pop(0)
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.)
                optimizer_enc.step()

                loss = config.loss_ratio_c_to_r * loss_c + (1 - config.loss_ratio_c_to_r) * loss_r
                loss_batch_c.append(loss_c.item())
                loss_batch_r.append(loss_r.item())
                loss_batch.append(loss.item())
                SER_tmp, _, _ = utils.SER(labels.long(), m_d, 0.9)
                BER_tmp = utils.BER(m, m_d)

                batch_BER.append(BER_tmp.item())
                batch_SER.append(SER_tmp.item())

                if it % 100 == 0:
                    pbar.set_description(
                        f"epoch {epoch + 1}: loss c, r, sum: {np.mean(loss_batch_c):.2e}, {np.mean(loss_batch_r):.2e}, {np.mean(loss_batch):.2e}")

            print(f"epoch {epoch + 1}: BER {np.mean(batch_BER):.2e} SER {np.mean(batch_SER):.2e}")
            # Save epoch loss
            # loss_ep_1.append(np.mean(loss_batch))

            if config.auto_save:
                torch.save(enc.state_dict(), config.path + '/encoder_nndec' + config.file_tag + '.pt')
                torch.save(dec.state_dict(), config.path + '/decoder_nndec' + config.file_tag + '.pt')

            return np.mean(batch_BER), np.mean(batch_SER)

        for epoch in range(config.max_epochs):
            ber_tmp, ser_tmp = run_epoch()
            BER_ep.append(ber_tmp)
            SER_ep.append(ser_tmp)

        return BER_ep, SER_ep
    
    def test_comm(self, snr_range, erasure_bound):
        enc, dec, config, device = self.enc, self.dec, self.config, self.device
        M = config.M
        n = config.n
        k = int(math.log2(M))
        ser, ber = [], []
        device = config.device

        m_tmp = torch.eye(M).to(device)
        table = torch.view_as_complex(enc(m_tmp).view(M, n, 2)).detach()
        data = self.train_dataset
        batch_size = config.batch_size

        def run_epoch():
            loader = DataLoader(data, shuffle=True, pin_memory=False, batch_size=batch_size,
                                num_workers=config.num_workers)
            batch_BER = []
            batch_SER = []

            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

            for it, x in pbar:
                labels = x.reshape((-1, 1)).to(device)
                x = F.one_hot(labels.long(), num_classes=M)
                x = x.float().reshape(-1, M)

                x = enc(x).detach()
                x = torch.view_as_complex(x.view(-1, n, 2)).to(device)
                y_c, h_c = ch_corr_Rayleigh_by_cov(x, config.cov, AWGN_std_c, device)

                # Decoding
                y_c = torch.view_as_real(y_c).view(-1, 2 * n).to(device)
                m_d = dec(y_c)
                m_hat = torch.argmax(dec(y_c), dim=1)

                SER_tmp1, _, _ = utils.SER(labels.long(), m_d, 0.9)
                labels = torch.squeeze(labels)

                labels_bin = torch.squeeze(utils.dec2bin(labels.int(), k))
                m_d_bin = torch.squeeze(utils.dec2bin(m_hat.int(), k))
                ne = torch.ne(labels_bin, m_d_bin)
                BER_tmp = torch.mean(ne.float())

                batch_BER.append(BER_tmp.item())
                batch_SER.append(SER_tmp1.item())

                pbar.set_description(2,
                    f"EbN0 {snr_range[epoch]} iter {it}: avg. BER {np.mean(batch_BER):.2e} FER {np.mean(batch_SER):.2e}")

            return np.mean(batch_SER), np.mean(batch_BER)

        num_samples = self.train_dataset.size(dim=0)
        for epoch, snr in enumerate(snr_range):
            AWGN_std_c = utils.EbNo_to_noise(snr, rate=k / n)  # for a normalized signal (per symbol)
            tmp_ser = 0
            tmp_ber = 0
            it_in_while = 0
            while tmp_ser * num_samples * (
                    it_in_while) < 5000:  # To guarantee we have enough samples for the Monte-Carlo method.
                if it_in_while > 1:
                    print("The number of samples is not enough. One more epoch is running. Total # of samples used: ",
                          num_samples * it_in_while)
                tmp_ser2, tmp_ber2 = run_epoch()
                tmp_ser = (it_in_while * tmp_ser + tmp_ser2) / (
                        it_in_while + 1)  # taking average of the error probability.
                tmp_ber = (it_in_while * tmp_ber + tmp_ber2) / (it_in_while + 1)
                it_in_while += 1

            ser.append(tmp_ser)
            ber.append(tmp_ber)

        return ser, ber
    '''
    def test_comm_ML(self, model, snr_range, erasure_bound):
        # model is encoder
        config = self.config
        M = config.M
        n = config.n
        k = int(math.log2(M))
        ser, ber = [], []
        device = config.device

        if config.is_emb:
            m_tmp = torch.arange(M).to(device)
        else:
            m_tmp = torch.eye(M).to(device)

        if config.is_conv and not config.is_emb:
            table = torch.view_as_complex(model(m_tmp.unsqueeze(1)).view(M, n, 2)).to(device)  # .detach()
        else:
            table = torch.view_as_complex(model(m_tmp).view(M, n, 2)).to(device)  # .detach()

        # table = torch.view_as_complex(torch.Tensor(table).to(device)).view(M_psk, 1)
        data = self.train_dataset
        batch_size = config.batch_size

        def run_epoch():

            loader = DataLoader(data, shuffle=True, pin_memory=False, batch_size=batch_size,
                                num_workers=config.num_workers)
            batch_BER = []
            batch_SER = []

            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

            for it, x in pbar:
                labels = x.reshape((-1, 1)).to(device)
                if config.is_emb:
                    x = x.squeeze().to(device)
                else:
                    x = F.one_hot(labels.long(), num_classes=M)
                    x = x.float().reshape(-1, M)
                if config.is_conv:
                    x = model(x.unsqueeze(1)).detach()
                else:
                    x = model(x).detach()
                x = torch.view_as_complex(x.view(-1, n, 2)).to(device)
                ##y_c, _, h_c, _, _ = channels_by_cov_2mat_nb(x, config.cov, AWGN_std_c, 0, device)
                y_c, h_c = ch_corr_Rayleigh_by_cov(x, config.cov, AWGN_std_c, device)

                ## NN detection
                #                y_c = torch.flatten(y_c)
                #                h_c = torch.flatten(h_c)
                #                h_c_repeat = h_c[:, None, :].repeat(1, M, 1)
                #                m_d = [utils.NN_det(y_c[i, :], table_cd[i, :, :]).item() for i in range(y_c.shape[0])]  # label
                # Maximum-Likelihood Decoding

                prob, m_hat = utils.ML_det_Rayleigh_ncd(y_c, table, config.cov, AWGN_std_c)

                bool_sym = torch.ne(torch.Tensor(labels).int(), m_hat.int())
                SER_tmp = torch.sum(bool_sym) / m_hat.shape[0]

                labels_bin = torch.squeeze(utils.dec2bin(labels.int(), k))
                m_d_bin = torch.squeeze(utils.dec2bin(m_hat.int(), k))
                ne = torch.ne(labels_bin, m_d_bin)
                BER_tmp = torch.mean(ne.float())

                # SER_tmp, _, _ = utils.SER(labels.long(), m_d, 0.9)
                batch_BER.append(BER_tmp.item())
                batch_SER.append(SER_tmp.item())

                pbar.set_description(
                    f"EbN0 {snr_range[epoch]} iter {it}: avg. BER {np.mean(batch_BER):.2e} FER {np.mean(batch_SER):.2e}")

            return np.mean(batch_SER), np.mean(batch_BER)

        num_samples = self.train_dataset.size(dim=0)
        for epoch, snr in enumerate(snr_range):
            AWGN_std_c = utils.EbNo_to_noise(snr, rate=k / n)  # for a normalized signal (per symbol)
            tmp_ser = 0
            tmp_ber = 0
            it_in_while = 0

            while tmp_ser * num_samples * (
                    it_in_while) < 5000:  # To guarantee we have enough samples for the Monte-Carlo method.
                if it_in_while > 1:
                    print("The number of samples is not enough. One more epoch is running. Total # of samples used: ",
                          num_samples * it_in_while)
                tmp_ser2, tmp_ber2 = run_epoch()
                tmp_ser = (it_in_while * tmp_ser + tmp_ser2) / (
                        it_in_while + 1)  # taking average of the error probability.
                tmp_ber = (it_in_while * tmp_ber + tmp_ber2) / (it_in_while + 1)
                it_in_while += 1

            ser.append(tmp_ser)
            ber.append(tmp_ber)

        return ser, ber
