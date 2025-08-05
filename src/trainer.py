import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from functools import partial

from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

import utils
from channel_models import *
import utils_standard_codec
from utils_plot import Constellation_Enc, Constellation_Enc_all, line_plots_Enc, plot_loss_over_epochs
from src.loss import customized_loss


class TrainerConfig:
    # optimization parameters
    M = 16
    n = 7
    n_r = n*2
    rate = math.log2(M) / n
    cov = torch.eye(n, n, dtype=torch.cfloat)
    f_dopp_norm = 0.1

    P = 10
    T = 1e-9
    tau_max = [2]

    generate_dataset = False
    dataset_size = 10 ** 6
    max_epochs = [10]
    batch_size = 512
    num_workers = 0  # for DataLoader
    global_optimizer = optim.NAdam
    lr_enc = [1e-3]
    loss_ratio_c_to_r = 1.
    sensing_loss_scale = 1
    training_strategy = 'joint_training'
    num_train_enc = 1
    num_train_dec = 5
    loss_c = 'decoder'
    loss_r = 'MCLB' 
    only_groundtruth = True

    LOADMODEL = False
    MLD = False
    epoch_loaded = None
    epoch_loaded_dec = None
    auto_save = True
    file_tag = "test"
    path = os.getcwd()
    path = path + '/state_dict'

    is_conv = False
    is_emb = False

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"

    writer = None
    args = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        ## File tag for distinguishing the files from this simulation.
        if self.loss_ratio_c_to_r <= 0.1 and self.loss_ratio_c_to_r > 0:
            self.file_tag = f"_k{self.k:d}N{self.n:d}_f{self.f_dopp_norm:.3f}_P{self.P}_taumax{self.tau_max}_gamma{self.loss_ratio_c_to_r:.3f}_{self.args.encoder}_{self.args.decoder}_{self.training_strategy}"
        else:
            self.file_tag = f"_k{self.k:d}N{self.n:d}_f{self.f_dopp_norm:.3f}_P{self.P}_taumax{self.tau_max}_gamma{self.loss_ratio_c_to_r:.1f}_{self.args.encoder}_{self.args.decoder}_{self.training_strategy}"


        if self.args is not None:
            self.num_train_enc = self.args.num_train_enc
            self.num_train_dec = self.args.num_train_dec

        # Sanity Check
        assert len(self.lr_enc) == len(self.max_epochs), "The length of lr and max_epochs should be the same."
        assert self.training_strategy in {'joint_training', 'iterative_training','likelihood_training'}, "training_strategy value is not supported. Available options: joint_training, iterative_training, likelihood_training, likelihood_training."

        self.cov = utils_standard_codec.Rayleigh_covariance_generator(self.n, self.f_dopp_norm, dtype_cov=torch.cfloat)
        plt.imshow(self.cov.real, cmap='hot')  # , interpolation='nearest')
        plt.title(r'Covariance, $\bar{f}_D=' + str(self.f_dopp_norm) + '$')
        plt.colorbar()
        plt.show()

        # Generate a random number for the folder name
        self.timestamp = datetime.now().strftime("%m%d%H%M%S")

        # Define the folder path (creates in the current directory)
        self.path_state_dict = os.path.join(os.getcwd() + '/state_dict', self.file_tag[1:] + '_' + self.timestamp)
        self.path_figures = os.path.join(os.getcwd() + '/figures', self.file_tag[1:] + '_' + self.timestamp)

        # Create the directory if it doesn't exist
        os.makedirs(self.path_state_dict, exist_ok=True)
        os.makedirs(self.path_figures , exist_ok=True)
        print(f"Folder created: {self.path_state_dict}")
        print(f"Folder created: {self.path_figures }")


class Trainer:
    def __init__(self, enc, config, dec=None): # decoder is optional
        self.enc = enc
        self.dec = dec

        if config.generate_dataset:
            self.train_dataset =  np.random.randint(2, size=(config.dataset_size, config.k, config.args.code_rate_k))
        else:
            self.train_dataset = None # It will be generated in every batch.

        self.config = config

        self.device = self.config.device
        self.noise_std_start = utils.SNR_to_noise(config.args.snr_train_start)
        self.noise_std_end = utils.SNR_to_noise(config.args.snr_train_end)
        self.channel_c = partial(ch_corr_Rayleigh_by_cov, cov=config.cov,
                                  device=self.device, decomposition_method='eigenvector')
        self.channel_s = partial(Toeplitz_channel, P=config.P, T=config.T,
                                  tau_max=config.tau_max, without_output=True)
        self.MLDetector = partial(utils.ML_det_Rayleigh_ncd, cov_h=config.cov)


    def save(self, milestone=None, model_to_save=None):
        if model_to_save in {'enc', 'both'}:
            assert milestone is not None, "Must specify milestone for saving"
            path_save_enc = self.config.path_state_dict + '/encoder' + self.config.file_tag + f'-{milestone}.pt'
            torch.save(self.enc.state_dict(), path_save_enc)
        if model_to_save in {'dec', 'both'}:
            assert milestone is not None, "Must specify milestone for saving"
            path_save_dec = self.config.path_state_dict + '/decoder' + self.config.file_tag + f'-{milestone}.pt'

            torch.save(self.dec.state_dict(), path_save_dec)

    def load(self, path_load=None, milestone=None, milestone_dec=None, encoder_only=False):
        if path_load is None:
            assert milestone is not None, "Must specify milestone for loading"
            path_load_e = self.config.path_state_dict[:-len(self.config.timestamp)-1] + '/encoder' + self.config.file_tag + f'-{milestone}.pt'
        elif path_load[-3] != '.pt':
            path_load_e = path_load + '/encoder' + self.config.file_tag + f'-{milestone}.pt'
        else:
            path_load_e = path_load
        try:
            self.enc.load_state_dict(torch.load(path_load_e))
        except:
            self.enc.load_state_dict(torch.load(path_load_e, map_location='cpu'))

        self.enc.eval()

        if not encoder_only:
            if path_load is None:
                assert milestone is not None, "Must specify milestone for loading"
                path_load_d = self.config.path_state_dict[:-len(self.config.timestamp)-1] + '/decoder' + self.config.file_tag + f'-{milestone_dec}.pt'
            elif path_load[-3] != '.pt':
                path_load_d = path_load + '/decoder' + self.config.file_tag + f'-{milestone_dec}.pt'
            else:
                path_load_d = path_load

            try:
                self.dec.load_state_dict(torch.load(path_load_d))
                self.dec.eval()
            except:
                self.dec.load_state_dict(torch.load(path_load_d, map_location='cpu'))
                self.dec.eval()

        self.config.epoch_loaded = milestone - 1

    def train(self):
        enc, config, device = self.enc, self.config, self.device
        dec = self.dec

        M = config.M
        k = int(math.log2(M))
        n = config.n

        if config.loss_r == 'MCRB':
            loss_fn_r = utils.MCRB_mean
        elif config.loss_r == 'CRLB':
            loss_fn_r = utils.CRLB_mean
        else:
            raise ValueError(config.loss_r)

        def run_epoch(train_mode=None, turbo_original_structure=False):
            if train_mode == 'enc':
                enc.train()
            elif train_mode == 'dec':
                dec.train()
            elif train_mode == 'enc-dec':
                enc.train()
                dec.train()
            bs = config.batch_size
            # To make the dataset batched
            if config.generate_dataset:
                # load generated dataset
                loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=False, batch_size=config.batch_size, drop_last=True)
                pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5, dynamic_ncols=True )
            else:
                # pbar does not contain dataset. It generates an iterater based on the dataset size and batch size.
                pbar = tqdm(enumerate( range(config.dataset_size // config.batch_size) ),
                            total=int(config.dataset_size // config.batch_size), mininterval=0.5, dynamic_ncols=True)

            # Initialize the lists for evaluation
            loss_batch, loss_batch_c, loss_batch_r, batch_BER, batch_SER = [], [], [], [], []

            for it, m in pbar:
                # In case there is no generated dataset, generate in the loop.
                if config.generate_dataset is False:
                    m = np.random.randint(2, size=(bs, config.k, config.args.code_rate_k))
                    m = torch.from_numpy(m)
                else:
                    bs = m.shape[0]  # get the actual batch size. It can be different when the last batch is smaller.

                enc.zero_grad()

                # Enter inputs to the encoders
                m = m.to(device).float()
                x = enc(m)  # each sample is a 2n-dimensional real vector
                # In case of Turbo, e.g., [bs, block_len, code_rate_n], [16, 100, 3] 300- real valued symbols (codeword), -> [150, 2] 150 complex valued codeword.

                # real-to-complex conversion
                if turbo_original_structure:
                    x = torch.transpose(x, 1, 2).contiguous()
                x_ = x.view(bs, -1, 2)
                x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

                # Communication Channel
                if config.loss_ratio_c_to_r == 0:
                    loss_c = torch.tensor([0])
                else:
                    # Sample SNR values uniformly between the start and end point.
                    snr_random = config.args.snr_train_start - config.args.snr_train_end * np.random.rand(bs, 1) + config.args.snr_train_end
                    noise_std = torch.from_numpy(np.repeat(utils.SNR_to_noise(snr_random),  n, axis=1))
                    # Sample noise standard deviation uniformly between the start and the end point.
                    y_c, h_c = self.channel_c(x_, noise_std=noise_std)

                    if train_mode in {"enc", "enc-dec"}:
                        # Maximum-Likelihood Decoding
                        loglikelihood, _ = self.MLDetector(y_c,
                                                           x_, only_groundtruth = config.args.only_groundtruth, noise_std=noise_std) # likelihood of the correct message.
                        loss_c = - loglikelihood # equivalent to the binary cross entropy.

                    elif config.loss_c == 'decoder' or train_mode == "dec":
                        if turbo_original_structure:
                            y_c_real = (torch.transpose(torch.view_as_real(y_c), 1,2).contiguous()).reshape(bs, config.args.block_len, config.args.code_rate_n*config.args.enc_num_output_dim)
                        else:
                            y_c_real = torch.view_as_real(y_c).reshape(bs, config.args.block_len, config.args.code_rate_n*config.args.enc_num_output_dim)
                        m_hat = dec(y_c_real)
                        m_hat = torch.clamp(m_hat, 0.0, 1.0)
                        loss_c = customized_loss(m_hat, m, config.args)

                    loss_c = loss_c.mean()

                # Sensing channel
                if train_mode == 'dec' or config.loss_ratio_c_to_r == 1.:
                    # In case training only decoder, sensing loss is unnecessary.
                    loss = loss_c
                    loss_r = torch.tensor([0])
                else:
                    # Sensing Loss.
                    snr_random = config.args.snr_train_start - config.args.snr_train_end * np.random.rand(bs, 1) + config.args.snr_train_end
                    noise_std = torch.from_numpy(utils.SNR_to_noise(snr_random)).to(device) # For sensing, it should be the same N0 for all instances to make the MCRB to hold.

                    y_r, h_s_a, tau_s_a = self.channel_s(x_, N0= 2 * noise_std ** 2)

                    # Backward pass for estimator
                    loss_r = 0
                    for p in range(self.config.P):
                        loss_r += loss_fn_r(2 * noise_std ** 2, h_s_a[:,p], x_, tau_s_a[:,p], config.T).cpu()
                    loss_r = loss_r/self.config.P
                    loss_c.to(device)
                    loss = config.loss_ratio_c_to_r * loss_c + (
                            1 - config.loss_ratio_c_to_r) * loss_r * config.sensing_loss_scale  # normalization and good scaling needed.

                loss.backward()

                if train_mode == 'enc':
                    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1)
                elif train_mode == 'enc-dec':
                    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1)
                    torch.nn.utils.clip_grad_norm_(dec.parameters(), 1)
                elif train_mode == 'dec':
                    torch.nn.utils.clip_grad_norm_(dec.parameters(), 1)

                optimizer.step()

                loss_batch_c.append(loss_c.item())
                loss_batch_r.append(loss_r.item())
                loss_batch.append(loss.item())

                if (config.loss_c == 'decoder' or train_mode == "dec") and config.loss_ratio_c_to_r > 0:
                    m_hat_binary = torch.round(m_hat)
                    bool_sym = torch.ne(torch.squeeze(m).int(), torch.squeeze(m_hat_binary).int())
                    BER_tmp = torch.sum(bool_sym) / (m_hat.shape[0]*m_hat.shape[1])

                else:
                    BER_tmp = torch.tensor([0])
                batch_BER.append(BER_tmp.item())

                if (it+1) % 10 == 0:
                    pbar.set_description(
                        f"epoch {epoch_global + 1}: loss c, loss r, sum: {np.mean(loss_batch_c):.2e}, {np.mean(loss_batch_r):.2e}, {np.mean(loss_batch):.2e}")
            if (epoch + 1) % 1 == 0:
                print(f"epoch {epoch_global + 1}: BER {np.mean(batch_BER):.2e} FER {np.mean(batch_SER):.2e} {config.loss_r}:{np.mean(loss_batch_r):.2e} ")
            if (epoch + 1) % 10 == 0:
                if config.auto_save:
                    if train_mode == 'enc':
                        self.save(milestone = epoch_global + 1, model_to_save='enc')
                    elif train_mode == 'dec':
                        self.save(milestone = epoch_global + 1, model_to_save='dec')
                    elif train_mode == 'enc-dec':
                        self.save(milestone = epoch_global + 1, model_to_save='both')
            if False:
                if config.n <= 20:
                    Constellation_Enc(enc, M, path=f"{config.path_figures}/Constellation{config.file_tag}-{epoch_global+1}.pdf",
                                      title=f"{epoch_global+1} epochs", args=config.args)
                    Constellation_Enc_all(enc, M,
                                      path=f"{config.path_figures}/Constellation_all{config.file_tag}-{epoch_global + 1}.pdf",
                                      title=f"{epoch_global + 1} epochs", args=config.args)

            return np.mean(loss_batch_c), np.mean(loss_batch_r), np.mean(loss_batch)

        if self.config.epoch_loaded is None:
            epoch_global = 0
        else:
            epoch_global = self.config.epoch_loaded + 1

        loss_c_list, loss_r_list, loss_sum_list = [], [], []
        for lr_count, epoch_per_lr in enumerate(config.max_epochs):
            # Set the parameters to update by the optimizer
            if config.training_strategy == 'joint_training':
                if config.loss_ratio_c_to_r > 0:
                    optimizer = config.global_optimizer([
                                {'params': enc.parameters(), 'lr': config.lr_enc[lr_count]},
                                {'params': dec.parameters(), 'lr': config.lr_enc[lr_count]} ])
                else:
                    optimizer = config.global_optimizer(enc.parameters(), lr=config.lr_enc[lr_count])
            elif config.training_strategy == 'likelihood_training':
                optimizer = config.global_optimizer(enc.parameters(), lr=config.lr_enc[lr_count])

            for epoch in range(epoch_per_lr):
                if config.training_strategy == 'joint_training':
                    if config.loss_ratio_c_to_r > 0:
                        loss_c_ep, loss_r_ep, loss_sum_ep = run_epoch(train_mode='enc-dec')
                    else:
                        loss_c_ep, loss_r_ep, loss_sum_ep = run_epoch(train_mode='enc')
                elif config.training_strategy == 'iterative_training':
                    optimizer = config.global_optimizer(enc.parameters(), lr=config.lr_enc[lr_count])
                    for i in range(config.num_train_enc):
                        loss_c_ep, loss_r_ep, loss_sum_ep = run_epoch(train_mode='enc')
                    optimizer = config.global_optimizer(dec.parameters(), lr=config.lr_enc[lr_count])
                    for j in range(config.num_train_dec):
                        loss_c_ep, loss_r_ep, loss_sum_ep = run_epoch(train_mode='dec')

                loss_c_list.append(loss_c_ep); loss_r_list.append(loss_r_ep); loss_sum_list.append(loss_sum_ep)
                if (epoch_global+1) % 50 == 0:
                    loss_list = [loss_c_list, loss_r_list, loss_sum_list]
                    plot_loss_over_epochs(loss_list, epoch_global+1, f"{config.path_figures}/loss_{epoch_global+1}.pdf", config.LOADMODEL)
                epoch_global += 1

            loss_c_np_array = np.array(loss_c_list); loss_r_np_array = np.array(loss_r_list); loss_sum_np_array = np.array(loss_sum_list)
        np.save(f"{config.path_state_dict}/loss_c_np_array.npy", loss_c_np_array)
        np.save(f"{config.path_state_dict}/loss_r_np_array.npy", loss_r_np_array)
        np.save(f"{config.path_state_dict}/loss_sum_np_array.npy", loss_sum_np_array)

        return loss_c_list, loss_r_list, loss_sum_list

    def test(self, noise_std=None, turbo_original_structure=False):
        enc, config, device = self.enc, self.config, self.device
        dec = self.dec

        M = config.M
        k = int(math.log2(M))
        n = config.n

        bs = config.batch_size

        if noise_std==None:
            noise_std = self.config.args.snr_points

        if config.loss_r == 'MCRB':
            loss_fn_r = utils.MCRB_mean
        elif config.loss_r == 'CRLB':
            loss_fn_r = utils.CRLB_mean
        else:
            raise ValueError(config.loss_r)
        #line_plots_Enc(enc, M, args=config.args, path=f"{config.path_figures}/Line_plots{config.file_tag}.pdf")
        # To make the dataset batched
        # pbar does not contain dataset. It generates an iterater based on the dataset size and batch size.

        if config.n <= 20:
            if self.config.epoch_loaded is None:
                epoch_global = 0
            else:
                epoch_global = self.config.epoch_loaded
            line_plots_Enc(enc, M, args=config.args,
                           path=f"{config.path_figures}/Line_plots{config.file_tag}-{epoch_global + 1}.pdf"
                           )


            Constellation_Enc(enc, M,
                              path=f"{config.path_figures}/Constellation{config.file_tag}-{epoch_global + 1}.pdf",
                              args=config.args)
            #Constellation_Enc_all(enc, M,
            #                      path=f"{config.path_figures}/Constellation_all{config.file_tag}-{epoch_global + 1}.pdf",
            #                      args=config.args)


        pbar = tqdm(enumerate(range(config.dataset_size // config.batch_size)),
                    total=int(config.dataset_size // config.batch_size), mininterval=0.5, dynamic_ncols=True)

        # Initialize the lists for evaluation
        loss_batch, loss_batch_r, batch_BER, batch_SER, batch_BER_MLD, batch_SER_MLD = [], [], [], [], [], []

        for it, m in pbar:
            # In case there is no generated dataset, generate in the loop.
            m = np.random.randint(2, size=(bs, config.k, config.args.code_rate_k))
            m = torch.from_numpy(m)

            # Enter inputs to the encoders
            m = m.to(device).float()
            x = enc(m)  # each sample is a 2n-dimensional real vector
            # In case of Turbo, e.g., [bs, block_len, code_rate_n], [16, 100, 3] 300- real valued symbols (codeword), -> [150, 2] 150 complex valued codeword.

            # real-to-complex conversion
            if turbo_original_structure:
                x = torch.transpose(x, 1, 2).contiguous()
            x_ = x.view(bs, -1, 2)
            x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

            # Communication Channel
            y_c, h_c = self.channel_c(x_, noise_std=noise_std)

            # ML Decoding
            if config.MLD and M < 1025:

                m_tmp = torch.arange(M).to(device)
                m_tmp_bin = torch.squeeze(utils.dec2bin(m_tmp.int(), k))
                m_tmp_bin = m_tmp_bin.view(-1, config.k, config.args.code_rate_k).float()
                table = torch.view_as_complex(enc(m_tmp_bin).view(M, n, 2)).to(device).detach()

                table = table.view(M, n)
                loglikelihood_M, m_hat_MLD = self.MLDetector(y_c, table, only_groundtruth=False, std_z=noise_std)

                m_hat_MLD_binary = utils.dec2bin(m_hat_MLD, k)
                bool_bin_MLD = torch.ne(torch.squeeze(m).int(), torch.squeeze(m_hat_MLD_binary).int())
                BER_MLD_tmp = torch.sum(bool_bin_MLD) / (bool_bin_MLD.shape[0] * bool_bin_MLD.shape[1])
                batch_BER_MLD.append(BER_MLD_tmp.item())


            # BER / FER
            if turbo_original_structure:
                y_c_real = (torch.transpose(torch.view_as_real(y_c), 1,2).contiguous()).reshape(bs, config.args.block_len, config.args.code_rate_n)
            else:
                y_c_real = torch.view_as_real(y_c).reshape(bs, config.args.block_len, config.args.code_rate_n)
            m_hat = dec(y_c_real)
            m_hat = torch.clamp(m_hat, 0.0, 1.0)

            # Sensing channel
            # Sensing Loss.
            if True: #config.args.loss_ratio_c_to_r < 1:
                y_r, h_s_a, tau_s_a = self.channel_s(x_, N0= 2 * noise_std ** 2)

                # Backward pass for estimator
                loss_r = []
                for p in range(self.config.P):
                    loss_r.append( loss_fn_r(2 * noise_std ** 2, h_s_a[:, p], x_, tau_s_a[:, p], config.T).cpu().detach().numpy() )

                loss_r = np.hstack(np.array(loss_r))
                loss_batch_r.append(loss_r)
            else:
                loss_batch_r.append(0)

            # BER

            m_hat_binary = torch.round(m_hat)
            bool_sym = torch.ne(torch.squeeze(m).int(), torch.squeeze(m_hat_binary).int())
            BER_tmp = torch.sum(bool_sym) / (m_hat.shape[0] * m_hat.shape[1])

            batch_BER.append(BER_tmp.item())

            if (it + 1) % 10 == 0:
                pbar.set_description(
                    f"Test results:  BER, {config.loss_r}: BER {np.mean(batch_BER):.2e}, {np.mean(loss_batch_r):.2e}")

        print(
            f"SNR {utils.noise_to_SNR(noise_std):.2f}dB: BER {np.mean(batch_BER):.2e} FER {np.mean(batch_SER):.2e} {config.loss_r}:{np.mean(loss_batch_r):.2e}")


        if config.MLD:
            return np.mean(batch_BER), np.mean(batch_SER), loss_batch_r, np.mean(batch_BER_MLD)
        else:
            return np.mean(batch_BER), np.mean(batch_SER), loss_batch_r
