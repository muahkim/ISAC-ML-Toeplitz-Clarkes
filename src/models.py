#An Implementation of Diffusion Network Model
#Oringinal source: https://github.com/acids-ircam/diffusion_models

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from channel_models import *

import utils


# Binary Input Encoders/decoders: 
class Encoder(nn.Module):
    def __init__(self, k, M, n, **extra_kwargs):
        super(Encoder, self).__init__()

        self.k = k
        self.M = M
        self.n = n

        use_cuda =  torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self._f = nn.Sequential(            
            nn.Linear(k, M),
            nn.ELU(),
            nn.Linear(M,M),
            nn.ELU(),
            nn.Linear(M, n)#,norm_layer
        )
        
    def power_constraint(self, codes):
        codes_mean = torch.mean(codes)
        codes_std  = torch.std(codes)
        codes_norm = (codes-codes_mean)/ codes_std
        
        return codes_norm
        
    def forward(self, inputs):
        inputs = inputs.type(torch.FloatTensor).to(self.this_device)
        _x = self._f(inputs)
        x = self.power_constraint(_x)
        return x

class Decoder_1h(nn.Module):
    def __init__(self,k, M ,n, **extra_kwargs):
        super(Decoder_1h, self).__init__()
        self._f = nn.Sequential(
            nn.Linear(n, M),
            nn.ELU(),
            nn.Linear(M, M),
            nn.ELU(),
            nn.Linear(M, M),
            #nn.Linear(M, M),
            #nn.Sigmoid()
)
        
    def forward(self, inputs):
        device_inputs = inputs.device
        inputs = inputs.type(torch.FloatTensor)
        inputs = inputs.to(device_inputs)
        x = self._f(inputs)
        return x
    
# one-hot encoders/decoders:
    
class Encoder_1h_n(nn.Module):
    def __init__(self, M, n, **extra_kwargs):
        super(Encoder_1h_n, self).__init__()
        
        use_cuda =  torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        
        self._f = nn.Sequential(
            nn.Linear(M, M),
            nn.ELU(),
            nn.Linear(M,M),
            nn.ELU(),
            #nn.Linear(M,n),
            #nn.ELU(),
            nn.Linear(M, n)#,norm_layer
        )
        
    def power_constraint(self, codes):
        codes_mean = torch.mean(codes)
        codes_std  = torch.std(codes)
        codes_norm = (codes-codes_mean)/ codes_std / (2 ** 0.5) # Because of complex signals
        
        return codes_norm 
        
    def forward(self, inputs):
        # inputs: tensor [bs, 1], integer numbers from 0 to M-1.
        inputs = F.one_hot(inputs.long(), num_classes=self.M).float()#.reshape(-1, self.M)
        inputs = inputs.to(self.this_device)
        _x = self._f(inputs)
        x = self.power_constraint(_x)
        return x


class Encoder_1h(nn.Module):
    def __init__(self, M, n, power_constraint_type=None, **extra_kwargs):
        super(Encoder_1h, self).__init__()

        use_cuda =  torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.n = n
        self.M = M
        self._f = nn.Sequential(
            nn.Linear(M, M),
            nn.ELU(),
            nn.Linear(M,M),
            nn.ELU(),
            #nn.Linear(M,n),
            #nn.ELU(),
            nn.Linear(M, n)#,norm_layer
        )
        
        power_constraint_dict = {
            'batch_norm' : self.power_constraint_batch_norm,
            'per_sym_avgd_over_m' :  self.power_constraint_per_sym_avgd_over_messages,
            'per_frame' : self.power_constraint_per_frame}
        
        if power_constraint_type == None:
            power_constraint_type = 'per_frame'

        assert (power_constraint_dict.get(power_constraint_type) is not None), "Invalid power normalization type. Possible types are batch_norm, per_sym_avgd_over_m, per_frame."
  
        self.power_constraint = power_constraint_dict.get(power_constraint_type)
        
    # Power Normalization #1
    def power_constraint_batch_norm(self, codes):
        codes_mean = torch.mean(codes)
        codes_std  = torch.std(codes)
        codes_norm = (codes-codes_mean)/ codes_std * np.sqrt(self.P_in)
        return codes_norm

    # Power Normalization #2
    def power_per_sym_avgd_over_messages(self):
        inputs = torch.eye(self.M).to(self.this_device)
        codes = self._f(inputs)
        codes = codes.view(-1, self.n //2, 2)
        P = torch.mean(torch.sum(codes**2, dim=2), dim = 0)
        return P #[n] average power tensor
    
    def power_constraint_per_sym_avgd_over_messages(self, codes):
        P = self.power_per_sym_avgd_over_messages() 
        bs = codes.size(0)
        P = P[None,:,None].repeat(bs,1,2).view(bs, -1) # bs, 2*n
        codes_norm = codes / (P ** 0.5)
        return codes_norm
        
    # Power Normalization #3
    def power_constraint_per_frame(self, codes):
        P_frame = torch.sum(codes **2, dim=1)
        codes_norm = codes / (P_frame[:,None].repeat(1, self.n) ** 0.5) * (self.n//2) ** 0.5
        return codes_norm

    def forward(self, inputs):
        # inputs: tensor [bs, 1], integer numbers from 0 to M-1.
        inputs = F.one_hot(inputs.long(), num_classes=self.M).float()#.reshape(-1, self.M)
        inputs = inputs.to(self.this_device)
        _x = self._f(inputs)
        x = self.power_constraint(_x)
        return x


class Encoder_1h_n2_deeper(nn.Module):
    def __init__(self, M, n, n_hl=None, power_constraint_type=None, **extra_kwargs):
        super(Encoder_1h_n2_deeper, self).__init__()

        use_cuda =  torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.M = M
        self.n = n
        if n_hl == None:
            n_hl = M
        
        self._f = nn.Sequential(
            nn.Linear(M, n_hl),
            nn.ELU(),
            nn.Linear(n_hl, n_hl),
            nn.ELU(),
            nn.Linear(n_hl,n_hl),
            nn.ELU(),
            nn.Linear(n_hl, n)#,norm_layer
        )
        
        power_constraint_dict = {
            'batch_norm' : self.power_constraint_batch_norm,
            'per_sym_avgd_over_m' :  self.power_constraint_per_sym_avgd_over_messages,
            'per_frame' : self.power_constraint_per_frame}
        
        if power_constraint_type == None:
            power_constraint_type = 'per_frame'

        assert (power_constraint_dict.get(power_constraint_type) is not None), "Invalid power normalization type. Possible types are batch_norm, per_sym_avgd_over_m, per_frame."
              
        self.power_constraint = power_constraint_dict.get(power_constraint_type)
        
    # Power Normalization #1
    def power_constraint_batch_norm(self, codes):
        codes_mean = torch.mean(codes)
        codes_std  = torch.std(codes)
        codes_norm = (codes-codes_mean)/ codes_std * np.sqrt(self.P_in)
        return codes_norm

    # Power Normalization #2
    def power_per_sym_avgd_over_messages(self):
        inputs = torch.eye(self.M).to(self.this_device)
        codes = self._f(inputs)
        codes = codes.view(-1, self.n //2, 2)
        P = torch.mean(torch.sum(codes**2, dim=2), dim = 0)
        return P #[n] average power tensor
    
    def power_constraint_per_sym_avgd_over_messages(self, codes):
        P = self.power_per_sym_avgd_over_messages() 
        bs = codes.size(0)
        P = P[None,:,None].repeat(bs,1,2).view(bs, -1) # bs, 2*n
        codes_norm = codes / (P ** 0.5)
        return codes_norm
        
    # Power Normalization #3
    def power_constraint_per_frame(self, codes):
        P_frame = torch.sum(codes **2, dim=1)
        codes_norm = codes / (P_frame[:,None].repeat(1, self.n) ** 0.5) * (self.n//2) ** 0.5
        return codes_norm

    def forward(self, inputs):
        # inputs: tensor [bs, 1], integer numbers from 0 to M-1.
        inputs = F.one_hot(inputs.long(), num_classes=self.M).float()#.reshape(-1, self.M)
        inputs = inputs.to(self.this_device)
        _x = self._f(inputs)
        x = self.power_constraint(_x)
        return x
        
class Encoder_1h_n2_deeper_e(nn.Module):
    def __init__(self, M, n, n_hl=None, power_constraint_type=None, **extra_kwargs):
        super(Encoder_1h_n2_deeper_e, self).__init__()

        use_cuda =  torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.M = M
        self.n = n
        if n_hl == None:
            n_hl = M

        self._emb = nn.Embedding(M, n)
        self._f = nn.Sequential(
            nn.Linear(M, n_hl),
            nn.ELU(),
            nn.Linear(n_hl, n_hl),
            nn.ELU(),
            nn.Linear(n_hl,n_hl),
            nn.ELU(),
            nn.Linear(n_hl, n)#,norm_layer
        )
        
        power_constraint_dict = {
            'batch_norm' : self.power_constraint_batch_norm,
            'per_sym_avgd_over_m' :  self.power_constraint_per_sym_avgd_over_messages,
            'per_frame' : self.power_constraint_per_frame}
        
        if power_constraint_type == None:
            power_constraint_type = 'per_frame'

        assert power_constraint_dict.get(power_constraint_type) is not None, "Invalid power normalization type. Possible types are batch_norm, per_sym_avgd_over_m, per_frame."  
  
        self.power_constraint = power_constraint_dict.get(power_constraint_type)
        
    # Power Normalization #1
    def power_constraint_batch_norm(self, codes):
        codes_mean = torch.mean(codes)
        codes_std  = torch.std(codes)
        codes_norm = (codes-codes_mean)/ codes_std * np.sqrt(self.P_in)
        return codes_norm

    # Power Normalization #2
    def power_per_sym_avgd_over_messages(self):
        inputs = torch.eye(self.M).to(self.this_device)
        codes = self._f(inputs)
        codes = codes.view(-1, self.n //2, 2)
        P = torch.mean(torch.sum(codes**2, dim=2), dim = 0)
        return P #[n] average power tensor
    
    def power_constraint_per_sym_avgd_over_messages(self, codes):
        P = self.power_per_sym_avgd_over_messages() 
        bs = codes.size(0)
        P = P[None,:,None].repeat(bs,1,2).view(bs, -1) # bs, 2*n
        codes_norm = codes / (P ** 0.5)
        return codes_norm
        
    # Power Normalization #3
    def power_constraint_per_frame(self, codes):
        P_frame = torch.sum(codes **2, dim=1)
        codes_norm = codes / (P_frame[:,None].repeat(1, self.n) ** 0.5) * (self.n//2) ** 0.5
        return codes_norm

    def forward(self, inputs):
        # inputs: tensor [bs, 1], integer numbers from 0 to M-1.
        inputs = F.one_hot(inputs.long(), num_classes=self.M).float()#.reshape(-1, self.M)
        inputs = inputs.to(self.this_device)
        _x = self._f(inputs)
        inputs_dec = torch.argmax(inputs,dim=1).to(self.this_device)
        emb = self._emb(inputs_dec)
        x = self.power_constraint(_x * emb)
        return x

class Encoder_1h_n2_deeper_ee(nn.Module):
    def __init__(self, M, n, n_hl=None, power_constraint_type=None, **extra_kwargs):
        super(Encoder_1h_n2_deeper_ee, self).__init__()

        use_cuda =  torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.M = M
        self.n = n
        if n_hl == None:
            n_hl = M

        self.lin1 = nn.Linear(M, n_hl)
        self.lin2 = nn.Linear(n_hl, n_hl)
        self.lin3 = nn.Linear(n_hl, n_hl)
        self.lin4 = nn.Linear(n_hl, n_hl)
        self.lin5 = nn.Linear(n_hl, n_hl)
        self.lin6 = nn.Linear(n_hl, n)
        
        self.emb1 = nn.Embedding(M, n_hl)
        self.emb2 = nn.Embedding(M, n_hl)
        self.emb3 = nn.Embedding(M, n_hl)
        self.emb4 = nn.Embedding(M, n)

        self.elu = nn.ELU()

        
        power_constraint_dict = {
            'batch_norm' : self.power_constraint_batch_norm,
            'per_sym_avgd_over_m' :  self.power_constraint_per_sym_avgd_over_messages,
            'per_frame' : self.power_constraint_per_frame}
        
        if power_constraint_type == None:
            power_constraint_type = 'per_frame'

        assert power_constraint_dict.get(power_constraint_type) is not None, "Invalid power normalization type. Possible types are batch_norm, per_sym_avgd_over_m, per_frame."  
  
        self.power_constraint = power_constraint_dict.get(power_constraint_type)
        
    # Power Normalization #1
    def power_constraint_batch_norm(self, codes):
        codes_mean = torch.mean(codes)
        codes_std  = torch.std(codes)
        codes_norm = (codes-codes_mean)/ codes_std * np.sqrt(self.P_in)
        return codes_norm

    # Power Normalization #2
    def power_per_sym_avgd_over_messages(self):
        inputs = torch.eye(self.M).to(self.this_device)
        codes = self._f(inputs)
        codes = codes.view(-1, self.n //2, 2)
        P = torch.mean(torch.sum(codes**2, dim=2), dim = 0)
        return P #[n] average power tensor
    
    def power_constraint_per_sym_avgd_over_messages(self, codes):
        P = self.power_per_sym_avgd_over_messages() 
        bs = codes.size(0)
        P = P[None,:,None].repeat(bs,1,2).view(bs, -1) # bs, 2*n
        codes_norm = codes / (P ** 0.5)
        return codes_norm
        
    # Power Normalization #3
    def power_constraint_per_frame(self, codes):
        P_frame = torch.sum(codes **2, dim=1)
        codes_norm = codes / (P_frame[:,None].repeat(1, self.n) ** 0.5) * (self.n//2) ** 0.5
        return codes_norm

    def forward(self, inputs):
        # inputs: tensor [bs, 1], integer numbers from 0 to M-1.
        inputs = F.one_hot(inputs.long(), num_classes=self.M).float()#.reshape(-1, self.M)
        inputs = inputs.to(self.this_device)
        inputs_dec = torch.argmax(inputs, dim=1).to(self.this_device)
        
        x = self.elu( self.lin1(inputs) ) * self.emb1(inputs_dec)
        x = self.elu( self.lin2(x) )  
        x = self.elu( self.lin3(x) )  * self.emb2(inputs_dec)
        x = self.elu( self.lin4(x) )  
        x = self.elu( self.lin5(x) )  * self.emb3(inputs_dec)
        x = self.lin6(x) * self.emb4(inputs_dec)
        
        x = self.power_constraint(x)
        return x


class Encoder_embedding(nn.Module):
    def __init__(self, M, n, power_constraint_type=None, **extra_kwargs):
        super(Encoder_embedding, self).__init__()

        use_cuda =  torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.M = M
        self.n = n

        self._f = nn.Embedding(M,n)        
        
        power_constraint_dict = {
            'batch_norm' : self.power_constraint_batch_norm,
            'per_sym_avgd_over_m' :  self.power_constraint_per_sym_avgd_over_messages,
            'per_frame' : self.power_constraint_per_frame}
        
        if power_constraint_type == None:
            power_constraint_type = 'per_frame'

        assert power_constraint_dict.get(power_constraint_type) is not None, "Invalid power normalization type. Possible types are batch_norm, per_sym_avgd_over_m, per_frame."  
  
        self.power_constraint = power_constraint_dict.get(power_constraint_type)
        
    # Power Normalization #1
    def power_constraint_batch_norm(self, codes):
        codes_mean = torch.mean(codes)
        codes_std  = torch.std(codes)
        codes_norm = (codes-codes_mean)/ codes_std * np.sqrt(self.P_in)
        return codes_norm

    # Power Normalization #2
    def power_per_sym_avgd_over_messages(self):
        inputs = torch.eye(self.M).to(self.this_device)
        codes = self._f(inputs)
        codes = codes.view(-1, self.n //2, 2)
        P = torch.mean(torch.sum(codes**2, dim=2), dim = 0)
        return P #[n] average power tensor
    
    def power_constraint_per_sym_avgd_over_messages(self, codes):
        P = self.power_per_sym_avgd_over_messages() 
        bs = codes.size(0)
        P = P[None,:,None].repeat(bs,1,2).view(bs, -1) # bs, 2*n
        codes_norm = codes / (P ** 0.5)
        return codes_norm
        
    # Power Normalization #3
    def power_constraint_per_frame(self, codes):
        P_frame = torch.sum(codes **2, dim=1)
        codes_norm = codes / (P_frame[:,None].repeat(1, self.n) ** 0.5) * (self.n//2) ** 0.5
        return codes_norm

    def forward(self, inputs):
#        inputs = inputs.to(self.this_device)
        inputs = torch.argmax(inputs, dim=1).to(self.this_device).squeeze()
        _x = self._f(inputs)
        x = self.power_constraint(_x)
        return x

class Encoder_TConv1D(torch.nn.Module):
    def __init__(self, M, n, input_channels=20):
        super(Encoder_TConv1D, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec
        self.M = M
        self.n = n

        self.TConv = nn.Sequential(
            nn.Embedding(num_messages, input_channels),
            nn.Unflatten(1, (1, input_channels)),
            nn.ConvTranspose1d(1, init_dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(init_dim, init_dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(init_dim, init_dim, 4, 2, 1),
            nn.ReLU()
        )
        
    def power_constraint_per_frame(self, codes):
        P_frame = torch.sum(codes **2, dim=1)
        codes_norm = codes / (P_frame[:,None].repeat(1, 2*self.n) ** 0.5) * (2*self.n//2) ** 0.5
        return codes_norm
        
    def forward(self, inputs):

        #inputs = torch.argmax(inputs, dim=1).to(self.device)
        inputs = inputs.to(self.device)
        x = self.TConv(inputs)
        
        x = torch.flatten(x, 1)
        dim_fc = x.shape[1]
        self.dim_fc = dim_fc

        self.final_fc1 = nn.Linear(dim_fc, (dim_fc-2*self.n)//2 + 2*self.n)
        self.final_fc2 = nn.Linear((dim_fc-2*self.n)//2 + 2*self.n, (dim_fc-n)//4 + 2*self.n)
        self.final_fc3 = nn.Linear((dim_fc-2*self.n)//4 + 2*self.n, 2*self.n)

        x = self.final_act(self.final_fc1(x))
        x = self.final_act(self.final_fc2(x))
        x = self.final_act(self.final_fc3(x))
        
        x = self.power_constraint_per_frame(x)
        
        return x

## Write a 1D LeNet-5, designed for longer block length. Refer to the U-Net code for a general and stable code. 

#class LeNet_1D(torch.nn.Module):
#    def 

        
## The channel models are for neuran encoder and decoder pairs. For ML decoder, they are not necessary.

class Channel(torch.nn.Module):
    def __init__(self, enc, dec):
        super(Channel, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec

    def forward(self, inputs, noise_std):
        #End-to-End Framework
        codeword  = self.enc(inputs)
        
        i,j = codeword.size()
        channel_noise = noise_std*torch.randn((i, j)).to(self.device) # X = sigma*Z, where Z is the standard normal N(0,1) and sigma the standard deviation, and X = N(0, sigma)
        
        rec_signal = codeword + channel_noise
        dec_signal = self.dec(rec_signal)

        return dec_signal
    
class Channel_ray_n(torch.nn.Module):
    def __init__(self, enc, dec):
        super(Channel_ray_n, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec

    def forward(self, inputs, noise_std):
        #End-to-End Framework
        codeword  = self.enc(inputs)
        
        i,j = codeword.size()
        
        h_ray = (1/np.sqrt(2))*torch.sqrt( torch.randn((i, j), device = self.device)**2 + torch.randn((i, j), device = self.device)**2)
        channel_noise = noise_std*torch.randn((i, j), device = self.device) # X = sigma*Z, where Z is the standard normal N(0,1) and sigma the standard deviation, and X = N(0, sigma)
        
        rec_signal = codeword + torch.div(channel_noise, h_ray)
        dec_signal = self.dec(rec_signal)

        return dec_signal
    
class Channel_ray(torch.nn.Module):
    def __init__(self, enc, dec):
        super(Channel_ray, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec

    def forward(self, inputs, noise_std, Ray_param):
        #End-to-End Framework
        codeword  = self.enc(inputs)
        
        i,j = codeword.size()
        
        h_ray = (1/np.sqrt(2))*Ray_param*torch.sqrt( torch.rand(size = codeword.size(), device = self.device)**2 +torch.rand(size = codeword.size(), device = self.device)**2)
        channel_noise = noise_std*torch.randn((i, j), device = self.device) # X = sigma*Z, where Z is the standard normal N(0,1) and sigma the standard deviation, and X = N(0, sigma)
        
        rec_signal = codeword*h_ray + channel_noise
        dec_signal = self.dec(rec_signal)

        return dec_signal

class Channel_ray_c(torch.nn.Module):
    def __init__(self, enc, dec):
        super(Channel_ray_c, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec

    def forward(self, inputs, noise_std):
        #End-to-End Framework
        codeword  = self.enc(inputs)
        n = codeword.size(1) // 2
        codeword = codeword.view(-1,n,2)
        codeword = torch.view_as_complex(codeword)
              
        rec_signal, h = ch_Rayleigh_AWGN_c_nb(codeword, 1, noise_std, self.device)#codeword*h_ray + channel_noise
        rec_signal = torch.view_as_real(rec_signal)
        rec_signal = rec_signal.view(-1,2*n)
        dec_signal = self.dec(rec_signal)

        return dec_signal

class Channel_ISAC_corrRayleigh(torch.nn.Module):
    def __init__(self, enc, Config_ch, device, is_conv=True, loss_ratio=0.5):
        super(Channel_ISAC_corrRayleigh, self).__init__()
        self.enc = enc
        self.conf = Config_ch
        self.device = device
        self.is_conv = is_conv
        self.is_emb = enc.is_emb
        self.loss_ratio = loss_ratio

    def forward(self, data):
        bs = data.size(0)
        M = self.conf.M
        n = self.conf.blocklength
        device = self.device
        enc = self.enc        
        loss_CE = nn.CrossEntropyLoss()
        
        labels = data.reshape((-1, 1)).to(device)
        if self.is_emb:
            m_tmp = torch.arange(M).to(device)
        else:
            m_tmp = torch.eye(M).to(device)
        
        if self.is_conv and not self.is_emb:
            table = torch.view_as_complex(enc(m_tmp.unsqueeze(1)).view(M, n, 2)).to(device).detach()
        else:
            table = torch.view_as_complex(enc(m_tmp).view(M, n, 2)).to(device).detach()

        enc.zero_grad()

        # Forward Pass:
        if not self.is_emb:
            m = F.one_hot(labels.long(), num_classes=M)
            m = m.float().reshape(-1, M)
        else:
            m = labels.squeeze().long()

        if self.is_conv and self.is_emb == False:
            x = enc(m.unsqueeze(1))
        else:    
            x = enc(m)  # each sample is a 2n-dimensional real vector

        x_ = x.view(-1, n, 2)  # each sample is a n x 2 tensor.
        x_ = torch.view_as_complex(x_)  # each sample is a n-dimensional complex vector

        # Communication and Radar Channel
        y_c, h_c = ch_corr_Rayleigh_by_cov(x_, self.conf.cov, self.conf.noise_std, device)

        # Maximum-Likelihood Decoding
        prob, m_hat = utils.ML_det_Rayleigh_ncd(y_c, table, self.conf.cov, self.conf.noise_std)

        # Backward pass for decoder
        loss_c = loss_CE(prob, labels.long().squeeze(1))
        loss_c = loss_c.mean()
        
        # Sensing
        y_r, h_r = ch_corr_Rayleigh_by_cov(x_, self.conf.cov, self.conf.noise_std, device)
        h_r_hat = utils.MMSE_H_estimator_corr(x_, y_r, self.conf.noise_std, self.conf.cov)
        #                h_r_hat = utils.MMSE_H_estimator_iid_w_priori(x_, y_r, config.noise_std_r, config.correlation, 0., 1.)

        # Backward pass for estimator
        assert h_r.shape == h_r_hat.shape
        loss_r = utils.MSE_c(h_r, h_r_hat)

        loss = self.loss_ratio * loss_c + (1 - self.loss_ratio) * loss_r  # normalization and good scaling needed.
        
        return loss
    
class Channel_by_cov_nb(torch.nn.Module):
    def __init__(self, enc, dec, cov):
        super(Channel_by_cov_nb, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec
        self.cov = cov

    def forward(self, inputs, noise_std):
        codeword = self.enc(inputs)
        n = codeword.size(1) // 2
        codeword = codeword.view(-1, n, 2)
        codeword = torch.view_as_complex(codeword)

        rec_signal, _, _, _, _ = channels_by_cov_2mat_nb(codeword, self.cov, noise_std, noise_std, self.device)

        rec_signal = torch.view_as_real(rec_signal)
        rec_signal = rec_signal.view(-1, 2 * n)
        dec_signal = self.dec(rec_signal)

        return dec_signal

class Channel_by_cov_nb_cd(torch.nn.Module):
    def __init__(self, enc, dec, cov):
        super(Channel_by_cov_nb_cd, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec
        self.cov = cov

    def forward(self, inputs, noise_std):
        codeword = self.enc(inputs)
        n = codeword.size(1) // 2
        codeword = codeword.view(-1, n, 2)
        codeword = torch.view_as_complex(codeword)

        rec_signal, _, h_c, _, _ = channels_by_cov_2mat_nb(codeword, self.cov, noise_std, noise_std, self.device)

        rec_signal = torch.view_as_real(rec_signal)
        rec_signal = rec_signal.view(-1, 2 * n)
        h_c_r = torch.view_as_real(h_c).reshape(-1, 2*n)
        dec_signal = self.dec(rec_signal, h_c_r)

        return dec_signal

class Channel_burst(torch.nn.Module):
    def __init__(self, enc, dec):
        super(Channel_burst, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec

    def forward(self, inputs, noise_std):
        #End-to-End Framework
        codeword  = self.enc(inputs)
        
        i,j = codeword.size()
        
        channel_noise = noise_std*torch.randn((i, j)).to(self.device) # X = sigma*Z, where Z is the standard normal N(0,1) and sigma the standard deviation, and X = N(0, sigma)
        burst = np.sqrt(2)*channel_noise
        burst_prob = torch.bernoulli(torch.tensor(.1)).to(self.device)
        
        rec_signal = codeword + channel_noise + burst*burst_prob
        dec_signal = self.dec(rec_signal)

        return dec_signal

## Neural Estimators 
class Hs_estimator(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self._f = nn.Sequential(
            nn.Linear(n**2,n**2),
            nn.ELU(),
            nn.Linear(n**2,n**2),
            nn.ELU(),
            #nn.Linear(M,n),
            #nn.ELU(),
            nn.Linear(n**2, n**2)#,norm_layer
        )
        
    def forward(self, x, y):
        assert(x.size() == y.size())
        bs = x.size(0)
        n_r = x.size(1)

        x = x.view(bs, n_r//2, 2)
        x_c = torch.view_as_complex(x)
        y = y.view(bs, n_r//2, 2)
        y_c = torch.view_as_complex(y)

        h_LS = torch.div(y_c, x_c)
        h_LS = torch.view_as_real(h_LS)
        h_LS = h_LS.view(bs, -1)

        #h_LS = torch.bmm(y[:,:, None], torch.div(1,x)[:, None,:])
        #shape_h_LS = h_LS.size()
        h_r_hat = self._f(h_LS)
        h_r_hat = h_r_hat.view(bs, self.n, self.n)
        return h_r_hat

class Hs_estimator2(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self._f = nn.Sequential(
            nn.Linear(2 * 2 * n, n ** 2),
            nn.ELU(),
            nn.Linear(n ** 2, n ** 2),
            nn.ELU(),
            # nn.Linear(M,n),
            # nn.ELU(),
            nn.Linear(n ** 2, 2 * n ** 2)  # ,norm_layer
        )

    def forward(self, x, y):
        assert (x.size() == y.size())
        x = x.to(self.device)
        y = y.to(self.device)
        xy = torch.cat((x,y), dim=1)
        h_r_hat = self._f(xy)
        h_r_hat = h_r_hat.view(-1, self.n, self.n, 2)
        return h_r_hat

class Hs_estimator3(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self._f = nn.Sequential(
            nn.Linear(2 * n, n ** 2),
            nn.ELU(),
            nn.Linear(n ** 2, n ** 2),
            nn.ELU(),
            # nn.Linear(M,n),
            # nn.ELU(),
            nn.Linear(n ** 2, 2 * n)  # ,norm_layer
        )

    def forward(self, x, y):
        assert (x.size() == y.size())
        bs = x.size(0)
        n = self.n
        x = x.to(self.device)
        y = y.to(self.device)
        x = x.view(bs, n, 2)
        x_c = torch.view_as_complex(x)
        y = y.view(bs, n, 2)
        y_c = torch.view_as_complex(y)

        h_LS = torch.div(y_c, x_c)
        h_LS = torch.view_as_real(h_LS)
        h_LS = h_LS.view(bs, -1)

        h_r_hat = self._f(h_LS)
       # h_r_hat_mat = torch.zeros(bs, self.n, self.n, 2, device=self.device)
       # h_r_hat_mat[:, np.arange(n), np.arange(n), :] = h_r_hat.view(-1, self.n, 2)
        return h_r_hat


