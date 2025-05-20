# Machine Learning-based Constellation Design for Integrated Sensing and Communication: Toeplitz Sensing Channel and Correlated Fading Communication Channel.

# Sensing Model
Toeplitz model with multiplaths fading chanel. The sensing objective is to estimate delays of all paths.
# Commuinication Model
Rayleigh fading channel without inter-symbol interference is considered. Non-coherent detection is assumed, and the fading can be correlated within the block with an arbitrary covariance matrix. 

# Executable Python scripts
## train.py
Train an autoencoder model for defined ISAC model. 
## test.py
Load a trained autoencoder pair and test its BER and MCLB. 
## test_CC.py
Test the BER of a convolutional code with modulation schemes. 
## test_MCLB.py
Test the MCLB of modulation schemes. For this channel coding is not considered.

## How to set the simulations by using the arguments. => Refer to the shell scripts for examples.
#### Turbo: If true, the autoencoders are defined by the turbo AE framework, using binary data, interleaving, and successive decoding, etc. If false, it is using simple autoencoder one-hot encoded.
#### encoder and decoder model: It is only active when <turbo> argument is true. The list of encoders and decoders can be found in src/get_args.py.
#### is_paralel: If a recurrent network is used, it should be 0. Otherwise, it should be 1. 



