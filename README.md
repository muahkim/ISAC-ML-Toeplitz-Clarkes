# Machine Learning-based Constellation Design for Integrated Sensing and Communication

This repository studies ML-based constellation/code design for integrated
sensing and communication (ISAC) with a Toeplitz sensing channel and a
correlated fading communication channel.

## Sensing Model

The sensing model is a Toeplitz multi-path fading channel. The sensing
objective is to estimate the delays of all paths. The main sensing metrics are
CRLB/MCRB-style lower bounds and empirical delay-estimation MSE.

## Communication Model

The communication model is a Rayleigh fading channel without inter-symbol
interference. Non-coherent detection is assumed, and the fading can be
correlated within a block according to an arbitrary covariance matrix.

## Environment Setup

The dependency snapshot in `requirements.txt` was generated from the local
`~/venv_ml` environment used for these experiments.

To recreate a similar environment:

```bash
python3 -m venv ~/venv_ml
source ~/venv_ml/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The local shell examples use `VENV_PATH`, defaulting to `~/venv_ml`. To use a
different environment without editing the scripts:

```bash
VENV_PATH=/path/to/your/venv bash sh_train.sh
```

## Main Workflows

### ML Model Training

`train.py` trains the autoencoder-based ISAC model.

Related files:

- `train_AWGN.py`: AWGN-focused training variant.
- `src/trainer.py`: training loop, checkpoint save/load, and model testing.
- `src/encoders.py`, `src/decoders.py`, `src/models.py`: neural model
  definitions.
- `src/get_args.py`: command-line argument definitions.

Local Bash examples:

- `sh_train.sh`
- `sh_train_p1_sensing.sh`

### ML Model BER and Bound Testing

`test.py` loads a trained encoder/decoder pair and evaluates BER, FER, and the
configured sensing lower bound (`MCRB` or `CRLB`) through the `Trainer.test`
path.

Local Bash examples:

- `sh_test.sh`
- `sh_test_P1.sh`

### Classical Code and Modulation BER Testing

`test_CC.py` evaluates BER for convolutional-code/modulation baselines using
helpers in `src/utils_standard_codec.py`.

Related files:

- `validate_CC.py`: validation utilities for the classical-code path.
- `test_MC.py`, `test_likelihood.py`: additional communication testing and
  likelihood-analysis scripts.

### CRLB/MCRB Comparison

`scripts/compare_crlb_mcrb.py` is the standalone CRLB-vs-MCRB comparison
workflow. It computes the marginalized covariance CRLB and compares it against
the existing MCRB curve for ZC, BPSK, OOK, and ML pilots. Use this script for
focused CRLB/MCRB studies and plots.

Related functions:

- `src/utils.py`: `CRLB_mean`, `MCRB_mean`, and SNR/noise utilities.
- `src/utils_standard_codec.py`: modulation, differential coding, and Zadoff-Chu
  sequence utilities.

### Empirical MSE Simulation

`test_MSE.py` is the current empirical delay-estimation MSE test workflow. It
supports top-peaks and SIC delay estimation, optional Brent refinement, genie
outage statistics, MCRB overlays, and aggregation of chunked trial-error files.

Reusable MSE-test helpers live in `src/utils_mse_test.py`. Lower-level Toeplitz
sensing simulation utilities live in `src/delay_simulation.py`.

Example:

```bash
python3 test_MSE.py \
  --P 2 \
  --tau-max 10 30 \
  --estimator sic \
  --sic-refine-iters 3 \
  --outage-threshold-ns 1.0 \
  --skip-mcrb
```

To aggregate chunked trial-error files:

```bash
python3 test_MSE.py \
  --combine-trial-errors 'results_genie_outage/*_trial_errors.npz' \
  --mcrb-pattern 'results_genie_outage/*_mcrb.npz' \
  --out-dir results_genie_outage
```

## Command-line Arguments

Most scripts use the shared parser in `src/get_args.py`. Refer to the shell
scripts for reproducible examples.

Important arguments:

- `-TURBO`: if true, use the TurboAE-style binary-data framework with
  interleaving and iterative decoding. If false, use the simpler autoencoder
  setup.
- `-encoder`, `-decoder`: select neural encoder and decoder classes. Available
  choices are listed in `src/get_args.py`.
- `-is_parallel`: use `0` for recurrent networks and `1` for CNN/non-recurrent
  networks.
- `-loss_ratio_c_to_r`: controls the communication/sensing loss tradeoff.
  `1` means communication-only training and `0` means sensing-only training.
- `-loss_r`: selects the sensing lower-bound objective, either `MCRB` or
  `CRLB`.
