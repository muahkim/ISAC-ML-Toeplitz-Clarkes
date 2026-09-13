#!/usr/bin/env bash

VENV_PATH="${VENV_PATH:-$HOME/venv_ml}"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
	echo "Virtual environment not found: $VENV_PATH"
	echo "Create it with: python3 -m venv \"$VENV_PATH\" && source \"$VENV_PATH/bin/activate\" && pip install -r requirements.txt"
	exit 1
fi
source "$VENV_PATH/bin/activate"
f_list=("0")
f_load=("0.1")
g_list=("0")
model_list=("rate3_cnn")
for f_l in "${f_list[@]}"; do
	for g in "${g_list[@]}"; do
		for model in "${model_list[@]}"; do
			if [[ "$model" == *"rnn"* ]]; then
				is_parallel=0
			else
				is_parallel=1
			fi
			python test.py -TURBO true -f_Dn "$f_l" -f_Dn_load "$f_load" -load_epoch 150 -P 1 -tau_max 3 -loss_ratio_c_to_r "$g" -snr_points 21 -snr_test_start -5 -snr_test_end 35 -training_strategy joint_training -block_len 6 -encoder "$model" -decoder "$model" -enc_num_unit 16 -enc_num_layer 2 -dec_num_unit 16 -dec_num_layer 5 -num_iter_ft 5 -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -num_iteration 6 -is_parallel "$is_parallel" -is_same_interleaver 1 -num_block 100 -batch_size 500 -train_channel_mode block_norm -test_channel_mode block_norm --print_test_traj --fixed_block_power -loss bce
		done
	done
done
