source ~/venv_ml/bin/activate

f_list=("0.1")
g_list=("0.05" "0.1" "0.25" "0.5" "0.75")
model_list=("rate3_cnn")
for f in "${f_list[@]}"; do
	for g in "${g_list[@]}"; do
		for model in "${model_list[@]}"; do
			if [[ "$model" == *"$rnn"* ]]; then
				is_parallel=0
			else
				is_parallel=1
			fi
			python train.py -TURBO true -f_Dn "$f" -load_epoch 0 -P 5 -tau_max 1 2 3 5 7 -loss_ratio_c_to_r "$g" -snr_train_start 0 -snr_train_end 30 -snr_test_start 0 -snr_test_end 30 -snr_points 10 -training_strategy iterative_training -block_len 6 -encoder "$model" -decoder "$model" -enc_num_unit 16 -enc_num_layer 2 -dec_num_unit 16 -dec_num_layer 5 -num_iter_ft 5 -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -num_iteration 6 -is_parallel "$is_parallel" -is_same_interleaver 1 -num_block 100 -batch_size 500 -train_channel_mode block_norm -test_channel_mode block_norm --print_test_traj --fixed_block_power -loss bce
		done
	done
done


