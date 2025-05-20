source ~/venv_ml/bin/activate
model_list=("rate3_cnn" "TurboAE_rate3_rnn" "TurboAE_rate3_cnn")
for model in "${model_list[@]}"; do
	if [[ "$model" == *"$rnn"* ]]; then
		is_parallel=0
	else
		is_parallel=1
	fi
	python test.py -TURBO true -f_Dn 0 -P 5 -tau_max 1 2 3 5 7 -loss_ratio_c_to_r 0 -snr_points 10 -snr_test_start 0 -snr_test_end 30 -training_strategy joint_training -block_len 6 -encoder "$model" -decoder "$model" -enc_num_unit 16 -enc_num_layer 2 -dec_num_unit 16 -dec_num_layer 5 -num_iter_ft 5 -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -num_iteration 6 -is_parallel "$is_parallel" -is_same_interleaver 1 -num_block 50 -batch_size 500 -loss_r MCLB -train_channel_mode block_norm -test_channel_mode block_norm --print_test_traj --fixed_block_power -loss bce
done

python test.py -TURBO false -f_Dn 0 -P 5 -tau_max 1 2 3 5 7 -loss_ratio_c_to_r 0 -snr_points 10 -snr_test_start 0 -snr_test_end 30 -training_strategy joint_training -block_len 6 -code_rate_k 1 -code_rate_n 3 -num_block 50 -batch_size 500 -loss_r MCLB -loss bce
