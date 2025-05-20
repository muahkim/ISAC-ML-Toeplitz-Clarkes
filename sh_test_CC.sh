source ~/venv_ml/bin/activate

mod_type_list=("ASK" "PSK" "DPSK")
f_D_list=("0" "0.1" "1")
for mod_type in "${mod_type_list[@]}"; do
	for f_D in "${f_D_list[@]}"; do
		python test_CC.py -f_Dn "$f_D" -P 1 -block_len 3 -code_rate_k 2 -code_rate_n 3 -snr_test_start=0 -snr_test_end=30 -snr_points 11 -batch_size 50 -mod_type="$mod_type"
	done
done
