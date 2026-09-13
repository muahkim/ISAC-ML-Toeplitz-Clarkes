#!/usr/bin/env python3
from src.utils_mse_test import run_mse_test_from_cli


def print_usage():
    print("Usage examples:")
    print("  python3 test_MSE.py --skip-mcrb --n-mc 100 --pilot-types zc --snr-db 40")
    print("  python3 test_MSE.py --P=2 --tau-max=10 30 --pilot-types=zc")
    print("  python3 test_MSE.py --P=1 --tau-max=10 30  # two P=1 geometries")
    print("  python3 test_MSE.py --outage-threshold-ns 0.5 --n-mc 10000")
    print("")
    print("Main options mirror MSE_sim_Brent_wLB.py:")
    print("  --n-mc N, --snr-db DB, --snr-min-db DB, --snr-max-db DB, --snr-points N")
    print("  --pilot-types zc,bpsk,ook,ml")
    print("  --rng-seed SEED, --pilot-seed SEED, --job-index IDX")
    print("  --N N, --K K  override sequence/code dimensions")
    print("  --P N, --tau-max VALUE [VALUE ...]  accepts --P=2 and --tau-max=10,30 too")
    print("  --skip-mcrb, --mcrb-only, --mcrb-samples N, --mcrb-batch-size N")
    print("  --brent-xatol-ns VALUE")
    print("  --estimator top-peaks|sic  default: top-peaks")
    print("  --sic-min-separation-ns VALUE  default: delay-grid step")
    print("  --sic-refine-iters N  default: 3")
    print("  --outage-threshold-ns VALUE  default: 0.5")
    print("  --out-dir DIR  default: results_genie_outage")
    print("  --combine-trial-errors 'results_genie_outage/*_trial_errors.npz'")
    print("      Combine genie-outage chunks into all/non-outage MSE, P_out, CSV, and plots.")
    print("      Optional: --mcrb-pattern 'results/*_mcrb.npz' --out-dir results_genie_outage")


if __name__ == "__main__":
    run_mse_test_from_cli(print_usage)
