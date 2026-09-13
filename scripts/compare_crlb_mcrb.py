#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import types
from datetime import datetime

import numpy as np
import torch
from numpy import arange
from numpy.random import mtrand

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.delay_simulation import (
    MSE_SECONDS2_TO_NS2,
    RunConfig,
    generate_ask_pilot,
    generate_bpsk_pilot,
    generate_one_hot_pilot,
    generate_ZC_pilot,
)
from src.utils import compute_marginalized_crlb_curve, compute_mcrb_curve

ML_K60_N90_CHECKPOINT = (
    "state_dict/k60N90_f0.100_P1_taumax[15.0]_gamma0.0_rate3_cnn_rate3_cnn_joint_training/"
    "encoder_k60N90_f0.100_P1_taumax[15.0]_gamma0.0_rate3_cnn_rate3_cnn_joint_training-150.pt"
)
ML_K60_N90_LABEL = "k60N90P4e150"
ML_K60_N90_ENCODER = "rate3_cnn"

PLOT_LABEL_ORDER = ["ML", "ZC", "BPSK", "OOK"]
PLOT_STYLES = {
    "ML": {"color": "tab:blue", "marker": "D"},
    "ZC": {"color": "tab:orange", "marker": "o"},
    "BPSK": {"color": "tab:green", "marker": "s"},
    "OOK": {"color": "tab:red", "marker": "^"},
}


def ordered_results_for_plot(results):
    order = {label: idx for idx, label in enumerate(PLOT_LABEL_ORDER)}
    return sorted(
        results,
        key=lambda result: (
            order.get(result["label"], len(order)),
            result["label"],
        ),
    )


def install_commpy_import_stub():
    if "commpy" in sys.modules:
        return

    commpy = types.ModuleType("commpy")
    channelcoding = types.ModuleType("commpy.channelcoding")
    convcode = types.ModuleType("commpy.channelcoding.convcode")
    turbo = types.ModuleType("commpy.channelcoding.turbo")
    interleavers = types.ModuleType("commpy.channelcoding.interleavers")
    modulation = types.ModuleType("commpy.modulation")

    class MissingCommpyDependency:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("commpy is required for this codec path.")

    def missing_commpy_function(*args, **kwargs):
        raise ModuleNotFoundError("commpy is required for this codec path.")

    convcode.Trellis = MissingCommpyDependency
    convcode.conv_encode = missing_commpy_function
    turbo.turbo_encode = missing_commpy_function
    interleavers.RandInterlv = MissingCommpyDependency

    commpy.channelcoding = channelcoding
    commpy.modulation = modulation
    channelcoding.convcode = convcode
    channelcoding.turbo = turbo
    channelcoding.interleavers = interleavers

    sys.modules["commpy"] = commpy
    sys.modules["commpy.channelcoding"] = channelcoding
    sys.modules["commpy.channelcoding.convcode"] = convcode
    sys.modules["commpy.channelcoding.turbo"] = turbo
    sys.modules["commpy.channelcoding.interleavers"] = interleavers
    sys.modules["commpy.modulation"] = modulation


def generate_ml_pilot(path_load_e, K, N, encoder_type=ML_K60_N90_ENCODER):
    try:
        import commpy  # noqa: F401
    except ModuleNotFoundError:
        install_commpy_import_stub()

    from src.encoders import import_enc
    from src.get_args import get_args

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        args = get_args()
    finally:
        sys.argv = original_argv
    args.block_len = K
    args.batch_size = 1
    args.code_rate_k = 1
    args.code_rate_n = 3
    args.enc_num_unit = 60
    args.enc_num_layer = 2
    args.encoder = encoder_type
    args.decoder = encoder_type
    if not hasattr(args, "no_cuda"):
        args.no_cuda = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_ML = import_enc(args)

    if args.is_interleave == 1:
        seed = np.random.randint(0, 1)
        rand_gen = mtrand.RandomState(seed)
        p_array1 = rand_gen.permutation(arange(args.block_len))
    elif args.is_interleave == 0:
        p_array1 = range(args.block_len)
    else:
        seed = np.random.randint(0, args.is_interleave)
        rand_gen = mtrand.RandomState(seed)
        p_array1 = rand_gen.permutation(arange(args.block_len))

    print("using random interleaver", p_array1)

    encoder = encoder_ML(args, p_array1)
    encoder = encoder.to(device)
    encoder.load_state_dict(torch.load(path_load_e, map_location=device))
    encoder.eval()

    m = np.random.randint(2, size=(1, K, 1))
    m = torch.from_numpy(m)
    m = m.to(device).float()

    x = encoder(m).detach().cpu()
    if x.numel() != 2 * N:
        raise ValueError(f"ML encoder produced {x.numel()} real values, expected {2 * N}.")
    x = x.view(N, 2)
    return torch.view_as_complex(x).numpy()


def build_pilot(
    pilot_type,
    N,
    K,
    rng_global,
    ml_checkpoint=ML_K60_N90_CHECKPOINT,
    ml_label=ML_K60_N90_LABEL,
    ml_encoder=ML_K60_N90_ENCODER,
):
    tag_extra_parts = []
    pilot_type_normalized = pilot_type.lower()

    if pilot_type_normalized == "zc":
        zc_root = 1
        x = generate_ZC_pilot(N, u=zc_root)
        tag_extra_parts.append(f"u{zc_root}")
    elif pilot_type_normalized == "bpsk":
        x = generate_bpsk_pilot(N, rng_global)
        x = x / np.linalg.norm(x) * np.sqrt(N)
    elif pilot_type_normalized in {"ook", "ask"}:
        x = generate_ask_pilot(N, rng_global)
        x = x / np.linalg.norm(x) * np.sqrt(N)
    elif pilot_type_normalized == "onehot":
        x = generate_one_hot_pilot(N, rng_global)
    elif pilot_type_normalized == "ml":
        tag_extra_parts.append(f"model{ml_label}")
        x = generate_ml_pilot(ml_checkpoint, K, N, encoder_type=ml_encoder)
    else:
        raise ValueError("Unknown pilot type.")

    return x, tag_extra_parts


def pilot_label(pilot_type):
    labels = {
        "zc": "ZC",
        "bpsk": "BPSK",
        "ook": "OOK",
        "ml": "ML",
    }
    return labels.get(pilot_type.lower(), pilot_type)


def save_csv(path, snr_grid, crlb, mcrb, valid_counts, failed_counts):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "SNR_dB",
                "CRLB_ns2",
                "MCRB_ns2",
                "CRLB_over_MCRB",
                "valid_crlb_samples",
                "failed_crlb_samples",
            ],
        )
        writer.writeheader()
        for idx, snr in enumerate(snr_grid):
            ratio = crlb[idx] / mcrb[idx] if mcrb[idx] > 0 else np.nan
            writer.writerow({
                "SNR_dB": float(snr),
                "CRLB_ns2": crlb[idx] * MSE_SECONDS2_TO_NS2,
                "MCRB_ns2": mcrb[idx] * MSE_SECONDS2_TO_NS2,
                "CRLB_over_MCRB": ratio,
                "valid_crlb_samples": int(valid_counts[idx]),
                "failed_crlb_samples": int(failed_counts[idx]),
            })


def save_combined_csv(path, snr_grid, results):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "SNR_dB",
                "CRLB_ns2",
                "MCRB_ns2",
                "CRLB_over_MCRB",
                "valid_crlb_samples",
                "failed_crlb_samples",
            ],
        )
        writer.writeheader()
        for result in results:
            label = result["label"]
            crlb = result["crlb_seconds2"]
            mcrb = result["mcrb_seconds2"]
            valid_counts = result["valid_crlb_samples"]
            failed_counts = result["failed_crlb_samples"]
            for idx, snr in enumerate(snr_grid):
                ratio = crlb[idx] / mcrb[idx] if mcrb[idx] > 0 else np.nan
                writer.writerow({
                    "label": label,
                    "SNR_dB": float(snr),
                    "CRLB_ns2": crlb[idx] * MSE_SECONDS2_TO_NS2,
                    "MCRB_ns2": mcrb[idx] * MSE_SECONDS2_TO_NS2,
                    "CRLB_over_MCRB": ratio,
                    "valid_crlb_samples": int(valid_counts[idx]),
                    "failed_crlb_samples": int(failed_counts[idx]),
                })


def save_npz(
    path,
    snr_grid,
    crlb,
    mcrb,
    valid_counts,
    failed_counts,
    *,
    N,
    K,
    P,
    Ts,
    L_norm,
    zc_root,
    crlb_samples,
    mcrb_samples,
    seed,
    pilot_type="zc",
    label="ZC",
):
    ratio = np.divide(
        crlb,
        mcrb,
        out=np.full_like(crlb, np.nan, dtype=float),
        where=np.asarray(mcrb, dtype=float) > 0,
    )
    np.savez_compressed(
        path,
        method_tag="zc_marginalized_covariance_crlb_vs_mcrb",
        pilot_type=pilot_type,
        label=label,
        snr_db_grid=np.asarray(snr_grid, dtype=float),
        crlb_seconds2=np.asarray(crlb, dtype=float),
        crlb_ns2=np.asarray(crlb, dtype=float) * MSE_SECONDS2_TO_NS2,
        mcrb_seconds2=np.asarray(mcrb, dtype=float),
        mcrb_ns2=np.asarray(mcrb, dtype=float) * MSE_SECONDS2_TO_NS2,
        crlb_over_mcrb=ratio,
        valid_crlb_samples=np.asarray(valid_counts, dtype=int),
        failed_crlb_samples=np.asarray(failed_counts, dtype=int),
        N=int(N),
        K=int(K),
        P=int(P),
        Ts=float(Ts),
        L_norm=np.asarray(L_norm, dtype=float),
        zc_root=int(zc_root),
        crlb_n_samples=int(crlb_samples),
        mcrb_n_samples=int(mcrb_samples),
        rng_seed=int(seed),
    )


def save_combined_npz(
    path,
    snr_grid,
    results,
    *,
    N,
    K,
    P,
    Ts,
    L_norm,
    zc_root,
    crlb_samples,
    mcrb_samples,
    seed,
    pilot_seed,
):
    labels = np.asarray([result["label"] for result in results])
    crlb = np.vstack([result["crlb_seconds2"] for result in results])
    mcrb = np.vstack([result["mcrb_seconds2"] for result in results])
    valid_counts = np.vstack([result["valid_crlb_samples"] for result in results])
    failed_counts = np.vstack([result["failed_crlb_samples"] for result in results])
    ratio = np.divide(
        crlb,
        mcrb,
        out=np.full_like(crlb, np.nan, dtype=float),
        where=mcrb > 0,
    )
    np.savez_compressed(
        path,
        method_tag="all_sequences_marginalized_covariance_crlb_vs_mcrb",
        labels=labels,
        snr_db_grid=np.asarray(snr_grid, dtype=float),
        crlb_seconds2=crlb,
        crlb_ns2=crlb * MSE_SECONDS2_TO_NS2,
        mcrb_seconds2=mcrb,
        mcrb_ns2=mcrb * MSE_SECONDS2_TO_NS2,
        crlb_over_mcrb=ratio,
        valid_crlb_samples=valid_counts,
        failed_crlb_samples=failed_counts,
        N=int(N),
        K=int(K),
        P=int(P),
        Ts=float(Ts),
        L_norm=np.asarray(L_norm, dtype=float),
        zc_root=int(zc_root),
        crlb_n_samples=int(crlb_samples),
        mcrb_n_samples=int(mcrb_samples),
        rng_seed=int(seed),
        pilot_seed=int(pilot_seed),
    )


def save_plot(path, snr_grid, crlb, mcrb):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 10

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.semilogy(
        snr_grid,
        crlb * MSE_SECONDS2_TO_NS2,
        marker="o",
        markerfacecolor="none",
        linewidth=1.6,
        label="Conventional CRLB",
    )
    ax.semilogy(
        snr_grid,
        mcrb * MSE_SECONDS2_TO_NS2,
        marker="s",
        markerfacecolor="none",
        linestyle="--",
        linewidth=1.5,
        label="MCRB",
    )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Average delay MSE bound (ns$^2$)")
    ax.set_title("ZC marginalized CRLB vs MCRB")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_combined_plot(path, snr_grid, results):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 10

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for result in ordered_results_for_plot(results):
        label = result["label"]
        style = PLOT_STYLES.get(label, {"color": None, "marker": "o"})
        ax.semilogy(
            snr_grid,
            result["crlb_seconds2"] * MSE_SECONDS2_TO_NS2,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="none",
            linewidth=1.5,
            label=f"{label} CRLB",
        )
        ax.semilogy(
            snr_grid,
            result["mcrb_seconds2"] * MSE_SECONDS2_TO_NS2,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="none",
            linestyle="--",
            linewidth=1.2,
            label=f"{label} MCRB",
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Average delay MSE bound (ns$^2$)")
    ax.set_title("All sequences marginalized CRLB vs MCRB")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def parse_pilot_types(raw):
    if raw.strip().lower() == "all":
        return ["zc", "bpsk", "ook", "ml"]
    pilot_types = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not pilot_types:
        raise ValueError("--pilot-types must include at least one pilot type.")
    return pilot_types


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the marginalized Gaussian CRLB to the existing MCRB for a ZC pilot."
    )
    parser.add_argument("--N", type=int, default=90)
    parser.add_argument("--K", type=int, default=60)
    parser.add_argument(
        "--P",
        type=int,
        default=None,
        help="Number of targets. Defaults to len(--tau-max), or 1 when --tau-max has one value.",
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        nargs="+",
        default=[15.0],
        help=(
            "One or more normalized delay-support upper bounds. "
            "Examples: --tau-max 15 or --tau-max 5 10 15."
        ),
    )
    parser.add_argument("--Ts", type=float, default=1e-9)
    parser.add_argument("--zc-root", type=int, default=1)
    parser.add_argument(
        "--pilot-types",
        default="all",
        help="Comma-separated pilot list among zc,bpsk,ook,ml, or 'all'.",
    )
    parser.add_argument("--pilot-seed", type=int, default=2026)
    parser.add_argument("--snr-start", type=float, default=10.0)
    parser.add_argument("--snr-stop", type=float, default=40.0)
    parser.add_argument("--snr-points", type=int, default=7)
    parser.add_argument("--crlb-samples", type=int, default=200)
    parser.add_argument("--mcrb-samples", type=int, default=200)
    parser.add_argument("--mcrb-batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=162)
    parser.add_argument("--out-dir", default=os.path.join("results_crlb"))
    return parser.parse_args()


def main():
    args = parse_args()
    snr_grid = np.linspace(args.snr_start, args.snr_stop, args.snr_points)
    cfg = RunConfig(snr_db_grid=snr_grid, n_mc=args.crlb_samples, rng_seed=args.seed)
    tau_max_values = np.asarray(args.tau_max, dtype=float).reshape(-1)
    if np.any(tau_max_values <= 0):
        raise ValueError("--tau-max values must be positive.")
    if args.P is None:
        P = tau_max_values.size
        L_norm = tau_max_values
    elif tau_max_values.size == 1:
        P = args.P
        L_norm = np.full(P, tau_max_values[0], dtype=float)
    elif args.P == tau_max_values.size:
        P = args.P
        L_norm = tau_max_values
    else:
        raise ValueError(
            f"--P={args.P} is inconsistent with {tau_max_values.size} --tau-max values. "
            "Either omit --P or pass one tau-max value to replicate."
        )
    requested_pilot_types = parse_pilot_types(args.pilot_types)
    rng_global = np.random.default_rng(args.pilot_seed)
    results = []

    for pilot_type in requested_pilot_types:
        label = pilot_label(pilot_type)
        print(f"Pilot type: {pilot_type}")
        if pilot_type == "zc":
            x = generate_ZC_pilot(args.N, u=args.zc_root)
        else:
            if pilot_type == "ml":
                install_commpy_import_stub()
            x, _ = build_pilot(
                pilot_type,
                args.N,
                args.K,
                rng_global,
                ml_checkpoint=ML_K60_N90_CHECKPOINT,
                ml_label=ML_K60_N90_LABEL,
            )

        print("Computing marginalized covariance CRLB...")
        crlb, valid_counts, failed_counts = compute_marginalized_crlb_curve(
            cfg=cfg,
            x=x,
            T=args.Ts,
            L_norm=L_norm,
            n_samples=args.crlb_samples,
        )

        print("Computing existing MCRB...")
        mcrb = compute_mcrb_curve(
            cfg=cfg,
            x=x,
            L_norm=L_norm,
            T=args.Ts,
            n_samples=args.mcrb_samples,
            batch_size=args.mcrb_batch_size,
            device="auto",
        )
        results.append({
            "label": label,
            "crlb_seconds2": crlb,
            "mcrb_seconds2": mcrb,
            "valid_crlb_samples": valid_counts,
            "failed_crlb_samples": failed_counts,
        })

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pilot_tag = "all_sequences" if requested_pilot_types == ["zc", "bpsk", "ook", "ml"] else "-".join(requested_pilot_types)
    stem = (
        f"{pilot_tag}_crlb_mcrb_P{P}_N{args.N}_taumax{'-'.join(f'{value:g}' for value in L_norm)}_"
        f"samples{args.crlb_samples}_{timestamp}"
    )
    csv_path = os.path.join(args.out_dir, f"{stem}.csv")
    npz_path = os.path.join(args.out_dir, f"{stem}.npz")
    png_path = os.path.join(args.out_dir, f"{stem}.png")

    if len(results) == 1:
        result = results[0]
        save_csv(
            csv_path,
            snr_grid,
            result["crlb_seconds2"],
            result["mcrb_seconds2"],
            result["valid_crlb_samples"],
            result["failed_crlb_samples"],
        )
        save_npz(
            npz_path,
            snr_grid,
            result["crlb_seconds2"],
            result["mcrb_seconds2"],
            result["valid_crlb_samples"],
            result["failed_crlb_samples"],
            N=args.N,
            K=args.K,
            P=P,
            Ts=args.Ts,
            L_norm=L_norm,
            zc_root=args.zc_root,
            crlb_samples=args.crlb_samples,
            mcrb_samples=args.mcrb_samples,
            seed=args.seed,
            pilot_type=requested_pilot_types[0],
            label=result["label"],
        )
        save_plot(png_path, snr_grid, result["crlb_seconds2"], result["mcrb_seconds2"])
    else:
        save_combined_csv(csv_path, snr_grid, results)
        save_combined_npz(
            npz_path,
            snr_grid,
            results,
            N=args.N,
            K=args.K,
            P=P,
            Ts=args.Ts,
            L_norm=L_norm,
            zc_root=args.zc_root,
            crlb_samples=args.crlb_samples,
            mcrb_samples=args.mcrb_samples,
            seed=args.seed,
            pilot_seed=args.pilot_seed,
        )
        save_combined_plot(png_path, snr_grid, results)

    print("Saved CSV:", csv_path)
    print("Saved NPZ:", npz_path)
    print("Saved plot:", png_path)
    print("")
    for result in results:
        print(result["label"])
        print("SNR(dB)   CRLB(ns^2)   MCRB(ns^2)   CRLB/MCRB")
        for snr, crlb_value, mcrb_value in zip(
            snr_grid,
            result["crlb_seconds2"],
            result["mcrb_seconds2"],
        ):
            ratio = crlb_value / mcrb_value if mcrb_value > 0 else np.nan
            print(
                f"{snr:6.1f}   "
                f"{crlb_value * MSE_SECONDS2_TO_NS2:.6e}   "
                f"{mcrb_value * MSE_SECONDS2_TO_NS2:.6e}   "
                f"{ratio:.6e}"
            )
        print("")


if __name__ == "__main__":
    main()
