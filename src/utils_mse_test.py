#!/usr/bin/env python3
import os
import sys
import glob
import re
from datetime import datetime

import numpy as np

from MSE_sim_Brent_wLB import (
    ML_K60_N90_CHECKPOINT,
    ML_K60_N90_LABEL,
    build_pilot,
    build_run_tag,
    choose_brent_xatol_from_mcrb,
    compute_mcrb_curve,
    infer_output_dir_from_pattern,
    load_npz_scalar_text,
    make_experiment_configs,
    method_label_from_tag,
    pilot_label,
    plot_all_median_error_vs_mcrb,
    plot_all_mse_vs_mcrb,
    save_mcrb_sidecar,
    save_summary_with_mcrb,
)
from src.delay_simulation import (
    MSE_SECONDS2_TO_NS2,
    RunConfig,
    best_permutation_metrics,
    build_single_path_hypothesis_vector,
    compute_score_vector,
    pick_top_p_from_scores,
    pick_top_p_local_maxima,
    refine_delay_estimates_brent,
    sample_theta_true,
    sample_y_given_theta,
    tag_number,
)

PLOT_LABEL_ORDER = ["ML", "ZC", "BPSK", "OOK"]
PLOT_STYLES = {
    "ML": {"color": "tab:blue", "marker": "D"},
    "ZC": {"color": "tab:orange", "marker": "o"},
    "BPSK": {"color": "tab:green", "marker": "s"},
    "OOK": {"color": "tab:red", "marker": "^"},
}
PLOT_DISPLAY_LABELS = {
    "ML": "CNN",
    "OOK": "1H-OOK",
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


def parse_value(flag, default, cast):
    env_name_by_flag = {
        "--N": "N",
        "--K": "K",
        "--P": "P",
        "--tau-max": "TAU_MAX_VALUES",
        "--estimator": "ESTIMATOR",
        "--sic-min-separation-ns": "SIC_MIN_SEPARATION_NS",
        "--sic-refine-iters": "SIC_REFINE_ITERS",
        "--out-dir": "OUT_DIR",
        "--snr-db": "SNR_DB",
        "--snr-min-db": "SNR_MIN_DB",
        "--snr-max-db": "SNR_MAX_DB",
        "--snr-points": "SNR_POINTS",
    }
    for idx, arg in enumerate(sys.argv[1:], start=1):
        if arg == flag:
            if idx + 1 >= len(sys.argv) or sys.argv[idx + 1].startswith("--"):
                raise ValueError(f"{flag} requires a value.")
            return cast(sys.argv[idx + 1])
        if arg.startswith(f"{flag}="):
            raw = arg.split("=", 1)[1]
            if raw == "":
                raise ValueError(f"{flag} requires a value.")
            return cast(raw)
    env_name = env_name_by_flag.get(flag)
    if env_name is not None and os.environ.get(env_name):
        return cast(os.environ[env_name].split()[0])
    return default


def parse_pilot_types():
    raw = parse_value("--pilot-types", None, str)
    if raw is None:
        return None
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def parse_tau_max_values():
    flag_idx = None
    values = []
    start_idx = None
    for idx, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--tau-max":
            flag_idx = idx
            start_idx = idx + 1
            break
        if arg.startswith("--tau-max="):
            flag_idx = idx
            raw = arg.split("=", 1)[1]
            values.extend(item for item in raw.split(",") if item)
            start_idx = idx + 1
            break

    if flag_idx is None:
        raw = os.environ.get("TAU_MAX_VALUES")
        if not raw:
            return None
        values = []
        for item in raw.replace(",", " ").split():
            if item:
                values.append(item)
        if not values:
            return None
        values = np.asarray([float(item) for item in values], dtype=float)
        if np.any(values <= 0):
            raise ValueError("TAU_MAX_VALUES values must be positive.")
        return values

    idx = start_idx
    while idx < len(sys.argv) and not sys.argv[idx].startswith("--"):
        values.extend(item for item in sys.argv[idx].split(",") if item)
        idx += 1
    if not values:
        raise ValueError("--tau-max requires at least one value.")
    values = np.asarray([float(item) for item in values], dtype=float)
    if np.any(values <= 0):
        raise ValueError("--tau-max values must be positive.")
    return values


def parse_snr_db_grid():
    snr_db = parse_value("--snr-db", None, float)
    if snr_db is not None:
        return np.asarray([snr_db], dtype=float)

    snr_min_db = parse_value("--snr-min-db", 10.0, float)
    snr_max_db = parse_value("--snr-max-db", 40.0, float)
    snr_points = parse_value("--snr-points", 7, int)
    if snr_points < 1:
        raise ValueError("--snr-points must be at least 1.")
    return np.linspace(snr_min_db, snr_max_db, snr_points)


def apply_geometry_overrides(experiment_configs, P_override=None, tau_max_values=None):
    if P_override is None and tau_max_values is None:
        return experiment_configs

    if tau_max_values is None:
        tau_max_values = np.asarray(experiment_configs[0]["L_norm"], dtype=float)
    else:
        tau_max_values = np.asarray(tau_max_values, dtype=float).reshape(-1)

    if P_override is None:
        L_norm_configs = [tau_max_values]
    elif int(P_override) <= 0:
        raise ValueError("--P must be positive.")
    elif tau_max_values.size == 1:
        L_norm_configs = [np.full(int(P_override), float(tau_max_values[0]), dtype=float)]
    elif int(P_override) == 1:
        L_norm_configs = [np.asarray([value], dtype=float) for value in tau_max_values]
    elif tau_max_values.size % int(P_override) == 0:
        L_norm_configs = [
            row.copy()
            for row in tau_max_values.reshape((-1, int(P_override)))
        ]
    else:
        raise ValueError(
            f"--P={P_override} is inconsistent with {tau_max_values.size} --tau-max values. "
            "Pass one value to replicate, exactly P values for one geometry, "
            "or a multiple of P values for multiple geometries."
        )

    updated_configs = []
    for L_norm in L_norm_configs:
        for experiment in experiment_configs:
            updated = dict(experiment)
            updated["L_norm"] = np.asarray(L_norm, dtype=float).copy()
            updated_configs.append(updated)
    return updated_configs


def apply_dimension_overrides(experiment_configs, N_override=None, K_override=None):
    if N_override is None and K_override is None:
        return experiment_configs
    if N_override is not None and int(N_override) <= 0:
        raise ValueError("--N must be positive.")
    if K_override is not None and int(K_override) <= 0:
        raise ValueError("--K must be positive.")

    updated_configs = []
    for experiment in experiment_configs:
        updated = dict(experiment)
        if N_override is not None:
            updated["N"] = int(N_override)
        if K_override is not None:
            updated["K"] = int(K_override)
        updated_configs.append(updated)
    return updated_configs


def make_results_csv(tag, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(out_dir, f"delay_score_{tag}_{timestamp}.csv")
    with open(filename, "w") as f:
        f.write(
            "method,SNR_dB,MeanSE_all_ns2,MeanSE_nonoutage_ns2,"
            "OutageProbability,MeanNMSE_all,MeanNMSE_nonoutage,n_mc,n_nonoutage,n_outage\n"
        )
    return filename


def write_outage_stats_csv(path, method_tag, stats):
    with open(path, "w") as f:
        f.write(
            "method,SNR_dB,MeanSE_all_ns2,MeanSE_nonoutage_ns2,MedianSE_all_ns2,"
            "MedianSE_nonoutage_ns2,P10_SE_all_ns2,P25_SE_all_ns2,P75_SE_all_ns2,"
            "P90_SE_all_ns2,P10_SE_nonoutage_ns2,P25_SE_nonoutage_ns2,"
            "P75_SE_nonoutage_ns2,P90_SE_nonoutage_ns2,OutageProbability,"
            "MeanNMSE_all,MeanNMSE_nonoutage,MedianNMSE_all,MedianNMSE_nonoutage,"
            "n_mc,n_nonoutage,n_outage,outage_threshold_ns\n"
        )
        for item in stats:
            all_pct = item["mse_percentiles"] * MSE_SECONDS2_TO_NS2
            non_pct = item["mse_percentiles_nonoutage"] * MSE_SECONDS2_TO_NS2
            f.write(
                f"{method_tag},{item['snr_db']:.2f},"
                f"{item['mse_mean'] * MSE_SECONDS2_TO_NS2:.8e},"
                f"{item['mse_mean_nonoutage'] * MSE_SECONDS2_TO_NS2:.8e},"
                f"{item['median_mse'] * MSE_SECONDS2_TO_NS2:.8e},"
                f"{item['median_mse_nonoutage'] * MSE_SECONDS2_TO_NS2:.8e},"
                f"{all_pct[0]:.8e},{all_pct[1]:.8e},{all_pct[3]:.8e},{all_pct[4]:.8e},"
                f"{non_pct[0]:.8e},{non_pct[1]:.8e},{non_pct[3]:.8e},{non_pct[4]:.8e},"
                f"{item['outage_probability']:.8e},"
                f"{item['nmse_mean']:.8e},{item['nmse_mean_nonoutage']:.8e},"
                f"{item['median_nmse']:.8e},{item['median_nmse_nonoutage']:.8e},"
                f"{item['n_mc']},{item['n_nonoutage']},{item['n_outage']},"
                f"{item['outage_threshold_ns']:.8e}\n"
            )


def percentile_or_nan(values, percentiles):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.full(len(percentiles), np.nan, dtype=float)
    return np.percentile(values, percentiles)


def mean_or_nan(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def estimate_path_gains(y, tau_abs_hat, x, Ts):
    tau_abs_hat = np.asarray(tau_abs_hat, dtype=float).reshape(-1)
    if tau_abs_hat.size == 0:
        return np.array([], dtype=complex), np.zeros_like(y, dtype=complex)

    A = np.column_stack([
        build_single_path_hypothesis_vector(tau_abs, x, Ts)
        for tau_abs in tau_abs_hat
    ])
    h_hat, *_ = np.linalg.lstsq(A, y, rcond=None)
    return h_hat, A @ h_hat


def estimate_delays_successive_cancellation(
    y,
    delay_abs_grid,
    Ts,
    x,
    P,
    *,
    use_brent=True,
    brent_xatol=None,
    brent_maxiter=50,
    min_separation=None,
    refine_iters=3,
):
    delay_abs_grid = np.asarray(delay_abs_grid, dtype=float)
    if delay_abs_grid.size == 0:
        return np.array([], dtype=float)

    if min_separation is None:
        min_separation = np.min(np.diff(np.unique(delay_abs_grid))) if delay_abs_grid.size > 1 else 0.0
    min_separation = float(min_separation)

    residual = np.asarray(y, dtype=complex).copy()
    detected = []
    for _ in range(P):
        scores = compute_score_vector(residual, delay_abs_grid, Ts, x)
        if detected and min_separation > 0:
            for tau_abs in detected:
                scores[np.abs(delay_abs_grid - tau_abs) < min_separation] = -np.inf

        if not np.any(np.isfinite(scores)):
            break

        selected_idx = np.asarray([int(np.nanargmax(scores))], dtype=int)
        if use_brent:
            tau_new = refine_delay_estimates_brent(
                selected_idx=selected_idx,
                delay_abs_grid=delay_abs_grid,
                y=residual,
                Ts=Ts,
                x=x,
                xatol=brent_xatol,
                maxiter=brent_maxiter,
            )
        else:
            tau_new = delay_abs_grid[selected_idx]

        detected.append(float(tau_new[0]))
        _, reconstructed = estimate_path_gains(y, detected, x, Ts)
        residual = y - reconstructed

    if refine_iters > 0 and len(detected) == P:
        detected = refine_sic_delay_estimates(
            y,
            detected,
            delay_abs_grid,
            Ts,
            x,
            n_iters=refine_iters,
            brent_xatol=brent_xatol,
            brent_maxiter=brent_maxiter,
        ).tolist()

    if len(detected) < P:
        scores = compute_score_vector(residual, delay_abs_grid, Ts, x)
        if detected and min_separation > 0:
            for tau_abs in detected:
                scores[np.abs(delay_abs_grid - tau_abs) < min_separation] = -np.inf
        fallback_count = P - len(detected)
        fallback_tau, _ = pick_top_p_from_scores(
            delay_abs_grid,
            scores,
            fallback_count,
            min_separation=min_separation if min_separation > 0 else None,
        )
        detected.extend(float(tau_abs) for tau_abs in np.asarray(fallback_tau).reshape(-1))

    return np.sort(np.asarray(detected[:P], dtype=float))


def refine_sic_delay_estimates(
    y,
    tau_abs_hat,
    delay_abs_grid,
    Ts,
    x,
    *,
    n_iters=3,
    brent_xatol=None,
    brent_maxiter=50,
):
    tau_abs_hat = np.asarray(tau_abs_hat, dtype=float).reshape(-1).copy()
    delay_abs_grid = np.asarray(delay_abs_grid, dtype=float)
    if tau_abs_hat.size == 0 or delay_abs_grid.size < 2:
        return tau_abs_hat

    for _ in range(int(n_iters)):
        h_hat, reconstructed = estimate_path_gains(y, tau_abs_hat, x, Ts)
        for path_idx in range(tau_abs_hat.size):
            current_atom = build_single_path_hypothesis_vector(tau_abs_hat[path_idx], x, Ts)
            path_observation = y - reconstructed + h_hat[path_idx] * current_atom
            selected_idx = np.asarray([
                int(np.argmin(np.abs(delay_abs_grid - tau_abs_hat[path_idx])))
            ])
            tau_refined = refine_delay_estimates_brent(
                selected_idx=selected_idx,
                delay_abs_grid=delay_abs_grid,
                y=path_observation,
                Ts=Ts,
                x=x,
                xatol=brent_xatol,
                maxiter=brent_maxiter,
            )
            tau_abs_hat[path_idx] = float(tau_refined[0])
            h_hat, reconstructed = estimate_path_gains(y, tau_abs_hat, x, Ts)

    return np.sort(tau_abs_hat)


def format_snr_grid(snr_grid):
    return "[" + ", ".join(f"{value:g}" for value in np.asarray(snr_grid, dtype=float)) + "]"


def load_mcrb_sidecars(mcrb_pattern):
    mcrb_entries_by_label = {}
    if mcrb_pattern is None:
        return {}

    mcrb_paths = sorted(glob.glob(mcrb_pattern))
    if not mcrb_paths:
        print(f"Warning: no MCRB sidecars matched pattern: {mcrb_pattern}")
        return {}

    required_keys = {"label", "method_tag", "snr_db_grid", "mcrb_seconds2"}
    for path in mcrb_paths:
        data = np.load(path, allow_pickle=False)
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            print(f"Warning: skipping non-MCRB sidecar {path}; missing keys: {sorted(missing_keys)}")
            continue
        label = load_npz_scalar_text(data["label"])
        method_tag = load_npz_scalar_text(data["method_tag"])
        if "mcrb_n_samples" in data:
            n_samples = int(np.asarray(data["mcrb_n_samples"]).item())
        else:
            match = re.search(r"_mcrb(\d+)", method_tag)
            n_samples = int(match.group(1)) if match else -1
        candidate = {
            "snr_grid": np.asarray(data["snr_db_grid"], dtype=float),
            "mcrb": np.asarray(data["mcrb_seconds2"], dtype=float),
            "n_samples": n_samples,
            "path": path,
        }
        mcrb_entries_by_label.setdefault(label, []).append(candidate)

    mcrb_by_label = {}
    for label, entries in mcrb_entries_by_label.items():
        reference = entries[0]
        for entry in entries[1:]:
            if not np.allclose(reference["snr_grid"], entry["snr_grid"]):
                raise ValueError(
                    f"MCRB SNR grid mismatch for {label}:\n"
                    f"  {reference['path']}\n"
                    f"  {entry['path']}"
                )
        weights = np.asarray(
            [entry["n_samples"] if entry["n_samples"] > 0 else 1 for entry in entries],
            dtype=float,
        )
        mcrb_stack = np.vstack([entry["mcrb"] for entry in entries])
        mcrb_by_label[label] = {
            "snr_grid": reference["snr_grid"],
            "mcrb": np.average(mcrb_stack, axis=0, weights=weights),
            "n_samples": int(sum(max(entry["n_samples"], 0) for entry in entries)),
            "path": ", ".join(entry["path"] for entry in entries),
        }
        if len(entries) > 1:
            print(
                f"Averaged {len(entries)} MCRB sidecars for {label} "
                f"({mcrb_by_label[label]['n_samples']} samples total)."
            )

    return mcrb_by_label


def plot_genie_outage_combined(results, filepath, yscale="log"):
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping genie outage MSE plot.")
        return False

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 8

    if yscale not in {"log", "linear"}:
        raise ValueError("yscale must be 'log' or 'linear'.")

    plt.figure(figsize=(5.2, 3.35))
    for result in ordered_results_for_plot(results):
        label = result["label"]
        style = PLOT_STYLES.get(label, {"color": None, "marker": "o"})
        plot_fn = plt.semilogy if yscale == "log" else plt.plot
        plot_fn(
            result["snr_grid"],
            result["mse_nonoutage"] * MSE_SECONDS2_TO_NS2,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="none",
            linewidth=1.1,
            markersize=4.0,
        )
        plot_fn(
            result["snr_grid"],
            result["mse_all"] * MSE_SECONDS2_TO_NS2,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="none",
            linestyle="-.",
            linewidth=1.1,
            markersize=4.0,
        )
        mcrb = result.get("mcrb")
        if mcrb is not None:
            plot_fn(
                result["snr_grid"],
                mcrb * MSE_SECONDS2_TO_NS2,
                color=style["color"],
                marker=style["marker"],
                markerfacecolor="none",
                linestyle="--",
                linewidth=0.95,
                markersize=4.0,
            )

    plt.xlabel("SNR (dB)", fontsize=9)
    plt.ylabel("Squared delay error (ns$^2$)", fontsize=9)
    plt.tick_params(axis="both", which="major", labelsize=8)
    plt.grid(True, which="both", linestyle=":", linewidth=0.55)
    sequence_handles = []
    for label in PLOT_LABEL_ORDER:
        if not any(result["label"] == label for result in results):
            continue
        style = PLOT_STYLES.get(label, {"color": "black", "marker": "o"})
        sequence_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                markerfacecolor="none",
                linestyle="-",
                linewidth=1.1,
                markersize=4.0,
                label=PLOT_DISPLAY_LABELS.get(label, label),
            )
        )
    metric_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.1, label="Non-outage mean"),
        Line2D([0], [0], color="black", linestyle="-.", linewidth=1.1, label="All-trial mean"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.95, label="MCRB"),
    ]
    plt.legend(
        handles=sequence_handles + metric_handles,
        fontsize=7.5,
        ncol=3,
        frameon=True,
        columnspacing=0.65,
        handlelength=1.45,
        handletextpad=0.35,
        borderpad=0.25,
        labelspacing=0.25,
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_genie_outage_probability(results, filepath):
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping genie outage probability plot.")
        return False

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 10

    plt.figure(figsize=(7, 4.2))
    for result in ordered_results_for_plot(results):
        label = result["label"]
        style = PLOT_STYLES.get(label, {"color": None, "marker": "o"})
        plt.plot(
            result["snr_grid"],
            result["outage_probability"],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.5,
            label=label,
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Genie outage probability")
    plt.grid(True, linestyle=":")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return True


def plot_genie_outage_median_combined(results, filepath):
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping genie outage median plot.")
        return False

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 8

    plt.figure(figsize=(5.2, 3.35))
    for result in ordered_results_for_plot(results):
        label = result["label"]
        style = PLOT_STYLES.get(label, {"color": None, "marker": "o"})
        plt.semilogy(
            result["snr_grid"],
            result["mse_median_nonoutage"] * MSE_SECONDS2_TO_NS2,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="none",
            linewidth=1.1,
            markersize=4.0,
        )
        plt.semilogy(
            result["snr_grid"],
            result["mse_median_all"] * MSE_SECONDS2_TO_NS2,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="none",
            linestyle="-.",
            linewidth=1.1,
            markersize=4.0,
        )
        mcrb = result.get("mcrb")
        if mcrb is not None:
            plt.semilogy(
                result["snr_grid"],
                mcrb * MSE_SECONDS2_TO_NS2,
                color=style["color"],
                marker=style["marker"],
                markerfacecolor="none",
                linestyle="--",
                linewidth=0.95,
                markersize=4.0,
            )

    plt.xlabel("SNR (dB)", fontsize=9)
    plt.ylabel("Squared delay error (ns$^2$)", fontsize=9)
    plt.tick_params(axis="both", which="major", labelsize=8)
    plt.grid(True, which="both", linestyle=":", linewidth=0.55)
    sequence_handles = []
    for label in PLOT_LABEL_ORDER:
        if not any(result["label"] == label for result in results):
            continue
        style = PLOT_STYLES.get(label, {"color": "black", "marker": "o"})
        sequence_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                markerfacecolor="none",
                linestyle="-",
                linewidth=1.1,
                markersize=4.0,
                label=PLOT_DISPLAY_LABELS.get(label, label),
            )
        )
    metric_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.1, label="Non-outage median"),
        Line2D([0], [0], color="black", linestyle="-.", linewidth=1.1, label="All-trial median"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.95, label="MCRB"),
    ]
    plt.legend(
        handles=sequence_handles + metric_handles,
        fontsize=7.5,
        ncol=3,
        frameon=True,
        columnspacing=0.65,
        handlelength=1.45,
        handletextpad=0.35,
        borderpad=0.25,
        labelspacing=0.25,
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_genie_outage_ratio(results, filepath):
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping genie outage ratio plot.")
        return False

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 10

    plt.figure(figsize=(7, 4.4))
    for result in ordered_results_for_plot(results):
        mcrb = result.get("mcrb")
        if mcrb is None or not np.all(np.isfinite(mcrb)):
            continue
        label = result["label"]
        style = PLOT_STYLES.get(label, {"color": None, "marker": "o"})
        plt.semilogy(
            result["snr_grid"],
            result["mse_nonoutage"] / mcrb,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.4,
            label=f"{label} non-outage",
        )
        plt.semilogy(
            result["snr_grid"],
            result["mse_all"] / mcrb,
            color=style["color"],
            marker=style["marker"],
            linestyle="-.",
            linewidth=1.4,
            label=f"{label} all",
        )

    plt.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="MCRB")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Mean MSE / MCRB")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return True


def plot_genie_outage_mcrb(results, filepath):
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping MCRB-only plot.")
        return False

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 10

    plt.figure(figsize=(7, 4.2))
    plotted = False
    for result in ordered_results_for_plot(results):
        mcrb = result.get("mcrb")
        if mcrb is None or not np.all(np.isfinite(mcrb)):
            continue
        label = result["label"]
        style = PLOT_STYLES.get(label, {"color": None, "marker": "o"})
        plt.semilogy(
            result["snr_grid"],
            mcrb * MSE_SECONDS2_TO_NS2,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="none",
            linewidth=1.5,
            label=f"{label} MCRB",
        )
        plotted = True

    if not plotted:
        print("No finite MCRB values found; skipping MCRB-only plot.")
        plt.close()
        return False

    plt.xlabel("SNR (dB)")
    plt.ylabel("MCRB (ns$^2$)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return True


def plot_genie_outage_nonoutage_mse(results, filepath):
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping non-outage MSE plot.")
        return False

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 10

    plt.figure(figsize=(7.4, 4.6))
    for result in ordered_results_for_plot(results):
        label = result["label"]
        style = PLOT_STYLES.get(label, {"color": None, "marker": "o"})
        plt.semilogy(
            result["snr_grid"],
            result["mse_nonoutage"] * MSE_SECONDS2_TO_NS2,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.5,
            label=f"{label} non-outage",
        )
        mcrb = result.get("mcrb")
        if mcrb is not None and np.any(np.isfinite(mcrb)):
            plt.semilogy(
                result["snr_grid"],
                mcrb * MSE_SECONDS2_TO_NS2,
                color=style["color"],
                linestyle="--",
                linewidth=1.1,
                label=f"{label} MCRB",
            )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Non-outage squared delay error (ns$^2$)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    return True


def combine_genie_outage_trial_error_files(trial_error_pattern, mcrb_pattern=None, out_dir=None):
    trial_paths = sorted(glob.glob(trial_error_pattern))
    if not trial_paths:
        raise FileNotFoundError(f"No trial-error files matched pattern: {trial_error_pattern}")
    if out_dir is None:
        out_dir = infer_output_dir_from_pattern(trial_error_pattern, default="results_genie_outage")
    os.makedirs(out_dir, exist_ok=True)

    mcrb_by_label = load_mcrb_sidecars(mcrb_pattern)
    grouped = {}
    outage_thresholds = set()
    for path in trial_paths:
        data = np.load(path, allow_pickle=False)
        method_tag = load_npz_scalar_text(data["method_tag"])
        label = method_label_from_tag(method_tag)
        snr_grid = np.asarray(data["snr_db_grid"], dtype=float)
        trial_mse = np.asarray(data["trial_mse_seconds2"], dtype=float)
        trial_nmse = np.asarray(data["trial_nmse"], dtype=float)
        trial_outage = np.asarray(data["trial_is_outage"], dtype=bool)
        outage_thresholds.add(float(np.asarray(data["outage_threshold_ns"]).item()))

        if label not in grouped:
            grouped[label] = {
                "snr_grid": snr_grid,
                "trial_mse": [],
                "trial_nmse": [],
                "trial_outage": [],
                "paths": [],
            }
        elif not np.allclose(grouped[label]["snr_grid"], snr_grid):
            raise ValueError(
                f"SNR grid mismatch for {label} in {path}\n"
                f"  expected: {format_snr_grid(grouped[label]['snr_grid'])}\n"
                f"  found:    {format_snr_grid(snr_grid)}\n"
                f"  first matching file: {grouped[label]['paths'][0]}\n"
                "Use a narrower --combine-trial-errors glob so each method label has one SNR grid."
            )

        grouped[label]["trial_mse"].append(trial_mse)
        grouped[label]["trial_nmse"].append(trial_nmse)
        grouped[label]["trial_outage"].append(trial_outage)
        grouped[label]["paths"].append(path)

    if len(outage_thresholds) > 1:
        print(f"Warning: multiple outage thresholds found: {sorted(outage_thresholds)}")
    outage_threshold_ns = sorted(outage_thresholds)[0] if outage_thresholds else np.nan

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    combined_csv = os.path.join(out_dir, f"combined_genie_outage_stats_{timestamp}.csv")
    combined_npz = os.path.join(out_dir, f"combined_genie_outage_trial_errors_{timestamp}.npz")
    combined_mse_plot = os.path.join(out_dir, f"combined_genie_outage_mse_vs_mcrb_{timestamp}.png")
    combined_mse_linear_plot = os.path.join(out_dir, f"combined_genie_outage_mse_vs_mcrb_linear_{timestamp}.png")
    combined_median_plot = os.path.join(out_dir, f"combined_genie_outage_median_vs_mcrb_{timestamp}.png")
    combined_nonoutage_mse_plot = os.path.join(out_dir, f"combined_genie_outage_nonoutage_mse_vs_mcrb_{timestamp}.png")
    combined_mcrb_plot = os.path.join(out_dir, f"combined_genie_outage_mcrb_{timestamp}.png")
    combined_pout_plot = os.path.join(out_dir, f"combined_genie_outage_probability_{timestamp}.png")
    combined_ratio_plot = os.path.join(out_dir, f"combined_genie_outage_mse_over_mcrb_{timestamp}.png")

    results = []
    percentiles = np.array([10, 25, 50, 75, 90, 99], dtype=float)
    with open(combined_csv, "w") as f:
        f.write(
            "label,SNR_dB,n_trials,n_nonoutage,n_outage,OutageProbability,"
            "MeanSE_all_ns2,MeanSE_nonoutage_ns2,MedianSE_all_ns2,MedianSE_nonoutage_ns2,"
            "P90_SE_all_ns2,P90_SE_nonoutage_ns2,P99_SE_all_ns2,P99_SE_nonoutage_ns2,"
            "MeanNMSE_all,MeanNMSE_nonoutage,MCRB_ns2,MeanAll_over_MCRB,"
            "MeanNonOutage_over_MCRB,outage_threshold_ns\n"
        )

        for label, group in grouped.items():
            snr_grid = group["snr_grid"]
            trial_mse = np.concatenate(group["trial_mse"], axis=1)
            trial_nmse = np.concatenate(group["trial_nmse"], axis=1)
            trial_outage = np.concatenate(group["trial_outage"], axis=1)
            nonoutage = ~trial_outage
            n_trials = trial_mse.shape[1]

            mse_all = np.mean(trial_mse, axis=1)
            nmse_all = np.mean(trial_nmse, axis=1)
            mse_percentiles = np.percentile(trial_mse, percentiles, axis=1).T
            outage_probability = np.mean(trial_outage, axis=1)
            n_outage = np.sum(trial_outage, axis=1)
            n_nonoutage = np.sum(nonoutage, axis=1)

            mse_nonoutage = np.full_like(snr_grid, np.nan, dtype=float)
            nmse_nonoutage = np.full_like(snr_grid, np.nan, dtype=float)
            mse_percentiles_nonoutage = np.full((len(snr_grid), len(percentiles)), np.nan, dtype=float)
            for idx in range(len(snr_grid)):
                mask = nonoutage[idx]
                if np.any(mask):
                    mse_nonoutage[idx] = np.mean(trial_mse[idx, mask])
                    nmse_nonoutage[idx] = np.mean(trial_nmse[idx, mask])
                    mse_percentiles_nonoutage[idx] = np.percentile(trial_mse[idx, mask], percentiles)

            if label in mcrb_by_label:
                mcrb_entry = mcrb_by_label[label]
                if not np.allclose(mcrb_entry["snr_grid"], snr_grid):
                    raise ValueError(f"MCRB SNR grid mismatch for {label}: {mcrb_entry['path']}")
                mcrb = mcrb_entry["mcrb"]
            else:
                mcrb = np.full_like(snr_grid, np.nan, dtype=float)

            for idx, snr in enumerate(snr_grid):
                mcrb_val = mcrb[idx]
                f.write(
                    f"{label},{snr:.2f},{n_trials},{int(n_nonoutage[idx])},{int(n_outage[idx])},"
                    f"{outage_probability[idx]:.8e},"
                    f"{mse_all[idx] * MSE_SECONDS2_TO_NS2:.8e},"
                    f"{mse_nonoutage[idx] * MSE_SECONDS2_TO_NS2:.8e},"
                    f"{mse_percentiles[idx, 2] * MSE_SECONDS2_TO_NS2:.8e},"
                    f"{mse_percentiles_nonoutage[idx, 2] * MSE_SECONDS2_TO_NS2:.8e},"
                    f"{mse_percentiles[idx, 4] * MSE_SECONDS2_TO_NS2:.8e},"
                    f"{mse_percentiles_nonoutage[idx, 4] * MSE_SECONDS2_TO_NS2:.8e},"
                    f"{mse_percentiles[idx, 5] * MSE_SECONDS2_TO_NS2:.8e},"
                    f"{mse_percentiles_nonoutage[idx, 5] * MSE_SECONDS2_TO_NS2:.8e},"
                    f"{nmse_all[idx]:.8e},{nmse_nonoutage[idx]:.8e},"
                    f"{mcrb_val * MSE_SECONDS2_TO_NS2 if np.isfinite(mcrb_val) else np.nan:.8e},"
                    f"{mse_all[idx] / mcrb_val if np.isfinite(mcrb_val) and mcrb_val > 0 else np.nan:.8e},"
                    f"{mse_nonoutage[idx] / mcrb_val if np.isfinite(mcrb_val) and mcrb_val > 0 else np.nan:.8e},"
                    f"{outage_threshold_ns:.8e}\n"
                )

            results.append({
                "label": label,
                "snr_grid": snr_grid.copy(),
                "mse_all": mse_all.copy(),
                "mse_nonoutage": mse_nonoutage.copy(),
                "mse_median_all": mse_percentiles[:, 2].copy(),
                "mse_median_nonoutage": mse_percentiles_nonoutage[:, 2].copy(),
                "nmse_all": nmse_all.copy(),
                "nmse_nonoutage": nmse_nonoutage.copy(),
                "outage_probability": outage_probability.copy(),
                "mcrb": mcrb.copy(),
                "n_trials": int(n_trials),
                "n_nonoutage": n_nonoutage.copy(),
                "n_outage": n_outage.copy(),
                "outage_threshold_ns": float(outage_threshold_ns),
            })

    np.savez_compressed(
        combined_npz,
        labels=np.asarray([result["label"] for result in results]),
        source_files=np.asarray(trial_paths),
        outage_threshold_ns=float(outage_threshold_ns),
    )

    print("Combined genie outage stats saved to:", combined_csv)
    print("Combined source manifest saved to:", combined_npz)
    if plot_genie_outage_combined(results, combined_mse_plot):
        print("Combined genie outage MSE plot saved to:", combined_mse_plot)
    if plot_genie_outage_combined(results, combined_mse_linear_plot, yscale="linear"):
        print("Combined genie outage linear MSE plot saved to:", combined_mse_linear_plot)
    if plot_genie_outage_median_combined(results, combined_median_plot):
        print("Combined genie outage median plot saved to:", combined_median_plot)
    if plot_genie_outage_nonoutage_mse(results, combined_nonoutage_mse_plot):
        print("Combined genie outage non-outage MSE plot saved to:", combined_nonoutage_mse_plot)
    if plot_genie_outage_mcrb(results, combined_mcrb_plot):
        print("Combined genie outage MCRB plot saved to:", combined_mcrb_plot)
    if plot_genie_outage_probability(results, combined_pout_plot):
        print("Combined genie outage probability plot saved to:", combined_pout_plot)
    if plot_genie_outage_ratio(results, combined_ratio_plot):
        print("Combined genie outage ratio plot saved to:", combined_ratio_plot)
    return results


def run_score_delay_simulation_with_genie_outage(
    cfg,
    user_kwargs,
    method_tag,
    csv_path,
    outage_threshold_ns,
):
    rng = np.random.default_rng(cfg.rng_seed)
    L_norm = np.asarray(user_kwargs["L_norm"], dtype=float)
    Ts = float(user_kwargs["Ts"])
    delay_abs_grid = np.asarray(user_kwargs["delay_abs_grid"], dtype=float)
    min_separation = user_kwargs.get("min_separation", None)
    use_brent = user_kwargs.get("use_brent", True)
    brent_xatol = user_kwargs.get("brent_xatol", None)
    brent_maxiter = user_kwargs.get("brent_maxiter", 50)
    estimator = user_kwargs.get("estimator", "top_peaks")
    sic_min_separation = user_kwargs.get("sic_min_separation", min_separation)
    sic_refine_iters = user_kwargs.get("sic_refine_iters", 3)
    x = user_kwargs["pilot"]
    P = len(L_norm)

    threshold_seconds = float(outage_threshold_ns) * 1e-9
    percentiles = np.asarray(user_kwargs.get("error_percentiles", (10, 25, 50, 75, 90)), dtype=float)
    save_trial_errors = user_kwargs.get("save_trial_errors", True)

    mse_all = np.zeros(len(cfg.snr_db_grid), dtype=float)
    mse_nonoutage = np.zeros(len(cfg.snr_db_grid), dtype=float)
    nmse_all = np.zeros(len(cfg.snr_db_grid), dtype=float)
    nmse_nonoutage = np.zeros(len(cfg.snr_db_grid), dtype=float)
    outage_prob = np.zeros(len(cfg.snr_db_grid), dtype=float)
    stats = []

    trial_mse_by_snr = np.zeros((len(cfg.snr_db_grid), cfg.n_mc), dtype=float) if save_trial_errors else None
    trial_nmse_by_snr = np.zeros((len(cfg.snr_db_grid), cfg.n_mc), dtype=float) if save_trial_errors else None
    trial_outage_by_snr = np.zeros((len(cfg.snr_db_grid), cfg.n_mc), dtype=bool) if save_trial_errors else None
    trial_abs_error_by_snr = np.zeros((len(cfg.snr_db_grid), cfg.n_mc), dtype=float) if save_trial_errors else None

    for si, snr_db in enumerate(cfg.snr_db_grid):
        print(f"[{method_tag}] Running SNR = {snr_db:.1f} dB")
        trial_mse = np.zeros(cfg.n_mc, dtype=float)
        trial_nmse = np.zeros(cfg.n_mc, dtype=float)
        trial_abs_error = np.zeros(cfg.n_mc, dtype=float)
        trial_outage = np.zeros(cfg.n_mc, dtype=bool)

        for trial_idx in range(cfg.n_mc):
            tau_abs_true = sample_theta_true(rng, L_norm, Ts)
            y = sample_y_given_theta(tau_abs_true, float(snr_db), rng, user_kwargs)
            if estimator == "sic":
                delay_abs_hat = estimate_delays_successive_cancellation(
                    y,
                    delay_abs_grid,
                    Ts,
                    x,
                    P,
                    use_brent=use_brent,
                    brent_xatol=brent_xatol,
                    brent_maxiter=brent_maxiter,
                    min_separation=sic_min_separation,
                    refine_iters=sic_refine_iters,
                )
            else:
                scores = compute_score_vector(y, delay_abs_grid, Ts, x)

                delay_abs_hat, selected_idx = pick_top_p_local_maxima(delay_abs_grid, scores, P)
                if tau_abs_true.size > delay_abs_hat.size:
                    num_delay_diff = tau_abs_true.size - delay_abs_hat.size
                    delay_abs_hat_supp, selected_idx_supp = pick_top_p_from_scores(
                        delay_abs_grid,
                        scores,
                        P,
                        min_separation=min_separation,
                    )
                    delay_abs_hat_added = []
                    selected_idx_added = []
                    for i_p in range(P):
                        if not np.any(selected_idx == selected_idx_supp[-1 - i_p]):
                            delay_abs_hat_added.append(delay_abs_hat_supp[-1 - i_p])
                            selected_idx_added.append(selected_idx_supp[-1 - i_p])
                        if len(delay_abs_hat_added) == num_delay_diff:
                            break
                    delay_abs_hat = np.concatenate((delay_abs_hat, np.asarray(delay_abs_hat_added)))
                    selected_idx = np.concatenate((selected_idx, np.asarray(selected_idx_added, dtype=int)))

                if use_brent:
                    delay_abs_hat = refine_delay_estimates_brent(
                        selected_idx=selected_idx,
                        delay_abs_grid=delay_abs_grid,
                        y=y,
                        Ts=Ts,
                        x=x,
                        xatol=brent_xatol,
                        maxiter=brent_maxiter,
                    )

            mse, nmse, delay_abs_hat = best_permutation_metrics(tau_abs_true, delay_abs_hat)
            abs_error = float(np.max(np.abs(tau_abs_true.reshape(-1) - delay_abs_hat.reshape(-1))))
            is_outage = abs_error > threshold_seconds

            trial_mse[trial_idx] = mse
            trial_nmse[trial_idx] = nmse
            trial_abs_error[trial_idx] = abs_error
            trial_outage[trial_idx] = is_outage

            if trial_idx < 3:
                print("True delay (ns):      ", " ".join(f"{v:.3f}" for v in tau_abs_true * 1e9))
                print("Estimated delay (ns): ", " ".join(f"{v:.3f}" for v in delay_abs_hat * 1e9))
                print(f"Abs error (ns):       {abs_error * 1e9:.3f}")
                print(f"Genie outage:         {is_outage}")
                print("----")

        nonoutage_mask = ~trial_outage
        n_nonoutage = int(np.sum(nonoutage_mask))
        n_outage = int(np.sum(trial_outage))
        mse_all[si] = float(np.mean(trial_mse))
        nmse_all[si] = float(np.mean(trial_nmse))
        mse_nonoutage[si] = mean_or_nan(trial_mse[nonoutage_mask])
        nmse_nonoutage[si] = mean_or_nan(trial_nmse[nonoutage_mask])
        outage_prob[si] = n_outage / cfg.n_mc

        mse_pct = percentile_or_nan(trial_mse, percentiles)
        nmse_pct = percentile_or_nan(trial_nmse, percentiles)
        mse_pct_nonoutage = percentile_or_nan(trial_mse[nonoutage_mask], percentiles)
        nmse_pct_nonoutage = percentile_or_nan(trial_nmse[nonoutage_mask], percentiles)
        stat = {
            "snr_db": float(snr_db),
            "mse_mean": float(mse_all[si]),
            "nmse_mean": float(nmse_all[si]),
            "percentiles": percentiles.copy(),
            "mse_percentiles": mse_pct,
            "nmse_percentiles": nmse_pct,
            "median_mse": float(mse_pct[2]),
            "median_nmse": float(nmse_pct[2]),
            "mse_mean_nonoutage": float(mse_nonoutage[si]),
            "nmse_mean_nonoutage": float(nmse_nonoutage[si]),
            "mse_percentiles_nonoutage": mse_pct_nonoutage,
            "nmse_percentiles_nonoutage": nmse_pct_nonoutage,
            "median_mse_nonoutage": float(mse_pct_nonoutage[2]),
            "median_nmse_nonoutage": float(nmse_pct_nonoutage[2]),
            "outage_probability": float(outage_prob[si]),
            "n_mc": int(cfg.n_mc),
            "n_nonoutage": n_nonoutage,
            "n_outage": n_outage,
            "outage_threshold_ns": float(outage_threshold_ns),
        }
        stats.append(stat)

        with open(csv_path, "a") as f:
            f.write(
                f"{method_tag},{snr_db:.2f},"
                f"{mse_all[si] * MSE_SECONDS2_TO_NS2:.8e},"
                f"{mse_nonoutage[si] * MSE_SECONDS2_TO_NS2:.8e},"
                f"{outage_prob[si]:.8e},"
                f"{nmse_all[si]:.8e},{nmse_nonoutage[si]:.8e},"
                f"{cfg.n_mc},{n_nonoutage},{n_outage}\n"
            )

        if save_trial_errors:
            trial_mse_by_snr[si, :] = trial_mse
            trial_nmse_by_snr[si, :] = trial_nmse
            trial_outage_by_snr[si, :] = trial_outage
            trial_abs_error_by_snr[si, :] = trial_abs_error

    write_outage_stats_csv(csv_path.replace(".csv", "_outage_stats.csv"), method_tag, stats)
    if save_trial_errors:
        np.savez_compressed(
            csv_path.replace(".csv", "_trial_errors.npz"),
            method_tag=method_tag,
            snr_db_grid=np.asarray(cfg.snr_db_grid, dtype=float),
            trial_mse_seconds2=trial_mse_by_snr,
            trial_mse_ns2=trial_mse_by_snr * MSE_SECONDS2_TO_NS2,
            trial_nmse=trial_nmse_by_snr,
            trial_abs_error_seconds=trial_abs_error_by_snr,
            trial_abs_error_ns=trial_abs_error_by_snr * 1e9,
            trial_is_outage=trial_outage_by_snr,
            outage_threshold_ns=float(outage_threshold_ns),
            n_mc=int(cfg.n_mc),
            rng_seed=int(cfg.rng_seed),
        )

    return mse_all, nmse_all, mse_nonoutage, nmse_nonoutage, outage_prob, stats

def run_mse_test_from_cli(print_usage_func):
    if "--help" in sys.argv or "-h" in sys.argv:
        print_usage_func()
        sys.exit(0)

    if "--combine-trial-errors" in sys.argv:
        trial_error_pattern = parse_value("--combine-trial-errors", None, str)
        mcrb_pattern = parse_value("--mcrb-pattern", None, str)
        out_dir = parse_value("--out-dir", None, str)
        combine_genie_outage_trial_error_files(
            trial_error_pattern,
            mcrb_pattern=mcrb_pattern,
            out_dir=out_dir,
        )
        sys.exit(0)

    n_mc = parse_value("--n-mc", 10000, int)
    snr_db_grid = parse_snr_db_grid()
    mcrb_n_samples_override = parse_value("--mcrb-samples", 100000, int)
    mcrb_batch_size_override = parse_value("--mcrb-batch-size", 1000, int)
    requested_pilot_types = parse_pilot_types()
    N_override = parse_value("--N", None, int)
    K_override = parse_value("--K", None, int)
    P_override = parse_value("--P", None, int)
    tau_max_values = parse_tau_max_values()
    rng_seed_base = parse_value("--rng-seed", 162, int)
    pilot_seed = parse_value("--pilot-seed", 2026, int)
    job_index = parse_value("--job-index", 0, int)
    outage_threshold_ns = parse_value("--outage-threshold-ns", 0.5, float)
    output_dir = parse_value("--out-dir", "results_genie_outage", str)
    estimator = parse_value("--estimator", "top-peaks", str).strip().lower().replace("_", "-")
    if estimator not in {"top-peaks", "sic"}:
        raise ValueError("--estimator must be either 'top-peaks' or 'sic'.")
    estimator_tag = "topPeaks" if estimator == "top-peaks" else "SIC"
    estimator_key = estimator.replace("-", "_")
    sic_min_separation_ns = parse_value("--sic-min-separation-ns", None, float)
    sic_min_separation = None if sic_min_separation_ns is None else sic_min_separation_ns * 1e-9
    sic_refine_iters = parse_value("--sic-refine-iters", 3, int)
    if sic_refine_iters < 0:
        raise ValueError("--sic-refine-iters must be nonnegative.")
    brent_xatol_ns = parse_value("--brent-xatol-ns", None, float)
    brent_xatol_override = None if brent_xatol_ns is None else brent_xatol_ns * 1e-9
    skip_mcrb = "--skip-mcrb" in sys.argv
    mcrb_only = "--mcrb-only" in sys.argv
    if skip_mcrb and mcrb_only:
        raise ValueError("--skip-mcrb and --mcrb-only cannot be used together.")

    cfg = RunConfig(
        snr_db_grid=snr_db_grid,
        n_mc=n_mc,
        rng_seed=rng_seed_base + job_index,
    )

    rng_global = np.random.default_rng(pilot_seed)
    experiment_configs = make_experiment_configs()
    experiment_configs = apply_geometry_overrides(
        experiment_configs,
        P_override=P_override,
        tau_max_values=tau_max_values,
    )
    experiment_configs = apply_dimension_overrides(
        experiment_configs,
        N_override=N_override,
        K_override=K_override,
    )
    if requested_pilot_types is not None:
        experiment_configs = [
            item for item in experiment_configs
            if item["pilot_type"].lower() in requested_pilot_types
        ]
        if not experiment_configs:
            raise ValueError(f"No experiment matched --pilot-types {sorted(requested_pilot_types)}")

    all_results = []
    all_nonoutage_results = []
    for experiment in experiment_configs:
        pilot_type = experiment["pilot_type"]
        N = experiment["N"]
        K = experiment["K"]
        Ts = experiment["Ts"]
        L_norm = experiment["L_norm"]
        delay_grid_step_norm = experiment["delay_grid_step_norm"]
        delay_abs_grid = Ts * np.arange(0.0, np.max(L_norm) + 1e-12, delay_grid_step_norm)
        x, tag_extra_parts = build_pilot(
            pilot_type,
            N,
            K,
            rng_global,
            ml_checkpoint=experiment.get("ml_checkpoint", ML_K60_N90_CHECKPOINT),
            ml_label=experiment.get("ml_label", ML_K60_N90_LABEL),
        )

        print("Pilot type:", pilot_type)
        print("P:", len(L_norm))
        print("tau_max/Ts:", " ".join(f"{value:g}" for value in np.asarray(L_norm, dtype=float)))
        print("Estimator:", estimator)
        if estimator == "sic":
            if sic_min_separation is None:
                print("SIC min separation: delay-grid step")
            else:
                print(f"SIC min separation: {sic_min_separation_ns:g} ns")
            print("SIC refinement iterations:", sic_refine_iters)
        print("||x||^2 =", np.linalg.norm(x) ** 2)

        if skip_mcrb:
            print("Skipping MCRB computation for this chunk.")
            mcrb = np.full(len(cfg.snr_db_grid), np.nan, dtype=float)
        else:
            mcrb = compute_mcrb_curve(
                cfg=cfg,
                x=x,
                L_norm=L_norm,
                Ts=Ts,
                n_samples=mcrb_n_samples_override,
                batch_size=mcrb_batch_size_override,
            )

        delay_grid_step = delay_abs_grid[1] - delay_abs_grid[0]
        brent_resolution_factor = 100
        if brent_xatol_override is not None:
            brent_xatol = brent_xatol_override
        elif skip_mcrb:
            brent_xatol = delay_grid_step * 1e-4
        else:
            brent_xatol = choose_brent_xatol_from_mcrb(
                mcrb_values=mcrb,
                delay_grid_step=delay_grid_step,
                resolution_factor=brent_resolution_factor,
            )

        tag_extra_parts.extend([
            f"est{estimator_tag}",
            f"sicRefine{sic_refine_iters}" if estimator == "sic" else None,
            "brent",
            f"xatol_ns{tag_number(brent_xatol * 1e9)}",
            f"rtolMCRB{brent_resolution_factor}",
            "maxiter50",
            f"genieOutage{tag_number(outage_threshold_ns)}ns",
        ])
        tag_extra_parts = [part for part in tag_extra_parts if part is not None]
        if skip_mcrb:
            tag_extra_parts.append("mcrbskip")
        else:
            tag_extra_parts.append(f"mcrb{mcrb_n_samples_override}")

        run_tag = build_run_tag(
            pilot_type=pilot_type,
            cfg=cfg,
            N=N,
            K=K,
            Ts=Ts,
            L_norm=L_norm,
            delay_abs_grid=delay_abs_grid,
            extra_parts=tag_extra_parts,
        )
        csv_path = make_results_csv(run_tag, out_dir=output_dir)
        print("Saving to:", csv_path)

        if not skip_mcrb:
            save_mcrb_sidecar(
                csv_path=csv_path,
                method_tag=run_tag,
                label=pilot_label(pilot_type),
                snr_grid=cfg.snr_db_grid,
                mcrb_values=mcrb,
                n_samples=mcrb_n_samples_override,
            )
        if mcrb_only:
            print("MCRB-only mode: skipping Monte Carlo simulation for this sequence.")
            continue

        user_kwargs = {
            "pilot": x,
            "Ts": Ts,
            "L_norm": L_norm,
            "delay_abs_grid": delay_abs_grid,
            "min_separation": None,
            "use_brent": True,
            "brent_xatol": brent_xatol,
            "brent_maxiter": 50,
            "estimator": estimator_key,
            "sic_min_separation": sic_min_separation,
            "sic_refine_iters": sic_refine_iters,
            "error_percentiles": (10, 25, 50, 75, 90),
            "save_trial_errors": True,
        }
        mse_all, nmse_all, mse_nonoutage, nmse_nonoutage, outage_prob, stats = (
            run_score_delay_simulation_with_genie_outage(
                cfg=cfg,
                user_kwargs=user_kwargs,
                method_tag=run_tag,
                csv_path=csv_path,
                outage_threshold_ns=outage_threshold_ns,
            )
        )

        summary_path = csv_path.replace(".csv", "_summary.txt")
        save_summary_with_mcrb(cfg.snr_db_grid, mse_all, nmse_all, mcrb, summary_path, error_stats=stats)
        print("Summary saved to:", summary_path)

        label = pilot_label(pilot_type)
        all_results.append({
            "label": f"{label} all",
            "snr_grid": cfg.snr_db_grid.copy(),
            "mse": mse_all.copy(),
            "mcrb": mcrb.copy(),
            "error_stats": stats,
        })
        nonoutage_stats = []
        for item in stats:
            nonoutage_stats.append({
                "snr_db": item["snr_db"],
                "mse_mean": item["mse_mean_nonoutage"],
                "nmse_mean": item["nmse_mean_nonoutage"],
                "percentiles": item["percentiles"],
                "mse_percentiles": item["mse_percentiles_nonoutage"],
                "nmse_percentiles": item["nmse_percentiles_nonoutage"],
                "median_mse": item["median_mse_nonoutage"],
                "median_nmse": item["median_nmse_nonoutage"],
            })
        all_nonoutage_results.append({
            "label": f"{label} non-outage",
            "snr_grid": cfg.snr_db_grid.copy(),
            "mse": mse_nonoutage.copy(),
            "mcrb": mcrb.copy(),
            "error_stats": nonoutage_stats,
        })

        print("\nSNR(dB)   MeanAll(ns^2)   MeanNonOut(ns^2)   P_out   MCRB(ns^2)   NonOut/MCRB")
        for idx, snr in enumerate(cfg.snr_db_grid):
            ratio = (
                mse_nonoutage[idx] / mcrb[idx]
                if np.isfinite(mcrb[idx]) and mcrb[idx] > 0
                else np.nan
            )
            print(
                f"{snr:6.1f}   "
                f"{mse_all[idx] * MSE_SECONDS2_TO_NS2:.2e}   "
                f"{mse_nonoutage[idx] * MSE_SECONDS2_TO_NS2:.2e}   "
                f"{outage_prob[idx]:.2e}   "
                f"{mcrb[idx] * MSE_SECONDS2_TO_NS2:.2e}   "
                f"{ratio:.2e}"
            )

    plot_inputs = all_results + all_nonoutage_results
    if plot_inputs:
        common = experiment_configs[0]
        combined_tag = build_run_tag(
            pilot_type="all_sequences",
            cfg=cfg,
            N=common["N"],
            K=common["K"],
            Ts=common["Ts"],
            L_norm=common["L_norm"],
            delay_abs_grid=common["Ts"] * np.arange(
                0.0,
                np.max(common["L_norm"]) + 1e-12,
                common["delay_grid_step_norm"],
            ),
            extra_parts=[
                part for part in [
                f"est{estimator_tag}",
                f"sicRefine{sic_refine_iters}" if estimator == "sic" else None,
                f"genie_outage_{tag_number(outage_threshold_ns)}ns",
                ]
                if part is not None
            ],
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        os.makedirs(output_dir, exist_ok=True)
        mse_plot = os.path.join(output_dir, f"{combined_tag}_{timestamp}_combined_mse_vs_mcrb.png")
        median_plot = os.path.join(output_dir, f"{combined_tag}_{timestamp}_combined_median_percentile_vs_mcrb.png")
        if plot_all_mse_vs_mcrb(plot_inputs, mse_plot):
            print("Combined plot saved to:", mse_plot)
        if plot_all_median_error_vs_mcrb(plot_inputs, median_plot):
            print("Combined median/percentile plot saved to:", median_plot)
