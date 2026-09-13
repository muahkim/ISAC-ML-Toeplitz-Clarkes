import itertools
import os
import heapq
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar

MSE_SECONDS2_TO_NS2 = 1e18


@dataclass
class RunConfig:
    snr_db_grid: np.ndarray
    n_mc: int
    rng_seed: int = 12


def tag_number(value):
    value = float(value)
    if value.is_integer():
        text = str(int(value))
    else:
        text = f"{value:.6g}"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def tag_text(value, max_len=None):
    text = str(value)
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if max_len is not None:
        cleaned = cleaned[:max_len].rstrip("_")
    return cleaned


def tag_array(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    return "-".join(tag_number(value) for value in values)


def build_run_tag(pilot_type, cfg, N, K, Ts, L_norm, delay_abs_grid, extra_parts=None):
    L_norm = np.asarray(L_norm, dtype=float).reshape(-1)
    delay_norm_grid = np.asarray(delay_abs_grid, dtype=float).reshape(-1) / Ts
    snr_grid = np.asarray(cfg.snr_db_grid, dtype=float).reshape(-1)

    grid_step = "none"
    if delay_norm_grid.size > 1:
        grid_steps = np.diff(delay_norm_grid)
        if np.allclose(grid_steps, grid_steps[0]):
            grid_step = tag_number(grid_steps[0])
        else:
            grid_step = "nonuniform"

    if snr_grid.size == 1:
        snr_part = f"snr{tag_number(snr_grid[0])}"
    else:
        snr_part = f"snr{tag_number(snr_grid[0])}to{tag_number(snr_grid[-1])}_{snr_grid.size}pts"

    parts = [
        tag_text(pilot_type),
        "mse_nmse",
        f"P{L_norm.size}",
        f"N{N}",
        f"K{K}",
        f"taumax{tag_array(L_norm)}",
        f"Ts{tag_number(Ts)}",
        snr_part,
        f"dTau{grid_step}",
        f"nmc{cfg.n_mc}",
        f"seed{cfg.rng_seed}",
    ]

    if extra_parts:
        parts.extend(tag_text(part, max_len=80) for part in extra_parts)

    return "_".join(parts)


def make_results_csv(tag, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(out_dir, f"delay_score_{tag}_{timestamp}.csv")
    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(out_dir, f"delay_score_{tag}_{timestamp}_{counter}.csv")
        counter += 1

    with open(filename, "w") as f:
        f.write("method,SNR_dB,MSE_ns2,NMSE,n_mc\n")
        f.flush()
    return filename


def generate_bpsk_pilot(N, rng):
    bits = rng.integers(0, 2, size=N)
    x = 2 * bits - 1
    return x.astype(complex)


def generate_ask_pilot(N, rng):
    bits = rng.integers(0, 2, size=N)
    x = bits * np.sqrt(2.0)
    return x.astype(complex)


def generate_one_hot_pilot(N, rng):
    x = np.zeros(N, dtype=complex)
    pos = rng.integers(0, N)
    x[pos] = np.sqrt(N)
    return x


def generate_ZC_pilot(N, u, normalize=True):
    if np.gcd(u, N) != 1:
        raise ValueError("u must be relatively prime to N")
    n = np.arange(N)
    if N % 2 == 0:
        x = np.exp(-1j * np.pi * u * n * n / N)
    else:
        x = np.exp(-1j * np.pi * u * n * (n + 1) / N)
    if normalize:
        x = x / np.linalg.norm(x) * np.sqrt(N)
    return x


def sample_theta_true(rng, L_norm, Ts):
    L_norm = np.asarray(L_norm, dtype=float)
    delay_norm_true = rng.uniform(low=0.0, high=L_norm)
    return Ts * delay_norm_true


def sample_theta_true_shared_support_rejection(
    rng,
    L_norm,
    Ts,
    min_separation_norm=0.0,
    support_low_norm=0.0,
    max_trials=10000,
):
    L_norm = np.asarray(L_norm, dtype=float).reshape(-1)
    P = L_norm.size
    support_low_norm = float(support_low_norm)
    support_high_norm = float(np.max(L_norm))
    min_separation_norm = float(min_separation_norm)

    if P == 0:
        return np.array([], dtype=float)
    if support_high_norm <= support_low_norm:
        raise ValueError("Shared delay support must have positive length.")
    if min_separation_norm < 0:
        raise ValueError("min_separation_norm must be nonnegative.")
    if (P - 1) * min_separation_norm > support_high_norm - support_low_norm:
        raise ValueError("Shared support is too small for the requested minimum separation.")

    for _ in range(max_trials):
        delay_norm_true = np.sort(
            rng.uniform(low=support_low_norm, high=support_high_norm, size=P)
        )
        if P == 1 or np.all(np.diff(delay_norm_true) >= min_separation_norm):
            return Ts * delay_norm_true

    raise RuntimeError(
        "Failed to sample separated delays. Reduce min_separation_norm or enlarge support."
    )


def sample_y_given_theta(tau_abs_true, snr_db, rng, user_kwargs):
    x = user_kwargs["pilot"]
    N = len(x)
    P = len(tau_abs_true)
    Ts = float(user_kwargs["Ts"])
    add_noise = user_kwargs.get("add_noise", True)
    return_components = user_kwargs.get("return_components", False)

    snr_linear = 10 ** (snr_db / 10.0)
    N0 = 1.0 / snr_linear

    h = (rng.standard_normal(P) + 1j * rng.standard_normal(P)) / np.sqrt(2.0)

    y = np.zeros(N, dtype=complex)
    for p in range(P):
        a_p = build_single_path_hypothesis_vector(tau_abs_true[p], x, Ts)
        y += a_p * h[p]

    if not add_noise:
        if return_components:
            return y, h, np.zeros(N, dtype=complex)
        return y

    w = np.sqrt(N0) * (
        (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2.0)
    )

    if return_components:
        return y + w, h, w
    return y + w


def toeplitz_from_col_row(first_col, first_row):
    first_col = np.asarray(first_col)
    first_row = np.asarray(first_row)
    if first_col[0] != first_row[0]:
        raise ValueError("Toeplitz first column and first row must share the first entry.")

    N = len(first_col)
    M = len(first_row)
    T = np.empty((N, M), dtype=np.result_type(first_col, first_row))

    for i in range(N):
        for j in range(M):
            if i >= j:
                T[i, j] = first_col[i - j]
            else:
                T[i, j] = first_row[j - i]
    return T


def build_Hs_delay_hyp(delay_abs_hyp, N, Ts):
    lags_down = np.arange(N, dtype=float)
    lags_up = np.arange(N, dtype=float)
    delay_norm_hyp = delay_abs_hyp / Ts

    # Toeplitz entries follow H[i, j] = sinc(i - j - delay_norm_hyp).
    first_col = np.sinc(lags_down - delay_norm_hyp)
    first_row = np.sinc(-lags_up - delay_norm_hyp)

    return toeplitz_from_col_row(first_col, first_row).astype(complex)


def build_single_path_hypothesis_vector(delay_abs_hyp, x, Ts):
    N = len(x)
    H = build_Hs_delay_hyp(delay_abs_hyp, N, Ts)
    return (H @ x).astype(complex)


def compute_score_vector(y, delay_abs_grid, Ts, x):
    scores = np.zeros(len(delay_abs_grid), dtype=float)

    for k, delay_abs_hyp in enumerate(delay_abs_grid):
        scores[k] = compute_delay_score(delay_abs_hyp, y, Ts, x)

    return scores


def compute_delay_score(delay_abs_hyp, y, Ts, x):
    a = build_single_path_hypothesis_vector(delay_abs_hyp, x, Ts)

    denom = np.vdot(a, a).real
    if denom < 1e-15:
        return 0.0

    mf = np.vdot(a, y)
    return float((np.abs(mf) ** 2) / denom)


def pick_top_p_from_scores(tau_grid, scores, P, min_separation=None):
    tau_grid = np.asarray(tau_grid, dtype=float)
    scores = np.asarray(scores, dtype=float)

    if min_separation is None:
        idx = np.argsort(scores)[-P:]
        return tau_grid[idx], idx

    selected_idx = []
    order = np.argsort(scores)[::-1]

    for idx in order:
        tau_candidate = tau_grid[idx]
        if all(abs(tau_candidate - tau_grid[j]) >= min_separation for j in selected_idx):
            selected_idx.append(idx)
            if len(selected_idx) == P:
                break

    selected_idx = np.array(selected_idx, dtype=int)
    return np.sort(tau_grid[selected_idx]), selected_idx


def find_local_maxima_by_derivative(scores):
    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        return np.array([], dtype=int)
    if len(scores) == 1:
        return np.array([0], dtype=int)

    d = np.diff(scores)
    maxima_idx = []
    if scores[0] >= scores[1]:
        maxima_idx.append(0)

    for k in range(1, len(scores) - 1):
        if d[k - 1] > 0 and d[k] < 0:
            maxima_idx.append(k)

    if scores[-1] >= scores[-2]:
        maxima_idx.append(len(scores) - 1)

    return np.array(maxima_idx, dtype=int)


def pick_top_p_local_maxima(delay_abs_grid, scores, P):
    delay_abs_grid = np.asarray(delay_abs_grid, dtype=float)
    scores = np.asarray(scores, dtype=float)

    maxima_idx = find_local_maxima_by_derivative(scores)

    if len(maxima_idx) == 0:
        idx = np.argsort(scores)[-P:]
        return np.sort(delay_abs_grid[idx]), idx

    order = np.argsort(scores[maxima_idx])[::-1]
    selected_idx = maxima_idx[order[:P]]
    return np.sort(delay_abs_grid[selected_idx]), selected_idx


def refine_delay_estimates_brent(
    selected_idx,
    delay_abs_grid,
    y,
    Ts,
    x,
    xatol=None,
    maxiter=50,
):
    delay_abs_grid = np.asarray(delay_abs_grid, dtype=float)
    selected_idx = np.asarray(selected_idx, dtype=int).reshape(-1)

    if delay_abs_grid.size < 2:
        return delay_abs_grid[selected_idx]

    if xatol is None:
        grid_step = np.min(np.diff(np.unique(delay_abs_grid)))
        xatol = grid_step * 1e-3

    refined = []
    for idx in selected_idx:
        left_idx = max(idx - 1, 0)
        right_idx = min(idx + 1, delay_abs_grid.size - 1)

        if left_idx == right_idx:
            refined.append(delay_abs_grid[idx])
            continue

        lower = delay_abs_grid[left_idx]
        upper = delay_abs_grid[right_idx]

        result = minimize_scalar(
            lambda tau: -compute_delay_score(tau, y, Ts, x),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": xatol, "maxiter": maxiter},
        )

        if result.success:
            refined.append(result.x)
        else:
            refined.append(delay_abs_grid[idx])

    return np.asarray(refined, dtype=float)


def best_permutation_metrics(tau_true, tau_hat, return_permutation=False, debug=False):
    tau_true = np.asarray(tau_true).reshape(-1)
    tau_hat = np.asarray(tau_hat).reshape(-1)

    if tau_true.shape != tau_hat.shape:
        raise ValueError("tau_true and tau_hat must have the same shape.")

    denom = np.sum(np.abs(tau_true) ** 2)
    if denom == 0:
        raise ValueError("NMSE is undefined because ||tau_true||^2 = 0.")

    best_mse = np.inf
    best_perm_idx = None
    best_tau_hat_perm = None

    n = len(tau_true)

    if debug:
        print("=== Checking all permutations ===")

    for perm_idx in itertools.permutations(range(n)):
        tau_hat_perm = tau_hat[list(perm_idx)]
        diff = tau_true - tau_hat_perm
        mse = np.sum(np.abs(diff) ** 2)

        if debug:
            mse_ns2 = mse * MSE_SECONDS2_TO_NS2
            print(f"perm {perm_idx} -> permuted: {tau_hat_perm}, MSE(ns^2) = {mse_ns2:.2e}")

        if mse < best_mse:
            best_mse = mse
            best_perm_idx = perm_idx
            best_tau_hat_perm = tau_hat_perm.copy()

    best_nmse = best_mse / denom

    if debug:
        print("\n=== Selected BEST permutation ===")
        print(f"Best perm index: {best_perm_idx}")
        print(f"Best permuted tau_hat: {best_tau_hat_perm}")
        print(f"Best MSE(ns^2): {best_mse * MSE_SECONDS2_TO_NS2:.2e}")
        print(f"Best NMSE: {best_nmse:.2e}")

        print("\n=== Verification (marking best) ===")
        for perm_idx in itertools.permutations(range(n)):
            tau_hat_perm = tau_hat[list(perm_idx)]
            mse = np.sum(np.abs(tau_true - tau_hat_perm) ** 2)
            mse_ns2 = mse * MSE_SECONDS2_TO_NS2
            marker = " <-- BEST" if perm_idx == best_perm_idx else ""
            print(f"perm {perm_idx} -> MSE(ns^2) = {mse_ns2:.2e}{marker}")

    if return_permutation:
        return best_mse, best_nmse, best_tau_hat_perm, best_perm_idx
    return best_mse, best_nmse, best_tau_hat_perm


def save_outlier_records(outlier_records, out_dir, method_tag, snr_db):
    if not outlier_records:
        return []

    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []
    ordered_records = sorted(outlier_records, key=lambda item: item["mse"], reverse=True)

    for rank, record in enumerate(ordered_records, start=1):
        path = os.path.join(
            out_dir,
            (
                f"outlier_{tag_text(method_tag, max_len=120)}_"
                f"snr{tag_number(snr_db)}_rank{rank}_"
                f"trial{record['trial_idx']}.npz"
            ),
        )
        np.savez_compressed(
            path,
            method_tag=method_tag,
            snr_db=float(snr_db),
            trial_idx=int(record["trial_idx"]),
            mse_seconds2=float(record["mse"]),
            mse_ns2=float(record["mse"] * MSE_SECONDS2_TO_NS2),
            nmse=float(record["nmse"]),
            Ts=float(record["Ts"]),
            tau_abs_true=record["tau_abs_true"],
            tau_abs_hat=record["tau_abs_hat"],
            selected_idx=record["selected_idx"],
            selected_delay_abs=record["delay_abs_grid"][record["selected_idx"]],
            selected_scores=record["scores"][record["selected_idx"]],
            delay_abs_grid=record["delay_abs_grid"],
            scores=record["scores"],
            pilot=record["pilot"],
            h=record["h"],
            noise=record["noise"],
            y=record["y"],
        )
        saved_paths.append(path)

    return saved_paths


def run_score_delay_simulation_streaming(
    cfg,
    sample_theta_true_fn,
    sample_y_given_theta_fn,
    user_kwargs,
    method_tag,
    csv_path,
):
    rng = np.random.default_rng(cfg.rng_seed)
    mse_per_snr = np.zeros(len(cfg.snr_db_grid), dtype=float)
    nmse_per_snr = np.zeros(len(cfg.snr_db_grid), dtype=float)

    L_norm = np.asarray(user_kwargs["L_norm"], dtype=float)
    Ts = float(user_kwargs["Ts"])
    delay_abs_grid = np.asarray(user_kwargs["delay_abs_grid"], dtype=float)
    min_separation = user_kwargs.get("min_separation", None)
    use_brent = user_kwargs.get("use_brent", True)
    brent_xatol = user_kwargs.get("brent_xatol", None)
    brent_maxiter = user_kwargs.get("brent_maxiter", 50)
    capture_outliers = user_kwargs.get("capture_outliers", False)
    outlier_top_k = int(user_kwargs.get("outlier_top_k", 5))
    outlier_threshold_ns2 = user_kwargs.get("outlier_threshold_ns2", None)
    outlier_dir = user_kwargs.get("outlier_dir", os.path.join("results", "outliers"))
    collect_error_stats = user_kwargs.get("collect_error_stats", False)
    error_percentiles = user_kwargs.get("error_percentiles", (10, 25, 50, 75, 90))
    error_stats = []
    error_stats_csv_path = csv_path.replace(".csv", "_error_stats.csv")
    save_trial_errors = user_kwargs.get("save_trial_errors", collect_error_stats)
    trial_mse_by_snr = (
        np.zeros((len(cfg.snr_db_grid), cfg.n_mc), dtype=float)
        if collect_error_stats and save_trial_errors
        else None
    )
    trial_nmse_by_snr = (
        np.zeros((len(cfg.snr_db_grid), cfg.n_mc), dtype=float)
        if collect_error_stats and save_trial_errors
        else None
    )
    if collect_error_stats:
        with open(error_stats_csv_path, "w") as f:
            percentile_cols = ",".join(
                f"P{tag_number(percentile)}_SE_ns2" for percentile in error_percentiles
            )
            f.write(
                "method,SNR_dB,MeanSE_ns2,MedianSE_ns2,"
                f"{percentile_cols},MeanNMSE,MedianNMSE,n_mc\n"
            )

    P = len(L_norm)

    for si, snr_db in enumerate(cfg.snr_db_grid):
        print(f"[{method_tag}] Running SNR = {snr_db:.1f} dB")
        mse_accum = 0.0
        nmse_accum = 0.0
        x = user_kwargs["pilot"]
        outlier_heap = []
        trial_mse_values = np.zeros(cfg.n_mc, dtype=float) if collect_error_stats else None
        trial_nmse_values = np.zeros(cfg.n_mc, dtype=float) if collect_error_stats else None

        for trial_idx in range(cfg.n_mc):
            tau_abs_true = sample_theta_true_fn(rng, L_norm, Ts)
            if capture_outliers:
                sample_kwargs = dict(user_kwargs)
                sample_kwargs["return_components"] = True
                y, h, noise = sample_y_given_theta_fn(
                    tau_abs_true, float(snr_db), rng, sample_kwargs
                )
            else:
                y = sample_y_given_theta_fn(tau_abs_true, float(snr_db), rng, user_kwargs)
                h = None
                noise = None
            scores = compute_score_vector(y, delay_abs_grid, Ts, x)

            delay_abs_hat, selected_idx = pick_top_p_local_maxima(
                delay_abs_grid, scores, P
            )

            if tau_abs_true.size > delay_abs_hat.size:
                numDelay_diff = tau_abs_true.size - delay_abs_hat.size
                delay_abs_hat_supp, selected_idx_supp = pick_top_p_from_scores(
                    delay_abs_grid, scores, P, min_separation=min_separation
                )
                delay_abs_hat_added = []
                selected_idx_added = []
                for i_p in range(P):
                    if not np.any(selected_idx == selected_idx_supp[-1 - i_p]):
                        delay_abs_hat_added.append(delay_abs_hat_supp[-1 - i_p])
                        selected_idx_added.append(selected_idx_supp[-1 - i_p])
                    if len(delay_abs_hat_added) == numDelay_diff:
                        break
                delay_abs_hat = np.concatenate((delay_abs_hat, np.array(delay_abs_hat_added)))
                selected_idx = np.concatenate((selected_idx, np.array(selected_idx_added, dtype=int)))

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

            if capture_outliers:
                should_store = outlier_threshold_ns2 is None or (
                    mse * MSE_SECONDS2_TO_NS2 >= float(outlier_threshold_ns2)
                )
                if should_store:
                    record = {
                        "trial_idx": trial_idx,
                        "mse": mse,
                        "nmse": nmse,
                        "Ts": Ts,
                        "tau_abs_true": tau_abs_true.copy(),
                        "tau_abs_hat": delay_abs_hat.copy(),
                        "selected_idx": selected_idx.copy(),
                        "delay_abs_grid": delay_abs_grid.copy(),
                        "scores": scores.copy(),
                        "pilot": x.copy(),
                        "h": h.copy(),
                        "noise": noise.copy(),
                        "y": y.copy(),
                    }
                    heap_item = (mse, trial_idx, record)
                    if len(outlier_heap) < outlier_top_k:
                        heapq.heappush(outlier_heap, heap_item)
                    elif mse > outlier_heap[0][0]:
                        heapq.heapreplace(outlier_heap, heap_item)

            if trial_idx < 3:
                selected_scores = scores[selected_idx]
                true_ns = tau_abs_true * 1e9
                est_ns = delay_abs_hat * 1e9

                print("True delay (ns):      ", " ".join(f"{v:.3f}" for v in true_ns))
                print("Estimated delay (ns): ", " ".join(f"{v:.3f}" for v in est_ns))
                print("Selected scores:      ", " ".join(f"{v:.4f}" for v in selected_scores))

                if P == 1:
                    true_grid_idx = np.argmin(np.abs(delay_abs_grid - tau_abs_true[0]))
                    print(f"Nearest grid delay (ns): {delay_abs_grid[true_grid_idx] * 1e9:.3f}")
                    print(f"Score at true grid:      {scores[true_grid_idx]:.4f}")

                print("----")

            mse_accum += mse
            nmse_accum += nmse
            if collect_error_stats:
                trial_mse_values[trial_idx] = mse
                trial_nmse_values[trial_idx] = nmse

        mse = mse_accum / cfg.n_mc
        nmse = nmse_accum / cfg.n_mc
        mse_per_snr[si] = mse
        nmse_per_snr[si] = nmse

        if collect_error_stats:
            percentile_values = np.percentile(trial_mse_values, error_percentiles)
            nmse_percentile_values = np.percentile(trial_nmse_values, error_percentiles)
            median_mse = float(np.percentile(trial_mse_values, 50))
            median_nmse = float(np.percentile(trial_nmse_values, 50))
            error_stats.append({
                "snr_db": float(snr_db),
                "mse_mean": float(mse),
                "nmse_mean": float(nmse),
                "percentiles": np.asarray(error_percentiles, dtype=float),
                "mse_percentiles": percentile_values,
                "nmse_percentiles": nmse_percentile_values,
                "median_mse": median_mse,
                "median_nmse": median_nmse,
            })
            with open(error_stats_csv_path, "a") as f:
                percentile_text = ",".join(
                    f"{value * MSE_SECONDS2_TO_NS2:.8e}" for value in percentile_values
                )
                f.write(
                    f"{method_tag},{snr_db:.2f},{mse * MSE_SECONDS2_TO_NS2:.8e},"
                    f"{median_mse * MSE_SECONDS2_TO_NS2:.8e},{percentile_text},"
                    f"{nmse:.8e},{median_nmse:.8e},{cfg.n_mc}\n"
                )
            if save_trial_errors:
                trial_mse_by_snr[si, :] = trial_mse_values
                trial_nmse_by_snr[si, :] = trial_nmse_values

        with open(csv_path, "a") as f:
            mse_ns2 = mse * MSE_SECONDS2_TO_NS2
            f.write(f"{method_tag},{snr_db:.2f},{mse_ns2:.8e},{nmse:.8e},{cfg.n_mc}\n")
            f.flush()

        if capture_outliers:
            records = [item[2] for item in outlier_heap]
            saved_paths = save_outlier_records(records, outlier_dir, method_tag, snr_db)
            if saved_paths:
                print(f"Saved {len(saved_paths)} outlier records for SNR {snr_db:.1f} dB:")
                for path in saved_paths:
                    print("  ", path)

    if collect_error_stats and save_trial_errors:
        trial_error_path = csv_path.replace(".csv", "_trial_errors.npz")
        np.savez_compressed(
            trial_error_path,
            method_tag=method_tag,
            snr_db_grid=np.asarray(cfg.snr_db_grid, dtype=float),
            trial_mse_seconds2=trial_mse_by_snr,
            trial_mse_ns2=trial_mse_by_snr * MSE_SECONDS2_TO_NS2,
            trial_nmse=trial_nmse_by_snr,
            n_mc=int(cfg.n_mc),
            rng_seed=int(cfg.rng_seed),
        )
        print("Trial-error samples saved to:", trial_error_path)

    if collect_error_stats:
        return mse_per_snr, nmse_per_snr, error_stats
    return mse_per_snr, nmse_per_snr


def save_summary_for_plotting(snr_grid, mse_values, nmse_values, filepath):
    with open(filepath, "w") as f:
        f.write("# SNR_dB    MSE_ns2    NMSE\n")
        for snr, mse, nmse in zip(snr_grid, mse_values, nmse_values):
            mse_ns2 = mse * MSE_SECONDS2_TO_NS2
            f.write(f"{snr:8.3f}    {mse_ns2:.2e}    {nmse:.2e}\n")
