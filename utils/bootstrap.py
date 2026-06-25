"""
bootstrap.py — Bootstrap Chain Ladder (England & Verrall, 1999/2002)
============================================================================
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from utils.core import YEARS, N, chain_ladder_factors, project_triangle, last_known_diag


@st.cache_data(show_spinner="Calcul des résidus Pearson ajustés...")
def compute_pearson_residuals(CHARGE: np.ndarray, f: np.ndarray) -> dict:
    n = CHARGE.shape[0]
    C_hat = np.full((n, n), np.nan)
    for i in range(n):
        C_hat[i, 0] = CHARGE[i, 0]
        for d in range(1, n):
            if not np.isnan(CHARGE[i, d]) and not np.isnan(CHARGE[i, d - 1]):
                C_hat[i, d] = CHARGE[i, d - 1] * f[d - 1]

    obs_fit = [
        (i, d) for i in range(n) for d in range(1, n)
        if not np.isnan(CHARGE[i, d]) and not np.isnan(C_hat[i, d]) and C_hat[i, d] > 0
    ]
    N_r = len(obs_fit)
    p_cl = n - 1
    resid_raw = np.array([
        (CHARGE[i, d] - C_hat[i, d]) / np.sqrt(C_hat[i, d]) for (i, d) in obs_fit
    ])
    scale = np.sqrt(N_r / max(N_r - p_cl, 1))
    resid_adj = resid_raw * scale
    phi_hat = np.sum(resid_raw ** 2) / max(N_r - p_cl, 1)

    return dict(C_hat=C_hat, obs_fit=obs_fit, N_r=N_r, p_cl=p_cl,
                resid_raw=resid_raw, resid_adj=resid_adj, scale=scale, phi_hat=phi_hat)


@st.cache_data(show_spinner="Simulation Bootstrap en cours (peut prendre quelques secondes)...")
def run_bootstrap(CHARGE: np.ndarray, f: np.ndarray, B: int = 1000, seed: int = 2024) -> dict:
    n = CHARGE.shape[0]
    pr = compute_pearson_residuals(CHARGE, f)
    C_hat, obs_fit, N_r = pr['C_hat'], pr['obs_fit'], pr['N_r']
    resid_adj = pr['resid_adj']

    last_known = last_known_diag(CHARGE)

    np.random.seed(seed)
    provisions_boot = np.zeros(B)
    ibnr_by_ay_boot = np.zeros((B, n))
    f_boot_all = np.zeros((B, n - 1))

    for b in range(B):
        r_star = np.random.choice(resid_adj, size=N_r, replace=True)
        C_star = CHARGE.copy()
        for k, (i, d) in enumerate(obs_fit):
            C_star[i, d] = max(C_hat[i, d] + r_star[k] * np.sqrt(C_hat[i, d]), 1.0)
        f_star = chain_ladder_factors(C_star)
        f_boot_all[b, :] = f_star
        C_star_proj = project_triangle(C_star, f_star)
        for i in range(n):
            ult_b = C_star_proj[i, n - 1]
            ibnr_by_ay_boot[b, i] = max(ult_b - last_known[i], 0)
        provisions_boot[b] = ibnr_by_ay_boot[b, :].sum()

    CHARGE_proj = project_triangle(CHARGE, f)
    ultimate_cl = CHARGE_proj[:, n - 1]
    ibnr_cl = ultimate_cl - last_known
    total_cl = ibnr_cl.sum()

    return dict(provisions_boot=provisions_boot, ibnr_by_ay_boot=ibnr_by_ay_boot,
                f_boot_all=f_boot_all, ibnr_cl=ibnr_cl, total_cl=total_cl,
                last_known=last_known, pearson=pr, B=B)


def dist_stats(arr: np.ndarray) -> dict:
    return {
        'Moyenne': np.mean(arr),
        'Médiane': np.median(arr),
        'Écart-type': np.std(arr),
        'CV (%)': np.std(arr) / np.mean(arr) * 100 if np.mean(arr) != 0 else 0,
        'Skewness': stats.skew(arr),
        'Kurtosis': stats.kurtosis(arr),
        'Min': np.min(arr),
        'Max': np.max(arr),
    }


def var_tvar(arr: np.ndarray, alpha: float):
    """VaR et TVaR au niveau alpha (ex: 95)."""
    var = np.percentile(arr, alpha)
    tail = arr[arr >= var]
    tvar = np.mean(tail) if len(tail) else np.nan
    return var, tvar
