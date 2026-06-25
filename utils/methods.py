"""
methods.py — Implémentation des méthodes déterministes
============================================================================
Chain Ladder, Cape Cod, Bornhuetter-Ferguson, Coût Moyen.
Fonctions pures + cache Streamlit pour éviter les recalculs.
"""
import numpy as np
import pandas as pd
import streamlit as st

from utils.core import (
    YEARS, N, chain_ladder_factors, project_triangle,
    cumulative_factors, last_known_diag,
)


# ---------------------------------------------------------------------------
# CHAIN LADDER
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calcul Chain Ladder...")
def run_chain_ladder(CHARGE: np.ndarray) -> dict:
    n = CHARGE.shape[0]
    f = chain_ladder_factors(CHARGE)
    CHARGE_proj = project_triangle(CHARGE, f)
    ultimate = CHARGE_proj[:, n - 1]
    last_known = last_known_diag(CHARGE)
    ibnr = ultimate - last_known

    results = pd.DataFrame({
        "Année surv.": YEARS,
        "Charge connue": last_known,
        "Ultimate": ultimate,
        "IBNR": ibnr,
        "% développement": np.where(ultimate != 0, last_known / ultimate * 100, 100.0),
    })
    return dict(f=f, CHARGE_proj=CHARGE_proj, ultimate=ultimate,
                last_known=last_known, ibnr=ibnr, results=results)


# ---------------------------------------------------------------------------
# CAPE COD
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calcul Cape Cod...")
def run_cape_cod(CHARGE: np.ndarray, primes: np.ndarray) -> dict:
    n = CHARGE.shape[0]
    f = chain_ladder_factors(CHARGE)
    cdf = cumulative_factors(f)

    last_known = last_known_diag(CHARGE)
    last_d_idx = np.array([int(np.sum(~np.isnan(CHARGE[i]))) - 1 for i in range(n)])
    pct_dev = np.array([
        1.0 / cdf[last_d_idx[i]] if cdf[last_d_idx[i]] > 0 else np.nan
        for i in range(n)
    ])

    denom = np.sum(primes * pct_dev)
    ELR = np.sum(last_known) / denom if denom > 0 else np.nan
    ibnr_cc = primes * ELR * (1 - pct_dev)
    ultimate_cc = last_known + ibnr_cc

    results = pd.DataFrame({
        "Année surv.": YEARS,
        "Prime": primes,
        "% développé": pct_dev * 100,
        "Charge connue": last_known,
        "IBNR Cape Cod": ibnr_cc,
        "Ultimate Cape Cod": ultimate_cc,
    })
    return dict(f=f, cdf=cdf, ELR=ELR, pct_dev=pct_dev,
                last_known=last_known, ibnr_cc=ibnr_cc,
                ultimate_cc=ultimate_cc, results=results)


# ---------------------------------------------------------------------------
# BORNHUETTER-FERGUSON
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calcul Bornhuetter-Ferguson...")
def run_bornhuetter_ferguson(CHARGE: np.ndarray, primes: np.ndarray) -> dict:
    n = CHARGE.shape[0]
    f = chain_ladder_factors(CHARGE)
    cum_f = cumulative_factors(f)

    CHARGE_proj_cl = project_triangle(CHARGE, f)
    ultimate_cl = CHARGE_proj_cl[:, n - 1]

    ratios = np.full(n, np.nan)
    for i in range(n):
        if primes[i] > 0 and ultimate_cl[i] > 0:
            ratios[i] = ultimate_cl[i] / primes[i]

    valid = (primes > 0) & (~np.isnan(ratios))
    weighted_avg_ratio = (
        np.sum(primes[valid] * ratios[valid]) / np.sum(primes[valid])
        if np.any(valid) else np.nan
    )
    ultimate_priori = primes * weighted_avg_ratio

    last_known_index = np.zeros(n, dtype=int)
    last_known_charge = np.zeros(n)
    for i in range(n):
        d_obs = -1
        for d in range(n):
            if not np.isnan(CHARGE[i, d]):
                d_obs = d
            else:
                break
        last_known_index[i] = d_obs
        last_known_charge[i] = CHARGE[i, d_obs] if d_obs >= 0 else 0.0

    ibnr_bf = np.zeros(n)
    ultimate_bf = np.zeros(n)
    for i in range(n):
        d = last_known_index[i]
        if d >= 0 and not np.isnan(cum_f[d]) and cum_f[d] > 0:
            pct_dev = 1.0 / cum_f[d]
            val = ultimate_priori[i] * (1.0 - pct_dev)
            ibnr_bf[i] = max(val, 0.0)
            ultimate_bf[i] = last_known_charge[i] + ibnr_bf[i]
        else:
            ultimate_bf[i] = last_known_charge[i]

    results = pd.DataFrame({
        "Année surv.": YEARS,
        "Prime": primes,
        "Charge connue": last_known_charge,
        "Ultimate a priori": ultimate_priori,
        "Ultimate BF": ultimate_bf,
        "IBNR BF": ibnr_bf,
    })
    return dict(f=f, cum_f=cum_f, weighted_avg_ratio=weighted_avg_ratio,
                ultimate_priori=ultimate_priori, last_known_charge=last_known_charge,
                ibnr_bf=ibnr_bf, ultimate_bf=ultimate_bf, results=results)


# ---------------------------------------------------------------------------
# COÛT MOYEN
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calcul Coût Moyen...")
def run_cout_moyen(rcc: pd.DataFrame, REG_cum: np.ndarray, SAP_inc: np.ndarray) -> dict:
    """Méthode du coût moyen, basée sur les comptages de sinistres (délai de
    déclaration) et la charge cumulée."""
    n = N
    years = YEARS

    if 'DELAI_DEC' not in rcc.columns:
        raise ValueError(
            "La colonne DATE_DEC (ou DELAI_DEC) est requise pour la méthode "
            "Coût Moyen. Vérifiez le fichier inventaire."
        )

    sin_u = rcc.drop_duplicates(subset='ID_SINISTRE')[
        ['ID_SINISTRE', 'ANNEE_SURV', 'DELAI_DEC']
    ]

    N_inc = np.full((n, n), np.nan)
    for i, ay in enumerate(years):
        sub = sin_u[sin_u['ANNEE_SURV'] == ay]
        for j in range(n):
            cal = ay + j
            if cal > 2025:
                break
            N_inc[i, j] = int(sub[sub['DELAI_DEC'] == j]['ID_SINISTRE'].count())

    N_cum = np.full((n, n), np.nan)
    for i in range(n):
        ay = years[i]
        cumul = 0
        for j in range(n):
            cal = ay + j
            if cal > 2025:
                break
            cumul += int(N_inc[i, j])
            N_cum[i, j] = cumul

    C = np.full((n, n), np.nan)
    for i in range(n):
        ay = years[i]
        cumreg = 0.0
        for j in range(n):
            cal = ay + j
            if cal > 2025:
                break
            sval = SAP_inc[i, j]
            rinc = (REG_cum[i, j] - (REG_cum[i, j - 1] if j > 0 else 0.0))
            cumreg += rinc
            C[i, j] = sval + cumreg

    f_N = chain_ladder_factors(N_cum)
    f_C = chain_ladder_factors(C)

    N_proj = project_triangle(N_cum, f_N)
    C_proj = project_triangle(C, f_C)

    N_ult = N_proj[:, n - 1]
    C_ult = C_proj[:, n - 1]

    CM_ult = np.divide(C_ult, N_ult, out=np.zeros_like(C_ult), where=N_ult != 0)

    C_obs = last_known_diag(C)
    N_obs = last_known_diag(N_cum)
    N_tard = N_ult - N_obs
    IBNR_total = C_ult - C_obs
    IBNR_tard = CM_ult * N_tard
    IBNR_dev = IBNR_total - IBNR_tard

    results = pd.DataFrame({
        "Année surv.": years,
        "N_obs": N_obs, "N_ult": N_ult, "N_tardifs": N_tard,
        "Coût moyen ultime": CM_ult,
        "IBNR développement": IBNR_dev,
        "IBNR tardifs": IBNR_tard,
        "IBNR total": IBNR_total,
    })
    return dict(f_N=f_N, f_C=f_C, N_cum=N_cum, C=C, CM_ult=CM_ult,
                IBNR_total=IBNR_total, IBNR_tard=IBNR_tard, IBNR_dev=IBNR_dev,
                results=results)
