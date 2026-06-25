"""
stochastic.py — Modèles stochastiques : GLM (Poisson/Gamma), Log-Normal
============================================================================
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import gammaln
from scipy import stats

from utils.core import YEARS, N


# ---------------------------------------------------------------------------
# IRLS GLM (Poisson surdispersé / Gamma)
# ---------------------------------------------------------------------------
def _irls_glm(y, X_design, family='poisson', max_iter=500, tol=1e-10):
    n_obs, n_params = X_design.shape
    beta = np.linalg.lstsq(X_design, np.log(np.maximum(y, 1e-6)), rcond=None)[0]
    it = 0
    for it in range(max_iter):
        eta = X_design @ beta
        mu = np.exp(eta)
        W = mu if family == 'poisson' else 1.0 / mu
        z = eta + (y - mu) / mu
        XtWX = X_design.T @ (W[:, None] * X_design)
        XtWz = X_design.T @ (W * z)
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    mu = np.exp(X_design @ beta)
    n_p = n_obs - n_params
    if family == 'poisson':
        pr2 = (y - mu) ** 2 / mu
        deviance = 2 * np.sum(y * np.log(np.maximum(y / mu, 1e-15)) - (y - mu))
        std_r = (y - mu) / np.sqrt(mu)
    else:
        pr2 = ((y - mu) / mu) ** 2
        deviance = 2 * np.sum(-np.log(np.maximum(y / mu, 1e-15)) + (y - mu) / mu)
        std_r = (y - mu) / mu

    phi = pr2.sum() / n_p
    W_fin = mu if family == 'poisson' else (1.0 / mu)
    XtWX_fin = X_design.T @ (W_fin[:, None] * X_design)
    try:
        cov = phi * np.linalg.inv(XtWX_fin)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)

    return dict(beta=beta, mu=mu, phi=phi, deviance=deviance,
                pearson_chi2=pr2.sum(), std_resid=std_r,
                raw_resid=y - mu, se=se, n_iter=it + 1, n_p=n_p,
                y=y, n_obs=n_obs, n_params=n_params)


def _build_design(X: np.ndarray):
    """Construit la matrice de design (effets AY + effets développement)
    à partir d'un triangle incrémental X (n x n, NaN = futur)."""
    n = X.shape[0]
    obs_pairs = [(i, j) for i in range(n) for j in range(n) if not np.isnan(X[i, j])]
    N_obs = len(obs_pairs)
    p = 2 * n - 1
    y = np.array([X[i, j] for (i, j) in obs_pairs])
    M = np.zeros((N_obs, p))
    for k, (i, j) in enumerate(obs_pairs):
        M[k, 0] = 1
        if i > 0:
            M[k, i] = 1
        if j > 0:
            M[k, n - 1 + j] = 1
    return M, y, obs_pairs, p


def _full_triangle_from_beta(beta, n, X_obs):
    Yh = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            v = beta[0]
            if i > 0:
                v += beta[i]
            if j > 0:
                v += beta[n - 1 + j]
            Yh[i, j] = v
    Xh = np.exp(Yh)
    prov = np.array([
        sum(Xh[i, j] for j in range(n) if np.isnan(X_obs[i, j]))
        for i in range(n)
    ])
    return Xh, prov


@st.cache_data(show_spinner="Estimation des modèles GLM (Poisson / Gamma)...")
def run_glm_poisson_gamma(X_inc: np.ndarray, SAP_actifs: np.ndarray) -> dict:
    n = X_inc.shape[0]
    # Les incréments doivent être strictement positifs pour le lien log
    X_pos = np.where((X_inc is not None) & (~np.isnan(X_inc)) & (X_inc <= 0), 1e-3, X_inc)

    M, y_obs, obs_pairs, p = _build_design(X_pos)

    res_P = _irls_glm(y_obs, M, 'poisson')
    res_G = _irls_glm(y_obs, M, 'gamma')

    Xh_P, prov_P = _full_triangle_from_beta(res_P['beta'], n, X_pos)
    Xh_G, prov_G = _full_triangle_from_beta(res_G['beta'], n, X_pos)

    prov_P_net = np.maximum(prov_P - SAP_actifs, 0)
    prov_G_net = np.maximum(prov_G - SAP_actifs, 0)

    results = pd.DataFrame({
        "Année surv.": YEARS,
        "Provision brute Poisson": prov_P,
        "Provision nette Poisson": prov_P_net,
        "Provision brute Gamma": prov_G,
        "Provision nette Gamma": prov_G_net,
        "SAP actifs": SAP_actifs,
    })

    return dict(res_P=res_P, res_G=res_G, prov_P=prov_P, prov_G=prov_G,
                prov_P_net=prov_P_net, prov_G_net=prov_G_net,
                obs_pairs=obs_pairs, results=results, p=p)


# ---------------------------------------------------------------------------
# MODÈLE FACTORIEL LOG-NORMAL (Kremer, 1982)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Estimation du modèle Log-Normal...")
def run_lognormal(X_inc: np.ndarray, SAP_actifs: np.ndarray) -> dict:
    n = X_inc.shape[0]
    Y = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if not np.isnan(X_inc[i, j]) and X_inc[i, j] > 0:
                Y[i, j] = np.log(X_inc[i, j])

    obs_pairs = [(i, j) for i in range(n) for j in range(n) if not np.isnan(Y[i, j])]
    N_obs = len(obs_pairs)
    p = 2 * n - 1
    Y_vec = np.array([Y[i, j] for (i, j) in obs_pairs])

    M = np.zeros((N_obs, p))
    for k, (i, j) in enumerate(obs_pairs):
        M[k, 0] = 1
        if i > 0:
            M[k, i] = 1
        if j > 0:
            M[k, n - 1 + j] = 1

    ksi_hat = np.linalg.solve(M.T @ M, M.T @ Y_vec)
    mu_hat = ksi_hat[0]
    alpha_hat = np.concatenate([[0.0], ksi_hat[1:n]])
    beta_hat = np.concatenate([[0.0], ksi_hat[n:2 * n - 1]])

    Y_hat_obs = M @ ksi_hat
    residuals = Y_vec - Y_hat_obs
    N_p = N_obs - p
    sigma2_hat = np.sum(residuals ** 2) / N_p
    sigma_hat = np.sqrt(sigma2_hat)
    R2 = 1 - np.sum(residuals ** 2) / np.sum((Y_vec - np.mean(Y_vec)) ** 2)

    Y_hat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            Y_hat[i, j] = mu_hat + alpha_hat[i] + beta_hat[j]
    X_hat = np.exp(Y_hat + sigma2_hat / 2)

    prov_brute = np.array([
        sum(X_hat[i, j] for j in range(n) if np.isnan(X_inc[i, j]))
        for i in range(n)
    ])
    prov_nette = np.maximum(prov_brute - SAP_actifs, 0)

    # Tests de normalité des résidus
    sw_stat, sw_p = stats.shapiro(residuals) if N_obs >= 3 else (np.nan, np.nan)

    results = pd.DataFrame({
        "Année surv.": YEARS,
        "Provision brute": prov_brute,
        "SAP actifs": SAP_actifs,
        "Provision nette": prov_nette,
    })

    return dict(mu_hat=mu_hat, sigma2_hat=sigma2_hat, sigma_hat=sigma_hat,
                R2=R2, residuals=residuals, N_obs=N_obs, p=p, N_p=N_p,
                shapiro_stat=sw_stat, shapiro_p=sw_p,
                prov_brute=prov_brute, prov_nette=prov_nette,
                obs_pairs=obs_pairs, results=results)


# ---------------------------------------------------------------------------
# SÉLECTION DE MODÈLE — quasi-AIC / BIC
# ---------------------------------------------------------------------------
def _ql_poisson(y, mu, phi, p):
    n_obs = len(y)
    q = (-n_obs / 2 * np.log(2 * np.pi * phi) - 0.5 * np.sum(np.log(mu))
         - 1 / (2 * phi) * np.sum((y - mu) ** 2 / mu))
    return q, -2 * q + 2 * p, -2 * q + p * np.log(n_obs)


def _ql_gamma(y, mu, phi, p):
    n_obs = len(y)
    nu = 1 / phi
    ll = np.sum((nu - 1) * np.log(y) - nu * np.log(mu) - y * nu / mu
                - gammaln(nu) + nu * np.log(nu))
    return ll, -2 * ll + 2 * p, -2 * ll + p * np.log(n_obs)


def _ql_lognorm(y, mu_x, sigma2, p):
    n_obs = len(y)
    log_y = np.log(y)
    mu_ln_h = np.log(mu_x) - sigma2 / 2
    ll = (-n_obs / 2 * np.log(2 * np.pi) - n_obs / 2 * np.log(sigma2)
          - np.sum(log_y) - 1 / (2 * sigma2) * np.sum((log_y - mu_ln_h) ** 2))
    return ll, -2 * ll + 2 * p, -2 * ll + p * np.log(n_obs)


@st.cache_data(show_spinner="Calcul des critères de sélection (AIC quasi)...")
def model_selection_table(glm_res: dict, ln_res: dict) -> pd.DataFrame:
    """Compare Poisson surdispersé, Gamma et Log-Normal sur le critère AIC/BIC
    quasi. Les trois modèles sont ajustés sur le même triangle incrémental
    X_inc ; on suppose donc le même ensemble d'observations (cellules
    strictement positives), ce qui est garanti par construction dans
    run_glm_poisson_gamma() et run_lognormal()."""
    res_P, res_G = glm_res['res_P'], glm_res['res_G']
    p = glm_res['p']
    y = res_P['y']  # observations utilisées par les GLM (Poisson/Gamma)
    sigma2 = ln_res['sigma2_hat']

    qP, aicP, bicP = _ql_poisson(y, res_P['mu'], res_P['phi'], p)
    qG, aicG, bicG = _ql_gamma(y, res_G['mu'], res_G['phi'], p)

    # Pour le log-normal, mu (espérance, pas médiane) est recomposée à partir
    # des résidus stockés : residual = Y_vec - Y_hat_obs => Y_hat_obs = log(y) - residual
    if len(ln_res['residuals']) == len(y):
        Y_hat_obs = np.log(np.maximum(y, 1e-9)) - ln_res['residuals']
        mu_ln_obs = np.exp(Y_hat_obs + sigma2 / 2)
        qL, aicL, bicL = _ql_lognorm(y, mu_ln_obs, sigma2, p)
    else:
        qL, aicL, bicL = np.nan, np.nan, np.nan

    table = pd.DataFrame({
        "Modèle": ["Poisson surdispersé", "Gamma", "Log-Normal"],
        "Déviance / SCE résidus": [res_P['deviance'], res_G['deviance'], np.sum(ln_res['residuals'] ** 2)],
        "phi (dispersion)": [res_P['phi'], res_G['phi'], sigma2],
        "Quasi log-vraisemblance": [qP, qG, qL],
        "AIC quasi": [aicP, aicG, aicL],
        "BIC quasi": [bicP, bicG, bicL],
    })
    best_idx = table["AIC quasi"].idxmin()
    table["Meilleur (AIC)"] = ["★" if i == best_idx else "" for i in range(len(table))]
    return table
