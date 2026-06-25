"""
core.py — Chargement des données et construction des triangles actuariels
============================================================================
Toutes les fonctions lourdes (lecture fichiers, construction de triangles,
facteurs CL) sont mises en cache via @st.cache_data pour éviter les
recalculs à chaque interaction utilisateur.
"""
import io
import numpy as np
import pandas as pd
import streamlit as st

YEARS = list(range(2016, 2026))
N = 10
DEV_COLS = [f"D{d+1}" for d in range(N)]


# ---------------------------------------------------------------------------
# 1. CHARGEMENT DES FICHIERS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Lecture du fichier inventaire...")
def load_inventaire(file_bytes: bytes) -> pd.DataFrame:
    """Charge et nettoie le CSV inventaire auto, filtré sur la garantie RCC."""
    df = pd.read_csv(io.BytesIO(file_bytes), sep=';', engine='python')

    required = {'CODE_GAR', 'MNT_REG', 'MNT_SAP', 'ANNEE_SURV', 'ANNEE'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans l'inventaire : {missing}")

    rcc = df[df['CODE_GAR'] == 'RCC'].copy()
    rcc['MNT_REG'] = rcc['MNT_REG'].astype(str).str.replace(',', '.').astype(float)
    rcc['MNT_SAP'] = pd.to_numeric(rcc['MNT_SAP'], errors='coerce').fillna(0)
    rcc = rcc[(rcc['ANNEE_SURV'] >= 2016) & (rcc['ANNEE_SURV'] <= 2025)]

    if rcc.empty:
        raise ValueError("Aucune ligne RCC trouvée après filtrage 2016-2025.")

    # Délai de déclaration si la colonne existe (utile pour le triangle N)
    if 'DATE_DEC' in rcc.columns:
        try:
            rcc['ANNEE_DEC'] = rcc['DATE_DEC'].astype(str).str[-4:].astype(int)
            rcc['DELAI_DEC'] = rcc['ANNEE_DEC'] - rcc['ANNEE_SURV']
        except Exception:
            pass

    return rcc


@st.cache_data(show_spinner="Lecture du fichier primes...")
def load_primes(file_bytes: bytes) -> dict:
    """Charge le fichier Excel des primes acquises RC Auto -> dict {année: prime}."""
    xl = pd.read_excel(io.BytesIO(file_bytes))

    col_annee = None
    for col in xl.columns:
        cl = str(col).lower()
        if 'année' in cl or 'annee' in cl or 'year' in cl:
            col_annee = col
            break
    if col_annee is None:
        col_annee = xl.columns[0]

    col_prime = None
    for col in xl.columns:
        if 'rc' in str(col).lower() and 'auto' in str(col).lower():
            col_prime = col
            break
    if col_prime is None:
        # fallback : 2e colonne
        col_prime = xl.columns[1]

    primes_df = xl[[col_annee, col_prime]].copy()
    primes_df.columns = ['Annee', 'Prime']
    primes_df['Prime'] = (
        primes_df['Prime'].astype(str).str.replace(',', '.').astype(float)
    )
    primes_df['Annee'] = pd.to_numeric(primes_df['Annee'], errors='coerce')
    primes_df = primes_df.dropna(subset=['Annee'])
    primes_df['Annee'] = primes_df['Annee'].astype(int)
    primes_df = primes_df[primes_df['Annee'].between(2016, 2025)].sort_values('Annee')

    prime_dict = dict(zip(primes_df['Annee'], primes_df['Prime']))
    # Compléter les années manquantes avec 0
    for y in YEARS:
        prime_dict.setdefault(y, 0.0)
    return prime_dict


# ---------------------------------------------------------------------------
# 2. CONSTRUCTION DES TRIANGLES (SAP, RÈGLEMENTS, CHARGE)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Construction des triangles actuariels...")
def build_triangles(rcc: pd.DataFrame) -> dict:
    """Construit les triangles SAP incrémental, Règlements cumulés et Charge totale."""
    n = N
    years = YEARS

    sap_raw = (rcc.groupby(['ANNEE_SURV', 'ANNEE'])['MNT_SAP']
                  .sum().unstack(fill_value=0)
                  .reindex(index=years, columns=years, fill_value=0))
    reg_raw = (rcc.groupby(['ANNEE_SURV', 'ANNEE'])['MNT_REG']
                  .sum().unstack(fill_value=0)
                  .reindex(index=years, columns=years, fill_value=0))

    SAP_inc = np.zeros((n, n))
    REG_cum = np.full((n, n), np.nan)
    CHARGE = np.full((n, n), np.nan)
    X_inc = np.full((n, n), np.nan)  # règlements incrémentaux (pour GLM / Log-Normal)

    for i in range(n):
        ay = years[i]
        cumreg = 0.0
        for d in range(n):
            cal = ay + d
            if cal > 2025:
                break
            sap_val = sap_raw.loc[ay, cal] if cal in sap_raw.columns else 0.0
            reg_inc = reg_raw.loc[ay, cal] if cal in reg_raw.columns else 0.0
            cumreg += reg_inc
            SAP_inc[i, d] = sap_val
            REG_cum[i, d] = cumreg
            CHARGE[i, d] = sap_val + cumreg
            X_inc[i, d] = reg_inc

    # SAP actifs (provisions au bilan, dernière année calendaire = 2025)
    sap_actifs_ser = rcc[rcc['ANNEE'] == 2025].groupby('ANNEE_SURV')['MNT_SAP'].sum()
    SAP_actifs = np.array([float(sap_actifs_ser.get(y, 0)) for y in years])

    return dict(
        SAP_inc=SAP_inc, REG_cum=REG_cum, CHARGE=CHARGE,
        X_inc=X_inc, SAP_actifs=SAP_actifs,
        years=years, n=n,
    )


@st.cache_data(show_spinner="Calcul des facteurs Chain Ladder...")
def chain_ladder_factors(CHARGE: np.ndarray) -> np.ndarray:
    """Facteurs de développement Chain Ladder f[d] = Σ C(i,d+1) / Σ C(i,d)."""
    n = CHARGE.shape[0]
    f = np.zeros(n - 1)
    for d in range(n - 1):
        num, den = 0.0, 0.0
        for i in range(n):
            if (not np.isnan(CHARGE[i, d]) and
                    not np.isnan(CHARGE[i, d + 1]) and
                    CHARGE[i, d] > 0):
                num += CHARGE[i, d + 1]
                den += CHARGE[i, d]
        f[d] = num / den if den > 0 else np.nan
    return f


@st.cache_data(show_spinner="Projection du triangle...")
def project_triangle(CHARGE: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Complète le triangle par les facteurs Chain Ladder donnés."""
    n = CHARGE.shape[0]
    proj = CHARGE.copy()
    for i in range(n):
        for d in range(n):
            if np.isnan(proj[i, d]):
                prev = proj[i, d - 1]
                if not np.isnan(prev) and (d - 1) < (n - 1):
                    proj[i, d] = prev * f[d - 1]
    return proj


def cumulative_factors(f: np.ndarray) -> np.ndarray:
    """CDF[d] = f[d] x f[d+1] x ... x f[n-2], queue = 1."""
    n = len(f) + 1
    cdf = np.ones(n)
    for d in range(n - 2, -1, -1):
        cdf[d] = cdf[d + 1] * f[d]
    return cdf


def last_known_diag(CHARGE: np.ndarray) -> np.ndarray:
    """Dernière valeur connue (diagonale) de chaque ligne du triangle."""
    n = CHARGE.shape[0]
    return np.array([np.nanmax(CHARGE[i]) for i in range(n)])


def to_df(matrix: np.ndarray, index=None, columns=None) -> pd.DataFrame:
    index = index if index is not None else YEARS
    columns = columns if columns is not None else DEV_COLS
    return pd.DataFrame(matrix, index=index, columns=columns)


def fmt_money(x) -> str:
    try:
        return f"{x:,.0f}"
    except Exception:
        return str(x)
