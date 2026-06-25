"""
app.py — Application Streamlit : Provisionnement actuariel RCC
============================================================================
ASTREE Assurances — PSAP Branche Automobile — Garantie RCC (2016-2025)

Navigation par sidebar, contenu dynamique selon la méthode choisie.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils.core import YEARS, N, DEV_COLS, to_df, fmt_money
from utils import methods, stochastic, bootstrap as boot_mod
from utils.core import load_inventaire, load_primes, build_triangles
from utils.ui import inject_custom_css, page_header, kpi_row

st.set_page_config(
    page_title="PSAP RCC — ASTREE Assurances",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'figure.facecolor': 'white', 'axes.facecolor': '#FAFAFA',
})
C_BLUE, C_RED, C_GREEN, C_ORANGE = '#2E75B6', '#C00000', '#70AD47', '#ED7D31'


# ===========================================================================
# SIDEBAR — UPLOAD & NAVIGATION
# ===========================================================================
st.sidebar.title("📊 PSAP — Garantie RCC")
st.sidebar.caption("ASTREE Assurances · 2016–2025")
st.sidebar.divider()

st.sidebar.subheader("1. Données")
inv_file = st.sidebar.file_uploader(
    "Inventaire auto (.csv)", type=["csv"],
    help="Fichier CSV séparé par ';' contenant les colonnes CODE_GAR, MNT_REG, MNT_SAP, ANNEE_SURV, ANNEE."
)
primes_file = st.sidebar.file_uploader(
    "Primes acquises RC Auto (.xlsx)", type=["xlsx", "xls"],
    help="Fichier Excel contenant les primes acquises par année (colonne 'RC Auto')."
)

st.sidebar.divider()
st.sidebar.subheader("2. Méthode")
section = st.sidebar.radio(
    "Choisir une méthode",
    [
        "🏠 Accueil",
        "📐 Chain Ladder",
        "🎯 Cape Cod",
        "📈 Bornhuetter-Ferguson",
        "💰 Coût Moyen",
        "🧮 GLM (Poisson / Gamma)",
        "📉 Log-Normal",
        "🏆 Sélection du modèle",
        "🔁 Bootstrap",
        "🏁 Comparaison des méthodes",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(
    "💡 Les calculs sont mis en cache : ils ne sont relancés que si les "
    "fichiers ou paramètres changent."
)


# ===========================================================================
# CHARGEMENT DES DONNÉES (avec garde-fous)
# ===========================================================================
def require_data():
    """Vérifie la présence des fichiers et retourne (rcc, triangles, primes)
    ou None si les données ne sont pas prêtes."""
    if inv_file is None:
        st.warning(
            "⬅️ Merci d'importer le fichier **Inventaire auto (.csv)** dans "
            "la barre latérale pour commencer."
        )
        return None
    try:
        rcc = load_inventaire(inv_file.getvalue())
    except Exception as e:
        st.error(f"Erreur lors de la lecture de l'inventaire : {e}")
        return None

    try:
        tri = build_triangles(rcc)
    except Exception as e:
        st.error(f"Erreur lors de la construction des triangles : {e}")
        return None

    primes = None
    if primes_file is not None:
        try:
            prime_dict = load_primes(primes_file.getvalue())
            primes = np.array([prime_dict[y] for y in YEARS])
        except Exception as e:
            st.error(f"Erreur lors de la lecture des primes : {e}")
            primes = None

    return rcc, tri, primes


def require_primes(primes):
    if primes is None:
        st.warning(
            "⬅️ Cette méthode nécessite le fichier **Primes acquises RC Auto "
            "(.xlsx)**. Merci de l'importer dans la barre latérale."
        )
        st.stop()


# ===========================================================================
# PAGE : ACCUEIL
# ===========================================================================
if section == "🏠 Accueil":
    page_header(
        "📊 Évaluation des PSAP — Garantie RCC",
        "Application interactive de provisionnement actuariel pour la "
        "garantie Responsabilité Civile Corporelle — portefeuille automobile "
        "ASTREE Assurances, années de survenance 2016–2025.",
        badges=["Chain Ladder", "Cape Cod", "Bornhuetter-Ferguson",
                "Coût Moyen", "GLM", "Log-Normal", "Bootstrap"],
    )

    data = require_data()
    if data is None:
        st.info(
            "📁 Importez vos fichiers dans la barre latérale pour afficher "
            "un aperçu des données et activer les méthodes de calcul."
        )
    else:
        rcc, tri, primes = data
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes RCC chargées", f"{len(rcc):,}")
        c2.metric("Années couvertes", f"{YEARS[0]}–{YEARS[-1]}")
        c3.metric("Charge connue totale", fmt_money(np.nansum(tri['CHARGE'][:, 0])) + " (1ère diag.)")
        c4.metric("Primes chargées", "✅ Oui" if primes is not None else "❌ Non")

        st.subheader("Aperçu du triangle de charge totale (SAP + Règlements cumulés)")
        st.dataframe(
            to_df(tri['CHARGE']).style.format("{:,.0f}", na_rep="—"),
            use_container_width=True,
        )

        with st.expander("Voir un extrait des données brutes (10 lignes)"):
            st.dataframe(rcc.head(10), use_container_width=True)


# ===========================================================================
# PAGE : CHAIN LADDER
# ===========================================================================
elif section == "📐 Chain Ladder":
    st.title("📐 Méthode Chain Ladder")
    data = require_data()
    if data is None:
        st.stop()
    rcc, tri, primes = data

    res = methods.run_chain_ladder(tri['CHARGE'])

    st.subheader("Facteurs de développement")
    f_df = pd.DataFrame({
        "Transition": [f"D{d+1} → D{d+2}" for d in range(N - 1)],
        "Facteur f": res['f'],
    })
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(f_df.style.format({"Facteur f": "{:.6f}"}), use_container_width=True, hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(range(1, N), res['f'], 'o-', color=C_BLUE, linewidth=2, markersize=7)
        ax.set_xlabel("Transition de développement")
        ax.set_ylabel("Facteur")
        ax.set_title("Facteurs de développement Chain Ladder")
        ax.set_xticks(range(1, N))
        st.pyplot(fig, clear_figure=True)

    st.subheader("Triangle projeté")
    st.dataframe(to_df(res['CHARGE_proj']).style.format("{:,.0f}", na_rep="—"),
                 use_container_width=True)

    st.subheader("Résultats IBNR par année de survenance")
    show = res['results'].copy()
    st.dataframe(
        show.style.format({
            "Charge connue": "{:,.0f}", "Ultimate": "{:,.0f}",
            "IBNR": "{:,.0f}", "% développement": "{:.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )

    total_ibnr = res['ibnr'].sum()
    total_ult = res['ultimate'].sum()
    c1, c2 = st.columns(2)
    c1.metric("IBNR total (Chain Ladder)", fmt_money(total_ibnr))
    c2.metric("Ultimate total", fmt_money(total_ult))


# ===========================================================================
# PAGE : CAPE COD
# ===========================================================================
elif section == "🎯 Cape Cod":
    st.title("🎯 Méthode Cape Cod")
    data = require_data()
    if data is None:
        st.stop()
    rcc, tri, primes = data
    require_primes(primes)

    res = methods.run_cape_cod(tri['CHARGE'], primes)

    st.metric("ELR (Expected Loss Ratio)", f"{res['ELR']:.4%}")

    st.subheader("Résultats par année de survenance")
    st.dataframe(
        res['results'].style.format({
            "Prime": "{:,.0f}", "% développé": "{:.2f}%",
            "Charge connue": "{:,.0f}", "IBNR Cape Cod": "{:,.0f}",
            "Ultimate Cape Cod": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    cl_res = methods.run_chain_ladder(tri['CHARGE'])
    st.subheader("Comparaison Chain Ladder vs Cape Cod (IBNR)")
    comp = pd.DataFrame({
        "Année": YEARS,
        "IBNR Chain Ladder": cl_res['ibnr'],
        "IBNR Cape Cod": res['ibnr_cc'],
        "Écart": res['ibnr_cc'] - cl_res['ibnr'],
    })
    st.dataframe(comp.style.format({c: "{:,.0f}" for c in comp.columns if c != "Année"}),
                 use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(N)
    width = 0.35
    ax.bar(x - width / 2, cl_res['ibnr'] / 1e6, width, label="Chain Ladder", color=C_BLUE)
    ax.bar(x + width / 2, res['ibnr_cc'] / 1e6, width, label="Cape Cod", color=C_ORANGE)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y)[-2:] for y in YEARS])
    ax.set_ylabel("IBNR (M)")
    ax.set_xlabel("Année de survenance")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.metric("IBNR total Cape Cod", fmt_money(res['ibnr_cc'].sum()))


# ===========================================================================
# PAGE : BORNHUETTER-FERGUSON
# ===========================================================================
elif section == "📈 Bornhuetter-Ferguson":
    st.title("📈 Méthode Bornhuetter-Ferguson")
    data = require_data()
    if data is None:
        st.stop()
    rcc, tri, primes = data
    require_primes(primes)

    res = methods.run_bornhuetter_ferguson(tri['CHARGE'], primes)

    st.metric("Ratio de perte ultime moyen pondéré (a priori)", f"{res['weighted_avg_ratio']:.4f}")

    st.subheader("Résultats par année de survenance")
    st.dataframe(
        res['results'].style.format({
            "Prime": "{:,.0f}", "Charge connue": "{:,.0f}",
            "Ultimate a priori": "{:,.0f}", "Ultimate BF": "{:,.0f}",
            "IBNR BF": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.metric("IBNR total Bornhuetter-Ferguson", fmt_money(res['ibnr_bf'].sum()))


# ===========================================================================
# PAGE : COÛT MOYEN
# ===========================================================================
elif section == "💰 Coût Moyen":
    st.title("💰 Méthode du Coût Moyen")
    data = require_data()
    if data is None:
        st.stop()
    rcc, tri, primes = data

    if 'DELAI_DEC' not in rcc.columns:
        st.error(
            "⚠️ La méthode Coût Moyen nécessite la colonne **DATE_DEC** dans "
            "le fichier inventaire (pour calculer le délai de déclaration). "
            "Cette colonne n'a pas été détectée."
        )
        st.stop()

    try:
        res = methods.run_cout_moyen(rcc, tri['REG_cum'], tri['SAP_inc'])
    except Exception as e:
        st.error(f"Erreur lors du calcul : {e}")
        st.stop()

    st.subheader("Résultats par année de survenance")
    st.dataframe(
        res['results'].style.format({
            "N_obs": "{:.0f}", "N_ult": "{:.1f}", "N_tardifs": "{:.1f}",
            "Coût moyen ultime": "{:,.0f}",
            "IBNR développement": "{:,.0f}", "IBNR tardifs": "{:,.0f}",
            "IBNR total": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("IBNR développement (total)", fmt_money(res['IBNR_dev'].sum()))
    c2.metric("IBNR sinistres tardifs (total)", fmt_money(res['IBNR_tard'].sum()))
    c3.metric("IBNR total", fmt_money(res['IBNR_total'].sum()))


# ===========================================================================
# PAGE : GLM POISSON / GAMMA
# ===========================================================================
elif section == "🧮 GLM (Poisson / Gamma)":
    st.title("🧮 Modèles GLM — Poisson surdispersé & Gamma")
    data = require_data()
    if data is None:
        st.stop()
    rcc, tri, primes = data

    res = stochastic.run_glm_poisson_gamma(tri['X_inc'], tri['SAP_actifs'])
    res_P, res_G = res['res_P'], res['res_G']

    tab1, tab2, tab3 = st.tabs(["Poisson surdispersé", "Gamma", "Comparaison"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Déviance", f"{res_P['deviance']:,.1f}")
        c2.metric("χ² Pearson", f"{res_P['pearson_chi2']:,.1f}")
        c3.metric("φ (dispersion)", f"{res_P['phi']:.4f}")
        c4.metric("Itérations IRLS", res_P['n_iter'])
        st.dataframe(
            res['results'][["Année surv.", "Provision brute Poisson",
                             "SAP actifs", "Provision nette Poisson"]]
            .style.format({c: "{:,.2f}" for c in
                           ["Provision brute Poisson", "SAP actifs", "Provision nette Poisson"]}),
            use_container_width=True, hide_index=True,
        )
        st.metric("Provision nette totale (Poisson)", fmt_money(res['prov_P_net'].sum()))

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Déviance", f"{res_G['deviance']:,.4f}")
        c2.metric("χ² Pearson", f"{res_G['pearson_chi2']:,.4f}")
        c3.metric("φ (dispersion)", f"{res_G['phi']:.4f}")
        c4.metric("Itérations IRLS", res_G['n_iter'])
        st.dataframe(
            res['results'][["Année surv.", "Provision brute Gamma",
                             "SAP actifs", "Provision nette Gamma"]]
            .style.format({c: "{:,.2f}" for c in
                           ["Provision brute Gamma", "SAP actifs", "Provision nette Gamma"]}),
            use_container_width=True, hide_index=True,
        )
        st.metric("Provision nette totale (Gamma)", fmt_money(res['prov_G_net'].sum()))

    with tab3:
        comp = pd.DataFrame({
            "Année": YEARS,
            "Provision nette Poisson": res['prov_P_net'],
            "Provision nette Gamma": res['prov_G_net'],
        })
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(N)
        width = 0.35
        ax.bar(x - width / 2, comp["Provision nette Poisson"] / 1e6, width,
               label="Poisson", color=C_BLUE)
        ax.bar(x + width / 2, comp["Provision nette Gamma"] / 1e6, width,
               label="Gamma", color=C_GREEN)
        ax.set_xticks(x)
        ax.set_xticklabels([str(y)[-2:] for y in YEARS])
        ax.set_ylabel("Provision nette (M)")
        ax.set_xlabel("Année de survenance")
        ax.legend()
        st.pyplot(fig, clear_figure=True)
        st.dataframe(comp.style.format({c: "{:,.0f}" for c in comp.columns if c != "Année"}),
                     use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE : LOG-NORMAL
# ===========================================================================
elif section == "📉 Log-Normal":
    st.title("📉 Modèle Factoriel Log-Normal")
    data = require_data()
    if data is None:
        st.stop()
    rcc, tri, primes = data

    res = stochastic.run_lognormal(tri['X_inc'], tri['SAP_actifs'])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("μ̂", f"{res['mu_hat']:.4f}")
    c2.metric("σ̂²", f"{res['sigma2_hat']:.4f}")
    c3.metric("R²", f"{res['R2']:.4f}")
    c4.metric("Shapiro-Wilk (p)", f"{res['shapiro_p']:.4f}" if not np.isnan(res['shapiro_p']) else "—")

    st.subheader("Résultats par année de survenance")
    st.dataframe(
        res['results'].style.format({
            "Provision brute": "{:,.0f}", "SAP actifs": "{:,.0f}",
            "Provision nette": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Diagnostic des résidus")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(res['residuals'], bins=12, color=C_BLUE, alpha=0.75, edgecolor='white')
    axes[0].axvline(0, color='black', linewidth=1.2)
    axes[0].set_title("Histogramme des résidus")
    axes[0].set_xlabel("Résidu (échelle log)")

    from scipy import stats as sp_stats
    (osm, osr), (sl, ic, r2) = sp_stats.probplot(res['residuals'], dist='norm')
    axes[1].scatter(osm, osr, color=C_BLUE, alpha=0.8, s=40)
    axes[1].plot([min(osm), max(osm)], [sl * min(osm) + ic, sl * max(osm) + ic], color=C_RED, linewidth=2)
    axes[1].set_title(f"Q-Q Plot (R²={r2**2:.3f})")
    axes[1].set_xlabel("Quantiles théoriques N(0,1)")
    axes[1].set_ylabel("Quantiles empiriques")
    st.pyplot(fig, clear_figure=True)

    st.metric("Provision nette totale (Log-Normal)", fmt_money(res['prov_nette'].sum()))


# ===========================================================================
# PAGE : SÉLECTION DU MODÈLE (AIC quasi)
# ===========================================================================
elif section == "🏆 Sélection du modèle":
    page_header(
        "🏆 Sélection du modèle stochastique",
        "Comparaison Poisson surdispersé / Gamma / Log-Normal via la "
        "quasi-vraisemblance étendue (AIC, BIC).",
    )
    data = require_data()
    if data is None:
        st.stop()
    rcc, tri, primes = data

    glm_res = stochastic.run_glm_poisson_gamma(tri['X_inc'], tri['SAP_actifs'])
    ln_res = stochastic.run_lognormal(tri['X_inc'], tri['SAP_actifs'])
    table = stochastic.model_selection_table(glm_res, ln_res)

    st.dataframe(
        table.style.format({
            "Déviance / SCE résidus": "{:,.2f}",
            "phi (dispersion)": "{:.4f}",
            "Quasi log-vraisemblance": "{:.2f}",
            "AIC quasi": "{:.2f}",
            "BIC quasi": "{:.2f}",
        }),
        use_container_width=True, hide_index=True,
    )

    best_model = table.loc[table["AIC quasi"].idxmin(), "Modèle"]
    st.success(f"✅ Modèle retenu selon le critère AIC quasi (le plus faible) : **{best_model}**")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors_m = [C_BLUE, C_GREEN, C_ORANGE]
    axes[0].bar(table["Modèle"], table["AIC quasi"], color=colors_m, edgecolor='white')
    axes[0].set_title("AIC quasi (à minimiser)")
    axes[0].tick_params(axis='x', rotation=15)
    axes[1].bar(table["Modèle"], table["BIC quasi"], color=colors_m, edgecolor='white')
    axes[1].set_title("BIC quasi (à minimiser)")
    axes[1].tick_params(axis='x', rotation=15)
    st.pyplot(fig, clear_figure=True)

    with st.expander("ℹ️ Comment interpréter ces critères ?"):
        st.markdown(
            "- **AIC / BIC quasi** : étendent les critères d'Akaike et de "
            "Schwarz au cadre de la quasi-vraisemblance, utilisable même "
            "lorsque la loi exacte n'est pas entièrement spécifiée "
            "(cas du Poisson surdispersé).\n"
            "- Plus la valeur est **faible**, meilleur est l'arbitrage "
            "qualité d'ajustement / parcimonie du modèle.\n"
            "- Le **BIC** pénalise davantage la complexité que l'AIC pour "
            "des échantillons de taille modérée."
        )


# ===========================================================================
# PAGE : BOOTSTRAP
# ===========================================================================
elif section == "🔁 Bootstrap":
    st.title("🔁 Bootstrap Chain Ladder — England & Verrall")
    data = require_data()
    if data is None:
        st.stop()
    rcc, tri, primes = data

    st.markdown("Paramètres de la simulation :")
    c1, c2 = st.columns(2)
    B = c1.slider("Nombre de simulations B", min_value=100, max_value=5000, value=1000, step=100)
    seed = c2.number_input("Graine aléatoire (seed)", min_value=0, value=2024, step=1)

    f_cl = methods.chain_ladder_factors(tri['CHARGE']) if hasattr(methods, 'chain_ladder_factors') \
        else None
    from utils.core import chain_ladder_factors as _clf
    f_cl = _clf(tri['CHARGE'])

    with st.spinner(f"Simulation de {B:,} tirages bootstrap..."):
        res = boot_mod.run_bootstrap(tri['CHARGE'], f_cl, B=B, seed=int(seed))

    provisions_boot = res['provisions_boot']
    total_cl = res['total_cl']

    dstats = boot_mod.dist_stats(provisions_boot)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Moyenne bootstrap", fmt_money(dstats['Moyenne']))
    c2.metric("Médiane bootstrap", fmt_money(dstats['Médiane']))
    c3.metric("Référence Chain Ladder", fmt_money(total_cl))
    c4.metric("CV (%)", f"{dstats['CV (%)']:.2f}%")

    st.subheader("Mesures de risque (VaR / TVaR)")
    levels = [75, 90, 95, 99, 99.5]
    rows = []
    for lv in levels:
        var, tvar = boot_mod.var_tvar(provisions_boot, lv)
        rows.append({"Niveau": f"{lv}%", "VaR": var, "TVaR": tvar})
    risk_df = pd.DataFrame(rows)
    st.dataframe(risk_df.style.format({"VaR": "{:,.0f}", "TVaR": "{:,.0f}"}),
                 use_container_width=True, hide_index=True)

    st.subheader("Distribution de la provision totale")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    vals = provisions_boot / 1e6
    ax.hist(vals, bins=50, color=C_BLUE, alpha=0.65, edgecolor='white', density=True)
    ax.axvline(total_cl / 1e6, color=C_RED, linewidth=2.5, linestyle='--',
               label=f"CL réf. = {total_cl/1e6:.2f} M")
    ax.axvline(np.mean(vals), color=C_GREEN, linewidth=2, label=f"Moyenne = {np.mean(vals):.2f} M")
    q95, q99 = np.percentile(vals, [95, 99])
    ax.axvline(q95, color=C_ORANGE, linewidth=1.8, linestyle=':', label=f"VaR 95% = {q95:.2f} M")
    ax.axvline(q99, color='#9C0006', linewidth=1.5, linestyle=':', label=f"VaR 99% = {q99:.2f} M")
    ax.set_xlabel("Provision totale ΣIBNR (M)")
    ax.set_ylabel("Densité")
    ax.legend(fontsize=8)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Distribution IBNR par année de survenance")
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    bp_data = [res['ibnr_by_ay_boot'][:, i] / 1e6 for i in range(N)]
    bp = ax2.boxplot(bp_data, patch_artist=True,
                      medianprops=dict(color='#1F3864', linewidth=2))
    for patch in bp['boxes']:
        patch.set_facecolor('#BDD7EE')
        patch.set_alpha(0.8)
    ax2.scatter(range(1, N + 1), res['ibnr_cl'] / 1e6, color=C_RED, marker='x',
                s=60, zorder=5, label="CL exact")
    ax2.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax2.set_xticks(range(1, N + 1))
    ax2.set_xticklabels([str(y)[-2:] for y in YEARS])
    ax2.set_xlabel("Année de survenance")
    ax2.set_ylabel("IBNR (M)")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)


# ===========================================================================
# PAGE : COMPARAISON DES MÉTHODES
# ===========================================================================
elif section == "🏁 Comparaison des méthodes":
    st.title("🏁 Comparaison des méthodes — IBNR total")
    data = require_data()
    if data is None:
        st.stop()
    rcc, tri, primes = data

    cl_res = methods.run_chain_ladder(tri['CHARGE'])
    rows = [("Chain Ladder", cl_res['ibnr'].sum())]

    if primes is not None:
        cc_res = methods.run_cape_cod(tri['CHARGE'], primes)
        bf_res = methods.run_bornhuetter_ferguson(tri['CHARGE'], primes)
        rows.append(("Cape Cod", cc_res['ibnr_cc'].sum()))
        rows.append(("Bornhuetter-Ferguson", bf_res['ibnr_bf'].sum()))
    else:
        st.info("Importez le fichier des primes pour inclure Cape Cod et Bornhuetter-Ferguson.")

    if 'DELAI_DEC' in rcc.columns:
        try:
            cm_res = methods.run_cout_moyen(rcc, tri['REG_cum'], tri['SAP_inc'])
            rows.append(("Coût Moyen", cm_res['IBNR_total'].sum()))
        except Exception:
            pass

    glm_res = stochastic.run_glm_poisson_gamma(tri['X_inc'], tri['SAP_actifs'])
    rows.append(("GLM Poisson", glm_res['prov_P_net'].sum()))
    rows.append(("GLM Gamma", glm_res['prov_G_net'].sum()))

    ln_res = stochastic.run_lognormal(tri['X_inc'], tri['SAP_actifs'])
    rows.append(("Log-Normal", ln_res['prov_nette'].sum()))

    comp_df = pd.DataFrame(rows, columns=["Méthode", "IBNR / Provision totale"])
    st.dataframe(comp_df.style.format({"IBNR / Provision totale": "{:,.0f}"}),
                 use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(comp_df)))
    bars = ax.bar(comp_df["Méthode"], comp_df["IBNR / Provision totale"] / 1e6,
                   color=colors, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, comp_df["IBNR / Provision totale"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val/1e6:.2f}M", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_ylabel("IBNR / Provision (M)")
    ax.tick_params(axis='x', rotation=20)
    st.pyplot(fig, clear_figure=True)

    st.subheader("🔁 Inclure le Bootstrap dans la comparaison")
    if st.checkbox("Lancer le Bootstrap (B=1000) pour ajouter médiane / VaR 95% au graphique"):
        from utils.core import chain_ladder_factors as _clf
        f_cl = _clf(tri['CHARGE'])
        with st.spinner("Simulation bootstrap..."):
            bres = boot_mod.run_bootstrap(tri['CHARGE'], f_cl, B=1000, seed=2024)
        med = np.median(bres['provisions_boot'])
        var95, _ = boot_mod.var_tvar(bres['provisions_boot'], 95)
        extra = pd.DataFrame({
            "Méthode": ["Bootstrap (médiane)", "Bootstrap (VaR 95%)"],
            "IBNR / Provision totale": [med, var95],
        })
        full_df = pd.concat([comp_df, extra], ignore_index=True)
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        colors2 = plt.cm.tab10(np.linspace(0, 1, len(full_df)))
        bars2 = ax2.bar(full_df["Méthode"], full_df["IBNR / Provision totale"] / 1e6,
                         color=colors2, edgecolor='white', linewidth=1.2)
        for bar, val in zip(bars2, full_df["IBNR / Provision totale"]):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{val/1e6:.2f}M", ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax2.set_ylabel("IBNR / Provision (M)")
        ax2.tick_params(axis='x', rotation=25)
        st.pyplot(fig2, clear_figure=True)
