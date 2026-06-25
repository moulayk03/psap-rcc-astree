"""
test_headless.py — Valide tous les modules de calcul sans passer par l'UI
Streamlit (st.cache_data fonctionne aussi hors serveur, il se contente de ne
pas persister entre exécutions).
"""
import sys
import traceback
import numpy as np

sys.path.insert(0, "/home/claude/streamlit_app")

from utils.core import (
    load_inventaire, load_primes, build_triangles,
    chain_ladder_factors, YEARS, N,
)
from utils import methods, stochastic, bootstrap as boot_mod

OK = "✅"
FAIL = "❌"


def section(name):
    print(f"\n{'='*70}\n{name}\n{'='*70}")


def main():
    errors = []

    section("1. Chargement des données")
    with open("/home/claude/streamlit_app/test_data/Inventaire_auto_test.csv", "rb") as f:
        inv_bytes = f.read()
    with open("/home/claude/streamlit_app/test_data/Primes_acquises_test.xlsx", "rb") as f:
        primes_bytes = f.read()

    try:
        rcc = load_inventaire(inv_bytes)
        print(f"{OK} Inventaire chargé : {len(rcc)} lignes")
    except Exception as e:
        print(f"{FAIL} load_inventaire: {e}")
        traceback.print_exc()
        errors.append("load_inventaire")
        return errors

    try:
        prime_dict = load_primes(primes_bytes)
        primes = np.array([prime_dict[y] for y in YEARS])
        print(f"{OK} Primes chargées : {primes}")
    except Exception as e:
        print(f"{FAIL} load_primes: {e}")
        traceback.print_exc()
        errors.append("load_primes")
        primes = None

    section("2. Construction des triangles")
    try:
        tri = build_triangles(rcc)
        print(f"{OK} Triangles construits. CHARGE shape={tri['CHARGE'].shape}")
        print("CHARGE[0]:", np.round(tri['CHARGE'][0], 1))
    except Exception as e:
        print(f"{FAIL} build_triangles: {e}")
        traceback.print_exc()
        errors.append("build_triangles")
        return errors

    section("3. Chain Ladder")
    try:
        res_cl = methods.run_chain_ladder(tri['CHARGE'])
        print(f"{OK} IBNR total CL = {res_cl['ibnr'].sum():,.0f}")
    except Exception as e:
        print(f"{FAIL} run_chain_ladder: {e}")
        traceback.print_exc()
        errors.append("run_chain_ladder")

    if primes is not None:
        section("4. Cape Cod")
        try:
            res_cc = methods.run_cape_cod(tri['CHARGE'], primes)
            print(f"{OK} ELR = {res_cc['ELR']:.4f}, IBNR total = {res_cc['ibnr_cc'].sum():,.0f}")
        except Exception as e:
            print(f"{FAIL} run_cape_cod: {e}")
            traceback.print_exc()
            errors.append("run_cape_cod")

        section("5. Bornhuetter-Ferguson")
        try:
            res_bf = methods.run_bornhuetter_ferguson(tri['CHARGE'], primes)
            print(f"{OK} IBNR total BF = {res_bf['ibnr_bf'].sum():,.0f}")
        except Exception as e:
            print(f"{FAIL} run_bornhuetter_ferguson: {e}")
            traceback.print_exc()
            errors.append("run_bornhuetter_ferguson")

    section("6. Coût Moyen")
    try:
        res_cm = methods.run_cout_moyen(rcc, tri['REG_cum'], tri['SAP_inc'])
        print(f"{OK} IBNR total CM = {res_cm['IBNR_total'].sum():,.0f}")
    except Exception as e:
        print(f"{FAIL} run_cout_moyen: {e}")
        traceback.print_exc()
        errors.append("run_cout_moyen")

    section("7. GLM Poisson / Gamma")
    try:
        res_glm = stochastic.run_glm_poisson_gamma(tri['X_inc'], tri['SAP_actifs'])
        print(f"{OK} Provision nette Poisson = {res_glm['prov_P_net'].sum():,.0f}")
        print(f"{OK} Provision nette Gamma   = {res_glm['prov_G_net'].sum():,.0f}")
    except Exception as e:
        print(f"{FAIL} run_glm_poisson_gamma: {e}")
        traceback.print_exc()
        errors.append("run_glm_poisson_gamma")
        res_glm = None

    section("8. Log-Normal")
    try:
        res_ln = stochastic.run_lognormal(tri['X_inc'], tri['SAP_actifs'])
        print(f"{OK} Provision nette Log-Normal = {res_ln['prov_nette'].sum():,.0f}")
        print(f"   sigma2={res_ln['sigma2_hat']:.4f}, R2={res_ln['R2']:.4f}")
    except Exception as e:
        print(f"{FAIL} run_lognormal: {e}")
        traceback.print_exc()
        errors.append("run_lognormal")
        res_ln = None

    if res_glm is not None and res_ln is not None:
        section("9. Sélection du modèle (AIC quasi)")
        try:
            table = stochastic.model_selection_table(res_glm, res_ln)
            print(table.to_string(index=False))
        except Exception as e:
            print(f"{FAIL} model_selection_table: {e}")
            traceback.print_exc()
            errors.append("model_selection_table")

    section("10. Bootstrap (B=200 pour rapidité du test)")
    try:
        f_cl = chain_ladder_factors(tri['CHARGE'])
        res_boot = boot_mod.run_bootstrap(tri['CHARGE'], f_cl, B=200, seed=42)
        print(f"{OK} Moyenne bootstrap = {np.mean(res_boot['provisions_boot']):,.0f}")
        print(f"   Référence CL      = {res_boot['total_cl']:,.0f}")
        var95, tvar95 = boot_mod.var_tvar(res_boot['provisions_boot'], 95)
        print(f"   VaR95={var95:,.0f}  TVaR95={tvar95:,.0f}")
    except Exception as e:
        print(f"{FAIL} run_bootstrap: {e}")
        traceback.print_exc()
        errors.append("run_bootstrap")

    section("RÉSUMÉ")
    if errors:
        print(f"{FAIL} {len(errors)} module(s) en erreur : {errors}")
    else:
        print(f"{OK} Tous les modules fonctionnent correctement.")
    return errors


if __name__ == "__main__":
    errs = main()
    sys.exit(1 if errs else 0)
