"""Génère un jeu de données synthétique pour tester l'application sans les
vraies données ASTREE (confidentielles)."""
import numpy as np
import pandas as pd

np.random.seed(42)
years = list(range(2016, 2026))
n = 10

rows = []
sin_id = 1
for ay in years:
    n_claims = np.random.randint(150, 400)
    for _ in range(n_claims):
        delai_dec = np.random.choice(range(0, min(4, 2026 - ay)), p=None) if min(4, 2026-ay) > 0 else 0
        annee_dec = ay + delai_dec
        if annee_dec > 2025:
            continue
        date_dec = f"01/01/{annee_dec}"

        # Sinistre se développe sur plusieurs années
        max_dev = 2025 - ay
        dev_years_for_claim = np.random.randint(1, max(2, max_dev + 1))
        ultimate_cost = np.random.lognormal(mean=8.5, sigma=1.2)

        cum_reg = 0.0
        for d in range(min(dev_years_for_claim + 1, max_dev + 1)):
            cal_year = ay + d
            if cal_year > 2025:
                break
            # paiement incrémental
            pay_frac = np.random.dirichlet(np.ones(dev_years_for_claim + 1))[d] if d <= dev_years_for_claim else 0
            mnt_reg = ultimate_cost * pay_frac if d <= dev_years_for_claim else 0.0
            mnt_sap = max(ultimate_cost - cum_reg - mnt_reg, 0) * np.random.uniform(0.3, 0.9) \
                if d == min(dev_years_for_claim, max_dev) else 0.0
            cum_reg += mnt_reg

            rows.append({
                "ID_SINISTRE": sin_id,
                "CODE_GAR": "RCC",
                "ANNEE_SURV": ay,
                "ANNEE": cal_year,
                "MNT_REG": round(mnt_reg, 2),
                "MNT_SAP": round(mnt_sap, 2),
                "DATE_DEC": date_dec,
            })
        sin_id += 1

df = pd.DataFrame(rows)
df.to_csv("/home/claude/streamlit_app/test_data/Inventaire_auto_test.csv",
          sep=";", index=False)

# Primes acquises
primes_rows = []
base_prime = 15_000_000
for i, y in enumerate(years):
    rc_auto = base_prime * (1.05 ** i) * np.random.uniform(0.95, 1.05)
    dom_auto = rc_auto * 0.6
    primes_rows.append({
        "ANNEE": y,
        "RC Auto": round(rc_auto, 2),
        "Dom Auto": round(dom_auto, 2),
        "TOTAL": round(rc_auto + dom_auto, 2),
    })
primes_df = pd.DataFrame(primes_rows)
primes_df.to_excel("/home/claude/streamlit_app/test_data/Primes_acquises_test.xlsx", index=False)

print("Lignes générées :", len(df))
print(df.head(10).to_string())
print()
print(primes_df.to_string())
