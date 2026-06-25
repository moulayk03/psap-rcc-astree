# 📊 PSAP RCC — Application Streamlit (ASTREE Assurances)

Application web interactive pour l'évaluation des Provisions Pour Sinistres
À Payer (PSAP) de la garantie **RCC**, années de survenance **2016–2025**.
Convertie depuis un notebook de recherche Python vers une app Streamlit
propre, mise en cache, et organisée en modules.

---

## 🏗️ Architecture du projet

```
streamlit_app/
│
├── app.py                     # Point d'entrée — UI, sidebar, routage des pages
├── requirements.txt           # Dépendances (testées en environnement isolé)
├── .streamlit/
│   └── config.toml            # Thème visuel (couleurs, police)
│
├── utils/                      # Logique métier — séparée de l'UI
│   ├── core.py                 # Chargement fichiers + construction triangles (cache)
│   ├── methods.py               # Méthodes déterministes : CL, Cape Cod, BF, Coût Moyen
│   ├── stochastic.py            # GLM Poisson/Gamma (IRLS) + Log-Normal + sélection AIC
│   ├── bootstrap.py             # Bootstrap Chain Ladder (England & Verrall)
│   └── ui.py                    # CSS custom + composants visuels réutilisables
│
├── test_data/                  # Données synthétiques pour tester sans données réelles
│   ├── Inventaire_auto_test.csv
│   └── Primes_acquises_test.xlsx
│
├── generate_test_data.py       # Script de génération du jeu de test synthétique
└── test_headless.py            # Validation de tous les modules sans lancer l'UI
```

### Pourquoi cette architecture ?

| Principe | Application |
|---|---|
| **Séparation UI / logique métier** | `app.py` ne contient que de l'affichage et du routage. Tous les calculs (NumPy/Pandas/SciPy) vivent dans `utils/`, testables indépendamment de Streamlit. |
| **Un fichier par famille de méthode** | `methods.py` (déterministe), `stochastic.py` (GLM/Log-Normal), `bootstrap.py` (Monte Carlo) — évite un fichier monolithique de 2000 lignes. |
| **Cache systématique** | Toute fonction coûteuse (lecture fichier, construction triangle, IRLS, simulation bootstrap) est décorée `@st.cache_data`. Rien n'est recalculé tant que les entrées (fichiers, paramètres) ne changent pas. |
| **Pas de variables globales partagées entre pages** | Chaque section recharge les résultats via les fonctions cache — pas de bugs de rafraîchissement liés à un état partagé incohérent. |
| **Garde-fous explicites** | `require_data()` / `require_primes()` centralisent les messages d'avertissement (`st.warning`) quand un fichier manque, plutôt que de laisser un `KeyError` remonter à l'utilisateur. |

---

## 🧭 Navigation de l'application

Sidebar → un `st.radio` pilote un grand `if/elif` dans `app.py` (pattern
recommandé pour une nav simple à plat, sans dépendance à
`st.navigation`/multipage qui imposerait un fichier par page) :

```
🏠 Accueil
📐 Chain Ladder
🎯 Cape Cod
📈 Bornhuetter-Ferguson
💰 Coût Moyen
🧮 GLM (Poisson / Gamma)
📉 Log-Normal
🏆 Sélection du modèle      ← compare AIC/BIC quasi des 3 modèles stochastiques
🔁 Bootstrap                ← simulation Monte Carlo, slider B configurable
🏁 Comparaison des méthodes  ← bar chart de toutes les méthodes + option Bootstrap
```

---

## 🚀 Lancer l'application

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Puis dans le navigateur :
1. Importer le CSV inventaire (`Inventaire auto 2025.csv`, séparateur `;`)
2. Importer le fichier Excel des primes (`Primes acquises.xlsx`)
3. Naviguer entre les méthodes via la sidebar

> Les méthodes **Cape Cod** et **Bornhuetter-Ferguson** nécessitent le
> fichier des primes. La méthode **Coût Moyen** nécessite que le CSV
> inventaire contienne la colonne `DATE_DEC` (délai de déclaration).

---

## 🧪 Tester sans les données réelles

Un jeu de données synthétique est fourni (`test_data/`) pour vérifier que
l'app fonctionne de bout en bout sans exposer les données confidentielles
ASTREE :

```bash
python generate_test_data.py     # régénère les fichiers de test si besoin
python test_headless.py          # exécute tous les modules de calcul hors UI
streamlit run app.py             # puis importer les fichiers de test_data/
```

`test_headless.py` appelle directement les fonctions de `utils/` (sans
serveur Streamlit) et affiche ✅/❌ pour chacune des 10 étapes du pipeline
(chargement → triangles → 4 méthodes déterministes → GLM → Log-Normal →
sélection de modèle → Bootstrap). C'est le moyen le plus rapide de détecter
une régression après modification du code.

---

## ⚙️ Détails techniques notables

- **IRLS (GLM Poisson/Gamma)** : implémentation manuelle (pas de
  `statsmodels`) pour rester fidèle au notebook d'origine et garder un
  contrôle total sur la convergence et le calcul du paramètre de
  dispersion φ̂.
- **Sélection de modèle (AIC/BIC quasi)** : nécessite que les trois modèles
  (Poisson, Gamma, Log-Normal) soient ajustés sur le **même ensemble
  d'observations** (cellules à montant strictement positif) — c'est
  garanti par construction dans `run_glm_poisson_gamma()` et
  `run_lognormal()`.
- **Bootstrap** : le nombre de simulations `B` et la graine aléatoire
  `seed` sont des widgets (`st.slider`, `st.number_input`), donc
  modifiables sans toucher au code ; le cache est invalidé automatiquement
  si l'utilisateur change `B` ou `seed`.
- **Méthode Coût Moyen** : dépend de la colonne `DATE_DEC` de l'inventaire
  pour reconstruire le délai de déclaration ; un message d'erreur explicite
  s'affiche si la colonne est absente plutôt qu'un `KeyError`.

---

## 📌 Limites connues / extensions possibles

- L'app gère uniquement la garantie **RCC**. Pour étendre à RCM et DOM
  (comme dans le mémoire), il suffirait d'ajouter un `st.selectbox`
  "Garantie" dans la sidebar et de paramétrer `load_inventaire()` pour
  filtrer sur la garantie choisie plutôt que sur `'RCC'` en dur.
- Les graphiques utilisent Matplotlib (`st.pyplot`) pour rester fidèles
  aux figures du notebook ; une migration vers Plotly (`st.plotly_chart`)
  rendrait les graphiques zoomables/interactifs si souhaité.
