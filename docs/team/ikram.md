# Ikram · Évaluation, robustesse, figures · DenseNet-201

**Branche** `feat/evaluation` · **Runs** 15 (DenseNet-201 × 3 stratégies × 5 folds) ·
**Papier** Metrics, Results, Comparative Analysis · **Relecteur** Alaa

Tu peux commencer tout de suite : le format des fiches de run est fixé (INTERFACES §6) et des
fiches SIMULATED de test se génèrent sans dataset.

## À livrer, dans cet ordre

1. **`src/drtl/evaluate.py`** (INTERFACES §7) : accuracy, précision / rappel / F1 macro, ROC-AUC
   un-contre-tous, matrice de confusion, QWK. Classe absente : AUC exclue de la moyenne et listée
   dans `absent_classes`, jamais remplacée par 0,5.
2. **`tests/test_evaluate.py`** : résultats identiques à scikit-learn sur des cas connus ; somme de
   la matrice = nombre d'images ; cas avec une classe absente.
3. **Agrégation** (`python -m drtl.evaluate` ou une fonction appelée par un notebook) : lit
   `results/runs/*.json` + `*_val_predictions.csv`, recalcule tout depuis les prédictions, écrit
   `results/summary.csv` (moyenne ± écart-type par configuration, `n_folds` présents), signale les
   folds manquants et vérifie que `config_hash` / `splits_hash` sont identiques partout.
4. **`src/drtl/robustness.py`** (INTERFACES §8) : flou, sous-exposition, bruit, JPEG, niveaux 0–4.
   Youssef réutilise la même fonction pour le curseur de la démo.
5. **Runs DenseNet-201** après le pilote de Youssef.
6. **Évaluation finale** (après les 65 runs) : sélection par F1 macro moyen en validation, puis une
   seule évaluation sur le test figé pour la configuration retenue de chaque modèle, plus
   `results/robustness.csv`. Le test ne sert jamais à choisir.

Tester sans dataset (fiches SIMULATED, hors `results/`) :

```powershell
.\.venv\Scripts\python.exe -m drtl.run --model densenet201 --strategy feature_extraction --fold 0 --smoke --synthetic
# fiche : outputs\smoke\densenet201__feature_extraction__fold0\record\
```

## Commandes · Windows PowerShell

```powershell
git clone https://github.com/MOUATEAlaaeddine/dr-transfer-learning.git
cd dr-transfer-learning
git switch -c feat/evaluation origin/main
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m drtl.results        # régénère results\runs.csv (65 lignes, "missing" pour l'instant)
```

## Sur Kaggle

Guide commun [START_HERE](START_HERE.md#entraîner-sur-kaggle-identique-pour-tous) avec
`MODEL = "densenet201"`, `STRATEGY` ∈ `feature_extraction`, `partial_finetuning`, `full_finetuning`, `FOLD` 0 → 4.

## Terminé quand

- `evaluate.py` + tests fusionnés ; agrégation reproductible depuis les seuls fichiers de `results/`
- 15 fiches DenseNet `completed` avec les mêmes hashes que les autres
- `summary.csv`, `robustness.csv`, figures (courbes, matrices, ROC) générés par le code, chaque
  chiffre du papier traçable à un fichier de `results/`

## PR : en plus de la checklist commune

- [ ] Métriques recalculées depuis les prédictions = valeurs des fiches (tolérance 1e-6)
- [ ] Folds manquants affichés comme manquants, jamais interpolés
- [ ] Alaa a validé le schéma lu ; Youssef a validé `degrade()`

## Prompt à coller dans Claude

> Aide-moi à réaliser mon lot dans le dépôt `dr-transfer-learning`, sur la branche `feat/evaluation`.
> Lis d'abord `docs/team/START_HERE.md`, `docs/team/INTERFACES.md` (§6 à §8), `docs/team/ikram.md`,
> `src/drtl/results.py` et `src/drtl/train.py`, puis vérifie `git status` et ne perds aucun travail local.
> Ne modifie ni le schéma des fiches de run ni `run.py` ; si un champ manque, propose-le à Alaa.
> 1) Implémente `src/drtl/evaluate.py` : accuracy, précision/rappel/F1 macro (labels 0–4), ROC-AUC
> multiclasse one-vs-rest avec gestion explicite des classes absentes, matrice de confusion, QWK.
> 2) Écris `tests/test_evaluate.py` (comparaison avec scikit-learn, somme de la matrice, classe absente).
> 3) Ajoute l'agrégation : lire `results/runs/*.json` et les CSV de prédictions, recalculer les métriques,
> produire `results/summary.csv` (moyenne ± écart-type sur les folds présents, folds manquants signalés,
> contrôle des hashes). Teste-la avec des fiches générées par
> `python -m drtl.run --model densenet201 --strategy feature_extraction --fold 0 --smoke --synthetic`
> en les gardant hors de `results/`.
> 4) Implémente `src/drtl/robustness.py` avec `degrade(image, kind, level)` (flou, sous-exposition, bruit, JPEG ; niveaux 0–4).
> 5) Prépare le notebook des figures (courbes, matrices, ROC) qui n'affiche que des mesures réelles.
> Donne-moi les commandes PowerShell exactes. Rédige en anglais Metrics, Results et Comparative Analysis
> à partir des seuls résultats disponibles ; marque clairement ce qui manque. Tout exemple fictif est étiqueté SIMULATED.
> Termine par les fichiers modifiés, les tests passés et une description de PR prête à relire.
