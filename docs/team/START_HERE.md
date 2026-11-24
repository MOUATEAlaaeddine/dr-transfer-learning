# RetinaLens · Démarrer ici

Contrats de code : [INTERFACES.md](INTERFACES.md). Tout le code et tous les entraînements sont
réalisés par MOUATE ALAA EDDINE ; le tableau des contributions du papier reflète ce que chacun fait réellement.

## Chaîne complète (Mac Apple Silicon ou Kaggle)

```bash
.venv/bin/python -m pip install -e . && .venv/bin/python -m pytest -q        # tests
.venv/bin/python scripts/prepare_data.py --raw data/raw                       # EDA, doublons, split figé, cache 224 px
caffeinate -i .venv/bin/python scripts/run_grid.py                            # 65 runs CV, reprise automatique
.venv/bin/python -m drtl.evaluate                                             # runs.csv + summary.csv
.venv/bin/python scripts/final_evaluation.py                                  # test figé, robustesse, coût, figures
.venv/bin/python -m pip install -r requirements-app.txt && .venv/bin/python app/app.py   # démo
```

Windows PowerShell : mêmes commandes avec `.\.venv\Scripts\python.exe` à la place de `.venv/bin/python`
(sans `caffeinate`). Création de l'environnement : `py -m venv .venv` puis
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

Suivi pendant l'entraînement : `outputs/grid_status.json` (runs terminés, F1 de validation, erreurs),
`outputs/grid.log`. Un run interrompu reprend à sa dernière époque ; un run terminé n'est jamais relancé.

## Protocole (identique pour tous les modèles)

Graine 42 · 224 × 224 · test 15 % stratifié figé, jamais utilisé pour choisir · 5 folds stratifiés,
doublons regroupés dans la même partie · augmentation et poids de classes calculés sur les folds
d'entraînement seulement · checkpoint = meilleure F1 macro de validation · early stopping sur la loss
de validation (patience 5) · sélection de la stratégie par F1 macro moyenne en validation, puis une seule
évaluation sur le test (ensemble des 5 folds).

## Rôles restants pour l'équipe (contributions réelles à créditer)

Rédaction des sections du papier en anglais, références bibliographiques, relecture croisée des PR et
du papier, diapositives et répétitions de la soutenance. Chacun remplit sa ligne du tableau des
contributions avec ce qu'il a effectivement fait.

## Checklist avant merge

- [ ] `pytest -q` passe ; notebooks sans sorties, exécutables de haut en bas
- [ ] Aucune image du dataset, aucun poids (`*.pt`), aucun identifiant dans le diff
- [ ] Fiches de run : `status completed`, mêmes `config_hash` et `splits_hash`
- [ ] Tout chiffre fictif étiqueté **SIMULATED** ; aucun résultat manquant remplacé par une estimation

## Aperçus

Prototype : [bureau](previews/transfer_lab_desktop.png) · [mobile](previews/transfer_lab_mobile.png).
Démo Gradio en mode exemple SIMULATED : [bureau](previews/app_desktop.png) · [mobile](previews/app_mobile.png).
