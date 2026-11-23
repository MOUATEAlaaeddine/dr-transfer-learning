# RetinaLens · Démarrer ici

Page commune de l'équipe. Les contrats de code détaillés sont dans [INTERFACES.md](INTERFACES.md),
votre lot dans votre guide : [Abdelati](abdelati.md) · [Ikram](ikram.md) · [Youssef](youssef.md).

## État au 23 novembre 2026

| Brique | État |
|---|---|
| Stratégies de gel, groupes de learning rate, CNN from scratch (`models.py`) | prêt, testé |
| Boucle d'entraînement : AdamW, warm-up + cosine, poids de classes, AMP, early stopping, reprise (`train.py`) | prêt, testé |
| Lancement d'un run + fiche de résultats (`run.py`, `results.py`) | prêt, testé |
| `notebooks/02_train_and_evaluate.ipynb` | prêt, exécuté de bout en bout en mode smoke sur données SIMULATED |
| Découpage `folds.csv`, `transforms.py`, `data.py` | **en attente** · Abdelati |
| Métriques complètes, agrégation, robustesse | en attente · Ikram |
| Grad-CAM, démo Gradio | en attente · Youssef |
| Entraînements réels sur Kaggle | **0 / 65**, bloqués par `folds.csv` + `data.py` |

Aucun résultat réel n'existe encore. Tout chiffre issu des données synthétiques est étiqueté
**SIMULATED** et n'entre jamais dans `results/`.

## Qui fait quoi

| Membre | Branche | Entraînements CV | Attend |
|---|---|---:|---|
| Alaa Eddine | `feat/shared-training` | 15 · ViT-B/16 | `folds.csv` + `data.py` (Abdelati) |
| Abdelati | `feat/data-splits` | 20 · CNN scratch (5) + VGG19 (15) | le dataset Kaggle, puis le pipeline fusionné |
| Ikram | `feat/evaluation` | 15 · DenseNet-201 | le format des fiches de run (déjà fixé, §6 d'INTERFACES) |
| Youssef | `feat/retinalens-demo` | 15 · MobileNetV4 (pilote commun) | `transforms.py`, puis de vrais checkpoints |

Ordre : (1) Abdelati livre `folds.csv` + `data.py` → (2) Youssef lance le **pilote MobileNetV4 /
feature extraction / fold 0** → (3) on vérifie ses sorties ensemble → (4) chacun lance le fold 0 de
chacune de ses configurations → (5) les folds 1 à 4. Chaque PR est relue par une personne.

## Alaa · prochaines actions

1. Pousser `feat/shared-training`, ouvrir la PR vers `main`, demander la relecture des interfaces à
   Abdelati (`data.py`) et Ikram (fiches de run) ; fusionner après approbation.
2. Exécuter `notebooks/00_setup_check.ipynb` sur le Mac et sur Kaggle (la cellule des poids ImageNet
   a besoin d'Internet).
3. Après le pilote de Youssef : contrôler ses sorties, publier l'estimation du budget GPU ici,
   puis lancer `vit_b16 / feature_extraction / fold 0` et la suite des 15 runs ViT.
4. Rédiger Methodology (paramètres et stratégies déjà fixés dans le code) ; Conclusion seulement
   quand les résultats réels existent.

## Installation locale (code, tests, démo)

L'entraînement évalué se fait **uniquement sur Kaggle (GPU CUDA)**, pour que Mac et Windows ne
changent pas le protocole. En local : coder, tester, lancer un smoke run, la démo.

**Windows · PowerShell**, depuis le dossier où vous voulez le dépôt

```powershell
git clone https://github.com/MOUATEAlaaeddine/dr-transfer-learning.git
cd dr-transfer-learning
git switch -c feat/VOTRE-BRANCHE origin/main
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m drtl.run --model scratch_cnn --strategy scratch --fold 0 --smoke --synthetic
```

**Mac · Python arm64 natif** (`python3 -c "import platform; print(platform.machine())"` doit afficher `arm64` ;
un Anaconda Intel ne peut pas installer PyTorch récent)

```bash
git clone https://github.com/MOUATEAlaaeddine/dr-transfer-learning.git
cd dr-transfer-learning
git switch -c feat/VOTRE-BRANCHE origin/main
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q
.venv/bin/python -m drtl.run --model scratch_cnn --strategy scratch --fold 0 --smoke --synthetic
```

Attendu : `pytest` affiche `34 passed` (plus vos propres tests), le smoke run affiche 2 époques et
`status smoke`. Le smoke run écrit dans `outputs/smoke/` (ignoré par Git).

Tant que `feat/shared-training` n'est pas fusionnée dans `main`, partez de
`origin/feat/shared-training` à la place de `origin/main`, puis faites `git rebase origin/main` après la fusion.

## Entraîner sur Kaggle (identique pour tous)

1. **Compte** : téléphone vérifié (obligatoire pour Internet et GPU). Quota visible dans *Settings → Quotas*.
2. **Créer** : kaggle.com → *Create → New Notebook* → *File → Import Notebook* → collez
   `https://github.com/MOUATEAlaaeddine/dr-transfer-learning/blob/main/notebooks/02_train_and_evaluate.ipynb`.
3. **Réglages** (panneau de droite) : *Accelerator* GPU T4 ×2 ou P100 · *Internet* On ·
   *Add Input* → cherchez `jockeroika/diabetic-retinopathy` → *Add*.
4. **Paramètres** (cellule `Parameters`) : `MODEL`, `STRATEGY`, `FOLD`. Laissez `SMOKE = False`,
   `SYNTHETIC = False`, `BRANCH = "main"`.
5. **Vérifier vite** : *Run All* en interactif avec `SMOKE = True` (2 minutes) pour contrôler que le
   dataset est trouvé.
6. **Lancer le vrai run** : remettez `SMOKE = False` → *Save Version* → *Save & Run All (Commit)*.
   Le run tourne en arrière-plan (12 h max par session) et ses sorties restent dans l'onglet *Output*.
7. **Session coupée** : ajoutez l'Output de la version interrompue comme *Input*, mettez
   `RESUME_FROM = "/kaggle/input/<nom-du-notebook>/outputs"` et relancez : l'entraînement reprend à la
   dernière époque terminée.
8. **Récupérer** : dans *Output*, téléchargez `results_<run_id>.zip` (2 petits fichiers). Les poids
   (`best.pt`, `last.pt`) restent sur Kaggle.

Mémoire GPU insuffisante (VGG19, ViT complet) :
`OVERRIDES = {"training": {"batch_size": 16, "accumulation_steps": 2}}`. C'est noté dans la fiche du run.

## Résultats : une fiche par run, jamais de fichier partagé à la main

```text
results/runs/<modèle>__<stratégie>__fold<k>.json             # métriques + hashes + commit + durée + checkpoint
results/runs/<modèle>__<stratégie>__fold<k>_val_predictions.csv
```

Décompressez le zip dans `results/runs/` de votre branche, puis `git add results/runs` et
`git commit -m "exp: vit_b16__feature_extraction__fold0"`. Le tableau global
se régénère avec `python -m drtl.results` (les 65 runs attendus, `missing` pour ceux qui manquent).
Personne n'édite `results/runs.csv` à la main, et un run manquant n'est jamais remplacé par une estimation.

Contrôles de cohérence (Ikram les automatise) : même `config_hash` et même `splits_hash` sur toutes
les lignes, `status = completed`, `data_source = kaggle`.

## Budget GPU

Rien n'est mesuré pour l'instant. Après le pilote : durée d'un run = `duration_s` de sa fiche
(`sec_per_epoch` × `epochs_run`). Estimation de vos runs restants = Σ sur vos configurations de
(durée du fold 0 × 4). Comparez-la à votre quota restant et prévenez Alaa si elle dépasse : le repli
prévu (rapport §11) est k = 3 folds ou 20 époques max, **pour toute l'équipe à la fois**.

## Checklist de PR (commune)

- [ ] Branche à jour avec `main`, un seul sujet par PR, commits `feat:` / `fix:` / `docs:` / `exp:`
- [ ] `pytest -q` passe en local (coller la dernière ligne dans la PR)
- [ ] Notebooks : sorties effacées, exécutables de haut en bas dans un runtime neuf
- [ ] Aucune image, aucun poids (`*.pt`), aucun identifiant Kaggle dans le diff
- [ ] Fiches de run : `status completed`, mêmes hashes que les autres, lien vers la version Kaggle
- [ ] Tout chiffre fictif est étiqueté **SIMULATED**
- [ ] Description de PR : commandes lancées, tests passés, ce qui reste à faire
- [ ] Un relecteur assigné (Alaa par défaut ; Abdelati relit le code d'Alaa)

## Aperçus de design

Captures de référence du prototype Transfer Lab (valeurs SIMULATED) :
[bureau 1440 px](previews/transfer_lab_desktop.png) · [mobile 390 px](previews/transfer_lab_mobile.png) ·
écrans d'origine dans `docs/prototypes/transfer_lab/`.
