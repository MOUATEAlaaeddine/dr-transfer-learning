# Abdelati · Données, découpage, prétraitement · CNN scratch + VGG19

**Branche** `feat/data-splits` · **Runs** 20 (CNN scratch × 5 folds, VGG19 × 3 stratégies × 5 folds) ·
**Papier** Dataset, Preprocessing, Augmentation, Limitations · **Relecteur** Alaa

Toute l'équipe attend ton `folds.csv` et ton `data.py` : sans eux, aucun entraînement réel ne peut
démarrer. Vise une première PR « découpage + loaders » avant le 2 décembre, l'EDA complète ensuite.

## À livrer, dans cet ordre

1. **EDA sur Kaggle** (`notebooks/01_eda.ipynb`) : vrais noms de dossiers, images par classe, formats,
   tailles, fichiers illisibles, doublons (hachage perceptuel). Cherche s'il existe un identifiant
   patient vérifiable ; sinon `group_id = image_id` et tu l'écris dans les limites.
2. **`data/splits/folds.csv`** au format exact d'INTERFACES §1 : test 15 % stratifié, puis 5 folds
   stratifiés sur le reste, graine 42 (par groupe si `group_id` est réel). Corrige `class_names`
   dans `configs/default.yaml` si les dossiers diffèrent. Publie `python -c "from drtl.results import file_hash; print(file_hash('data/splits/folds.csv'))"` dans la PR.
3. **`src/drtl/transforms.py`** (INTERFACES §2) puis **`src/drtl/data.py`** (§3). Remplace dans
   `run.py` les deux constantes `IMAGENET_MEAN/STD` par un import depuis `transforms`.
4. **`tests/test_data.py`** : sortie `3×224×224`, moyenne ≈ 0 et écart-type ≈ 1, pas d'augmentation en
   val/test, aucune image commune entre test et folds, proportions de classes ±1 % par fold.
   `tests/test_integration_boundary.py` doit passer contre ton vrai module.
5. **Runs** après le pilote de Youssef : `scratch_cnn / scratch` folds 0–4, puis `vgg19` × 3 stratégies.
   VGG19 complet : vérifie la mémoire au fold 0 ; si besoin `batch_size 16` + `accumulation_steps 2`.

## Commandes · Windows PowerShell

```powershell
git clone https://github.com/MOUATEAlaaeddine/dr-transfer-learning.git
cd dr-transfer-learning
git switch -c feat/data-splits origin/main
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
# après avoir écrit data.py : un vrai mini-run local sur 4 batches (images téléchargées dans data/raw)
.\.venv\Scripts\python.exe -m drtl.run --model scratch_cnn --strategy scratch --fold 0 --smoke
```

Sur Windows, garde `num_workers` compatible avec 0 (pas de lambda dans le Dataset).

## Sur Kaggle

EDA : nouveau notebook, *Add Input* `jockeroika/diabetic-retinopathy`, GPU inutile.
Entraînement : guide commun [START_HERE](START_HERE.md#entraîner-sur-kaggle-identique-pour-tous) avec

| `MODEL` | `STRATEGY` | `FOLD` |
|---|---|---|
| `scratch_cnn` | `scratch` | 0 → 4 |
| `vgg19` | `feature_extraction`, `partial_finetuning`, `full_finetuning` | 0 → 4 |

## Terminé quand

- `folds.csv` fusionné, hash publié, plus jamais modifié ensuite
- `pytest -q` passe avec `test_data.py` et le test de frontière
- 20 fiches `results/runs/*.json` en `status completed` avec le même `splits_hash`
- Sections anglaises Dataset / Preprocessing / Augmentation / Limitations rédigées à partir de l'EDA

## PR : en plus de la checklist commune

- [ ] Figure de distribution des classes + galerie par grade (sans images du dataset dans Git : figures légères seulement)
- [ ] Décision documentée sur les doublons et les fichiers illisibles
- [ ] Alaa et Ikram ont validé l'interface `data.py` (commentaire dans la PR)

## Prompt à coller dans Claude

> Aide-moi à réaliser mon lot dans le dépôt `dr-transfer-learning`, sur la branche `feat/data-splits`.
> Lis d'abord `docs/team/START_HERE.md`, `docs/team/INTERFACES.md` (§1 à §3), `docs/team/abdelati.md`,
> `configs/default.yaml` et `src/drtl/run.py`, puis vérifie `git status` et ne perds aucun travail local.
> Respecte exactement les signatures d'INTERFACES ; ne modifie pas `models.py`, `train.py`, `results.py`.
> 1) Écris `notebooks/01_eda.ipynb`, exécutable de haut en bas sur Kaggle avec le dataset
> `jockeroika/diabetic-retinopathy` en Input : noms réels des classes, effectifs, formats, tailles,
> fichiers illisibles, doublons par hachage perceptuel, identifiant patient s'il est vérifiable.
> 2) Génère `data/splits/folds.csv` (test 15 % stratifié figé, 5 folds stratifiés sur le reste, graine 42,
> groupés par patient si possible) et affiche son hash avec `drtl.results.file_hash`.
> 3) Implémente `src/drtl/transforms.py` (recadrage du disque rétinien, 224 px, normalisation ImageNet,
> augmentations d'entraînement seulement : flips p=0,5, rotation ±20°, zoom 0,9–1,0, luminosité/contraste ±15 %)
> et `src/drtl/data.py` (`load_splits`, `make_fold_loaders`, `make_test_loader`), puis remplace les
> constantes de normalisation de `run.py` par un import.
> 4) Ajoute `tests/test_data.py` et fais passer `pytest -q`, y compris `tests/test_integration_boundary.py`.
> Donne-moi les commandes PowerShell exactes à lancer sur mon PC Windows.
> 5) Prépare en anglais les sections Dataset, Preprocessing, Augmentation et Limitations à partir des
> chiffres mesurés seulement. N'invente aucun chiffre ; tout exemple fictif est étiqueté SIMULATED.
> Termine par les fichiers modifiés, les tests passés, le hash de `folds.csv`, et une description de PR prête à relire.
