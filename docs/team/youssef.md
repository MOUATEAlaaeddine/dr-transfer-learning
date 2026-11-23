# Youssef · Grad-CAM et démo Gradio · MobileNetV4 (pilote commun)

**Branche** `feat/retinalens-demo` · **Runs** 15 (MobileNetV4 × 3 stratégies × 5 folds) ·
**Papier** Introduction, Related Work, Interpretability, Perspectives · **Relecteur** Alaa

Ton premier run est le **pilote de toute l'équipe** : MobileNetV4 / feature extraction / fold 0,
dès que `folds.csv` et `data.py` d'Abdelati sont fusionnés. Il mesure la durée par époque et
vérifie la chaîne complète avant que les autres lancent leurs runs.

## À livrer, dans cet ordre

1. **Pilote** (START_HERE, étape Kaggle) puis partage dans le canal d'équipe : `sec_per_epoch`,
   `epochs_run`, `val_f1_macro`, `config_hash`, `splits_hash`, lien de la version Kaggle.
2. **`src/drtl/gradcam.py`** (INTERFACES §9) avec `pytorch-grad-cam` (à ajouter dans `requirements.txt`) :
   dernière couche convolutive pour les CNN ; pour ViT-B/16, `blocks[-1].norm1` + reshape 14 × 14
   (ou attention rollout), méthode toujours nommée dans le libellé. v0 tout de suite sur des modèles
   non entraînés pour tester la mécanique.
3. **`tests/test_gradcam.py`** : forme et plage [0, 1] de la carte ; la carte change quand la classe ciblée change.
4. **`app/app.py`** (Gradio) selon la spécification ci-dessous, puis `notebooks/03_gradcam.ipynb`.
5. **Runs MobileNetV4** restants (3 stratégies × 5 folds).

## Spécification de la démo (à partir du prototype Transfer Lab)

Références : `docs/prototypes/transfer_lab/index.html` et ses captures, plus les aperçus
[bureau](previews/transfer_lab_desktop.png) / [mobile](previews/transfer_lab_mobile.png).

**Onglet « Analyze an image »**
1. **Entrée** : upload JPG/PNG (ou exemple), choix du backbone parmi ceux qui ont des checkpoints,
   curseur « qualité de capture » (flou, sous-exposition, bruit ; niveaux 0–4) qui appelle
   `drtl.robustness.degrade` d'Ikram avant le prétraitement.
2. **Prétraitement** : `load_best(best.pt)` puis `build_eval_transform(metadata["image_size"])`
   d'Abdelati, importés tels quels, jamais recopiés. Les noms de classes viennent de `metadata["class_names"]`.
3. **Comparaison** : 3 colonnes (feature extraction · partiel · complet) **sur la même image**, chacune avec
   les **scores des 5 grades** (barres triées dans l'ordre des grades, valeur en %) et la **carte
   d'attribution superposée**, légendée `Grad-CAM · <modèle> · <stratégie> · classe ciblée : <grade>`.
4. **États** :
   - checkpoint absent → la colonne affiche « Checkpoint not available: `<run_id>` (not trained yet) », sans score ;
   - attribution impossible → scores affichés, carte remplacée par « Attribution unavailable for this model » ;
   - upload invalide (pas une image, illisible, trop petit) → message clair, rien n'est prédit ;
   - aucun vrai checkpoint → mode exemple avec bandeau **SIMULATED** sur chaque score et chaque carte.
5. **Pied de page** permanent : « Educational research demo · not a medical device · no diagnosis or
   recommendation ». Pas de conseil d'orientation, pas d'affirmation sur des lésions ni de « confiance clinique ».

**Onglet « Compare models »** : tableau lu depuis `results/summary.csv` et `results/robustness.csv`
(Ikram), « missing » pour les configurations non entraînées.

**Mobile** (constat sur l'aperçu 390 px) : le cercle d'aperçu occupe tout le premier écran et repousse
le choix de la photo sous la ligne de flottaison ; dans Gradio, place l'upload en premier et limite la
hauteur de l'image d'entrée. Les 3 colonnes passent en pile.

Checkpoints locaux pour la démo : `checkpoints/<run_id>/best.pt` (dossier ignoré par Git), téléchargés
depuis l'Output Kaggle ; par défaut le fold retenu par Ikram pour l'évaluation finale.

## Commandes · Windows PowerShell

```powershell
git clone https://github.com/MOUATEAlaaeddine/dr-transfer-learning.git
cd dr-transfer-learning
git switch -c feat/retinalens-demo origin/main
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . gradio grad-cam
.\.venv\Scripts\python.exe -m pytest -q
# checkpoint SIMULATED pour développer l'app sans GPU :
.\.venv\Scripts\python.exe -m drtl.run --model mobilenetv4 --strategy feature_extraction --fold 0 --smoke --synthetic
# -> outputs\smoke\mobilenetv4__feature_extraction__fold0\best.pt
.\.venv\Scripts\python.exe app\app.py
```

## Sur Kaggle

Guide commun [START_HERE](START_HERE.md#entraîner-sur-kaggle-identique-pour-tous) avec
`MODEL = "mobilenetv4"`, `STRATEGY` ∈ `feature_extraction` (pilote), `partial_finetuning`, `full_finetuning`, `FOLD` 0 → 4.

## Terminé quand

- Pilote partagé et validé par Alaa ; 15 fiches MobileNetV4 `completed`
- `gradcam.py` + tests fusionnés ; figure par grade (une bonne prédiction + une erreur) dans le notebook 03
- La démo donne les mêmes prédictions que le notebook sur 10 images du test et répond en moins de 2 s sur CPU
- Capture bureau + mobile de la démo dans la PR

## PR : en plus de la checklist commune

- [ ] Prétraitement et degradations importés (aucune copie)
- [ ] Les 4 états (checkpoint absent, attribution indisponible, upload invalide, SIMULATED) visibles sur captures
- [ ] Aucun texte médical prescriptif

## Prompt à coller dans Claude

> Aide-moi à réaliser mon lot dans le dépôt `dr-transfer-learning`, sur la branche `feat/retinalens-demo`.
> Lis d'abord `docs/team/START_HERE.md`, `docs/team/INTERFACES.md` (§2, §5, §8, §9), `docs/team/youssef.md`,
> `src/drtl/train.py` (`load_best`) et le prototype `docs/prototypes/transfer_lab/index.html`, puis vérifie
> `git status` et ne perds aucun travail local. Ne modifie pas le pipeline partagé.
> 1) Implémente `src/drtl/gradcam.py` avec pytorch-grad-cam : Grad-CAM sur la dernière couche convolutive
> des CNN ; pour ViT-B/16, Grad-CAM sur `blocks[-1].norm1` avec reshape 14×14 (ou attention rollout),
> méthode nommée dans le libellé. Ajoute `tests/test_gradcam.py` (forme, plage [0,1], la carte change avec la classe ciblée).
> 2) Implémente `app/app.py` en Gradio selon la spécification de `youssef.md` : upload JPG/PNG, choix du backbone,
> 3 stratégies comparées sur la même image avec scores des 5 grades et carte d'attribution légendée,
> curseur de qualité via `drtl.robustness.degrade`, prétraitement via `build_eval_transform` et classes lues
> dans le checkpoint. Gère checkpoint absent, attribution indisponible, upload invalide, et un mode exemple
> étiqueté SIMULATED tant qu'il n'y a pas de vrais checkpoints. Pied de page « Educational research demo,
> not a medical device » ; aucune recommandation clinique ni affirmation sur des lésions.
> 3) Teste l'app avec un checkpoint SIMULATED généré par
> `python -m drtl.run --model mobilenetv4 --strategy feature_extraction --fold 0 --smoke --synthetic`.
> Si gstack est installé, utilise son navigateur/QA pour un aperçu bureau et mobile ; sinon Playwright.
> Donne-moi les commandes PowerShell exactes.
> 4) Rédige en anglais Introduction, Related Work, Interpretability et Perspectives (références réelles uniquement).
> Termine par les captures, les fichiers modifiés, les tests passés et une description de PR prête à relire.
