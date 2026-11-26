# Diabetic Retinopathy Grading with Deep Transfer Learning

Master ISI S3 · Deep Learning (Transfer Learning) · FSSM, Université Cadi Ayyad · 2025-2026 · Pr. Mustapha AATILA

We grade diabetic retinopathy (DR) into five ICDR grades (No DR, Mild, Moderate, Severe,
Proliferative) from colour fundus photographs, and measure **what transfer learning actually
changes**: the same pretrained backbone is trained three ways and compared with a CNN trained
from scratch.

| | |
|---|---|
| Dataset | [APTOS 2019, 224 px (Kaggle · sovitrath)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data): 3,662 images, CC0. The assignment's example dataset (jockeroika) was deleted from Kaggle; APTOS 2019 is the fallback planned in the report §11 |
| Backbones (ImageNet) | MobileNetV4-S, DenseNet-201, VGG19-BN, ViT-Base/16 |
| Baseline | Small CNN trained from scratch |
| Transfer strategies | Feature extraction · Partial fine-tuning · Full fine-tuning |
| Protocol | 15 % frozen test set + stratified 5-fold CV, seed 42, identical settings for all models |
| Metrics | Accuracy, precision, recall, F1, ROC/AUC, confusion matrix, parameters, training time |
| Interpretation | Grad-CAM: diseased regions and detected signs |

That is 4 backbones × 3 strategies + 1 baseline = **13 configurations**, each on 5 folds.

## Status

| Step | Owner | State |
|---|---|---|
| 1. Repository and environment | ALAA EDDINE | done |
| 2–3. Data, split, preprocessing (`scripts/prepare_data.py`) | ALAA EDDINE | APTOS data prepared and split frozen |
| 4–5. Strategies, training loop, notebook 02 | ALAA EDDINE | done, tested on SIMULATED data |
| 6. Grid of 65 runs (`scripts/run_grid.py`) | ALAA EDDINE | fold-0 runs in progress |
| 7. Evaluation, robustness (`scripts/final_evaluation.py`) | ALAA EDDINE | frozen test evaluation pending the completed grid |
| 8–9. Grad-CAM, Gradio demo (`app/app.py`) | ALAA EDDINE | code done; SIMULATED example mode until real checkpoints |
| 10–11. Paper, defence | team | to do |

Team handoff: [docs/team/START_HERE.md](docs/team/START_HERE.md).

Raw validation records are in [results/runs](results/runs). The initial five runs cover all
three MobileNetV4 strategies and two DenseNet-201 strategies on fold 0. Predictions and
metrics in `docs/prototypes/` are **SIMULATED**. Final test results follow Step 7.

## Setup

### Local (macOS / Linux, Python ≥ 3.9)

```bash
git clone https://github.com/MOUATEAlaaeddine/dr-transfer-learning.git
cd dr-transfer-learning
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                                            # makes `python -m drtl.run` work from anywhere
pytest -q                                                   # fast tests, no internet needed
python -m nbconvert --to notebook --execute notebooks/00_setup_check.ipynb --output-dir /tmp
```

On Apple Silicon use a native arm64 Python (`python3 -c "import platform; print(platform.machine())"`
must print `arm64`); an Intel Anaconda running under Rosetta cannot install recent PyTorch.

To continue the local float32 experiments, run `bash scripts/mac_train_all.sh --folds 0`
from the repository root. Follow `outputs/grid.log` and `outputs/grid_status.json`. With all
required ImageNet weights cached, set `HF_HUB_OFFLINE=1` to train without network access.
Omit `--folds 0` to queue all five folds once enough disk space is available for both
best-model and optimizer checkpoints.

The three-fold fallback uses folds 0, 1 and 2 from the original split (39 runs).
Use `bash scripts/mac_train_all.sh --folds 0 1 2` and then
`python scripts/final_evaluation.py --folds 0 1 2` in the project environment.
Each run still trains on the other four CV folds. The 511-image frozen test set
is evaluated once after the selected grid is complete.

### Google Colab or Kaggle

Open `notebooks/00_setup_check.ipynb` (File → Open notebook → GitHub, paste the repo URL) and run all
cells. The first cell clones the repository and installs only what the runtime is missing.
On Kaggle, enable *Internet* and a *GPU* accelerator in the notebook settings.

## Repository layout

```
configs/default.yaml   shared settings: models, strategies, training parameters
src/drtl/              all reusable code (notebooks and the demo import it)
  config.py            YAML loading + per-run overrides
  utils.py             seeding, device detection (CUDA / MPS / CPU)
  models.py            timm backbones + from-scratch CNN, freezing strategies, LR groups
  train.py             training loop: AdamW, warm-up + cosine, AMP, early stopping, resume
  run.py               one model × strategy × fold run (also `python -m drtl.run`)
  results.py           per-run records and the merged results/runs.csv
  fixtures.py          SIMULATED images for tests and smoke runs only
notebooks/             thin notebooks that call src/drtl (00 setup check, 02 train one run)
tests/                 pytest smoke tests
data/splits/           frozen split file (images themselves stay on Kaggle)
results/               per-run records (results/runs/), merged CSV, figures
paper/                 English paper (required structure, assignment p.1–4)
docs/                  planning report (French) and UI prototypes
```

## Team

| Member | Contribution |
|---|---|
| MOUATE ALAA EDDINE (lead) | Repository and environment; all code in `src/drtl`, `scripts/` and `app/` (data pipeline, models and transfer strategies, training, evaluation, robustness, Grad-CAM, demo); all training runs |
| Abdelati | *to be filled with what he actually contributes (paper sections, review, references, presentation)* |
| Ikram | *idem* |
| Youssef | *idem* |

The contributions table of the paper is filled from what each member actually did and from the Git history.

## Working rules

See [CONTRIBUTING.md](CONTRIBUTING.md): branches and pull requests, notebook and code standards,
how results and simulated outputs are labelled.

## Originality and sources

The assignment requires original work (p.4). Libraries, pretrained weights, datasets and papers
we build on are cited in this README and in the paper's references; borrowed code is credited
in a comment where it is used.
