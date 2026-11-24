# Shared interfaces (agreed contracts)

These signatures let the four of us code in parallel. The shared pipeline (`run.py`, `train.py`)
already calls them; `tests/test_integration_boundary.py` checks the contract with a SIMULATED
stand-in until the real modules are merged. Change a signature only through a PR reviewed by the
person who depends on it.

All modules are implemented by MOUATE ALAA EDDINE.

## 1. Split table · `data/splits/folds.csv`

One row per image, frozen once, committed, never regenerated after the first real run.

| Column | Type | Meaning |
|---|---|---|
| `image_id` | str | unique id (file stem) |
| `path` | str | path relative to the dataset root, forward slashes |
| `label` | int 0–4 | ICDR grade, same order as `class_names` in `configs/default.yaml` |
| `class_name` | str | dataset folder name |
| `group_id` | str | patient id if the dataset provides a verifiable one, else `image_id` (documented limitation) |
| `split` | `cv` \| `test` | `test` = frozen 15 % stratified hold-out |
| `fold` | int | 0–4 for `cv` rows, −1 for `test` rows |

`splits_hash` = first 12 hex of SHA-256 of this file with CRLF → LF (`drtl.results.file_hash`), so
Windows and Mac clones give the same hash. Commit the file with LF endings (`.gitattributes`).

## 2. Preprocessing · `drtl/transforms.py`

```python
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_train_transform(image_size: int) -> Callable[[PIL.Image.Image], torch.Tensor]
def build_eval_transform(image_size: int) -> Callable[[PIL.Image.Image], torch.Tensor]
```
Both return a float tensor `[3, image_size, image_size]`, ImageNet-normalised, after the retina
crop. Augmentation only in `build_train_transform` (report §4.4). `build_eval_transform` is the
function the demo imports as is. `run.py` stores the same mean/std in each checkpoint; when
`transforms.py` lands, replace the two constants in `run.py` by an import from it.

## 3. Loaders · `drtl/data.py`

```python
def load_splits(path: Path) -> pd.DataFrame                    # columns of §1
def make_fold_loaders(config, splits, fold, image_root, batch_size) -> (train_loader, val_loader)
def make_test_loader(config, splits, image_root, batch_size) -> DataLoader   # final evaluation only
```
* Batches are `(images: float[B,3,H,W], labels: long[B])`.
* Train = `cv` rows with `fold != k`, shuffled with a generator seeded by `config["seed"]`;
  val = `cv` rows with `fold == k`, no shuffle, eval transform. `run.py` refuses a val loader whose
  length differs from the fold's row count (leakage guard).
* Each dataset exposes `.labels` (numpy int array).
* `num_workers = config["training"]["num_workers"]`; must also work with 0 on Windows.
* `image_root` is the folder Kaggle mounts (`run.find_kaggle_dataset` finds it).

## 4. Run · `drtl.run.run_experiment`

```python
run_experiment(model, strategy, fold, *, overrides=None, smoke=False, synthetic=False,
               resume=True, force=False, pretrained=None, image_root=None,
               output_root=None, results_dir=None, splits_file=None, device=None) -> dict
```
`model` ∈ `mobilenetv4, densenet201, vgg19, vit_b16, scratch_cnn`; `strategy` ∈
`feature_extraction, partial_finetuning, full_finetuning` (or `scratch` for `scratch_cnn`).
Same thing from a shell: `python -m drtl.run --model vit_b16 --strategy feature_extraction --fold 0`.

## 5. Checkpoints

`<outputs>/<run_id>/` with `<outputs>` = `/kaggle/working/outputs` on Kaggle, `outputs/` locally.

| File | Content |
|---|---|
| `best.pt` | `model_state`, `metadata`, `epoch`, `val_metrics` at the best validation macro F1 |
| `last.pt` | full resumable state (optimizer, scheduler, AMP scaler, RNG, early-stopping counters) |
| `history.csv` | one row per epoch: losses, val metrics, lr, seconds |
| `best_val_predictions.npz` | `probs [N,5]`, `labels [N]` of the best epoch |

`metadata` keys: `run_id, model, timm_name, strategy, fold, class_names, image_size,
normalization{mean,std}, seed, config_hash, splits_hash, git_commit, overrides, data_source, pretrained`.
Reload with `drtl.train.load_best(path) -> {"model", "metadata", "val_metrics"}` (model in eval mode).

## 6. Run record · `results/runs/<run_id>.json`

`run_id = <model>__<strategy>__fold<k>`, e.g. `vit_b16__feature_extraction__fold0`.
Fields (`drtl.results.RUN_COLUMNS`): `run_id, model, timm_name, strategy, fold, status, data_source,
seed, config_hash, splits_hash, git_commit, overrides, device, gpu_name, amp, batch_size,
accumulation_steps, params_total, params_trainable, epochs_run, best_epoch, stopped_early,
duration_s, sec_per_epoch, val_loss, val_accuracy, val_precision_macro, val_recall_macro,
val_f1_macro, checkpoint_ref, predictions_file, started_at, schema_version` (+ `class_names`).

* `status`: `completed` (real), `smoke`, `synthetic`. Only `completed` runs on the real dataset are merged (never `synthetic-SIMULATED`).
* Next to it: `<run_id>_val_predictions.csv` with columns `label, pred, prob_0 … prob_4`
  (validation fold, best epoch). Ikram recomputes every metric (AUC, QWK, confusion) from it.
* `python -m drtl.results` regenerates `results/runs.csv`: the 65 expected runs, `missing` where no
  record exists. Never edit `runs.csv` by hand; never fill a missing run.

## 7. Evaluation · `drtl/evaluate.py`

```python
def compute_metrics(y_true, y_prob, num_classes=5) -> dict
# keys: accuracy, precision_macro, recall_macro, f1_macro, auc_macro_ovr, qwk,
#       confusion_matrix (list of lists), per_class (dict), absent_classes (list)
```
A class absent from `y_true`: its AUC is undefined → excluded from the macro AUC and listed in
`absent_classes` (never silently 0.5). F1/precision/recall keep all 5 labels (`labels=range(5)`),
as in `drtl.train.basic_metrics`.

## 8. Degradations · `drtl/robustness.py`

```python
def degrade(image: PIL.Image.Image, kind: str, level: int) -> PIL.Image.Image
# kind ∈ {"blur", "underexposure", "noise", "jpeg"}, level ∈ {0,1,2,3,4}; level 0 = unchanged
```
Applied **before** `build_eval_transform`, identically in the robustness table and the demo slider.

## 9. Attribution · `drtl/gradcam.py`

```python
def attribution(model, image_tensor, target_class: int, model_name: str) -> tuple[np.ndarray, str]
# returns (heatmap [H,W] in [0,1], method label e.g. "Grad-CAM · features.norm5")
```
CNNs: Grad-CAM on the last convolutional block. ViT-B/16: Grad-CAM on `blocks[-1].norm1` with
the 196 patch tokens reshaped to 14 × 14 (or attention rollout), and the method named in the label.
