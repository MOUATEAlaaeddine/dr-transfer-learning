# Team standards

These rules keep the four of us able to read, run and present each other's work.

## Notebooks

- **Short explanations tied to our experiments.** Before each step, say in two or three sentences
  what we test and why, in terms of our data and models, not textbook theory.
- **Runs top to bottom in a fresh Colab runtime.** No hidden state, no local paths, no manual
  uploads: the first cell clones the repo and installs what is missing. Check with
  *Runtime → Restart and run all* before opening a pull request.
- **Every teammate can present it.** Each notebook explains, where it uses them, our data split
  (15 % frozen test set + stratified 5-fold CV), why each model was chosen, and what each metric
  tells us.
- Thin notebooks: logic lives in `src/drtl/`, notebooks call it and show results.
- Clear outputs before committing, except final results notebooks.

## Code

- Clear names and simple functions; one function, one job.
- Comment where a **decision** needs explaining (why this learning rate, why this layer is
  frozen), not what the line does.
- Preprocessing is defined once in `src/drtl` and imported by both training and the demo app,
  so the app sees images exactly as the model was trained on them.
- Shared settings live in `configs/default.yaml`; a run may override values but must record them.
- `pytest -q` passes before every merge.

## Repository

- `main` is protected: one branch per task (`feat/data-folds`, `feat/gradcam`, …), merged by pull
  request after review by another member.
- Readable commits: `feat:`, `fix:`, `docs:`, `exp:`, `chore:`. Commit regularly; the history is
  graded (assignment p.4).
- Never commit images or model weights. Data stays on Kaggle; weights go to Kaggle outputs or a
  private team Kaggle dataset.
- Each training run appends one line to `results/runs.csv` (model, strategy, fold, metrics,
  duration, commit).
- The README lists real setup steps, team roles and **only verified results**. Any mock prediction
  or metric (prototypes, placeholders) is labelled **SIMULATED** wherever it appears.
- Original work (assignment p.4): cite the libraries, weights, datasets and papers we use, and
  credit borrowed code in a comment where it is used.
