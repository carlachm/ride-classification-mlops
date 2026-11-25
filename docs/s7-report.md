# S7 lab report: drift and retraining (ride classifier)

2026-11-25, local runs. Model served: `ride-classifier@champion` v1 (test macro-F1 0.2675). Each run sends 500
**simulated** rides (validation and test rides drawn with replacement), about half with a feedback. Attach the
reports `reports/{baseline,drift,concept,unit_bug}.{json,html}` (JSON: our tests; HTML: Evidently; not in Git).

**Reference.** The 350 training rides (mean and standard deviation of the 6 channels, and ride length: 13
features). The lab asks for a healthy production window and not the training split; we have no production period
yet, and our split is random by ride (the fraud lab's is chronological), so the training rides describe the same
population. **Rule:** a feature drifts if PSI > 0.25 (5 bins) **and** K-S p < 0.01/13; `drift` needs 2 features
and 100 rides. A plausibility check (units, dead sensor) comes first.

## 1. The runs

| Run | Verdict (exit) | Max PSI | PSI > 0.25 | K-S significant | Both | Predicted classes, chi² p | Feedback accuracy | Drift type |
|---|---|---|---|---|---|---|---|---|
| Baseline | ok (0) | 0.160 (`n_samples`) | 0 | 0 | 0 | 0.81 | 0.57 (244) | none |
| Data drift | drift (3) | 4.24 (`az_std`) | 6 | 7 | 6 | 1.9e-05 | 0.48 (247) | data drift |
| Concept drift | ok (0) | 0.122 (`n_samples`) | 0 | 1 | 0 | 0.95 | 0.25 (253) | concept drift |
| Unit bug | upstream bug (5) | 7.37 (`ax_std`) | 6 | 6 | 6 | 2.9e-54 | 0.77 (252) | not a drift: a bug |

- **K-S and PSI.** In the data drift run, `ax_mean` has K-S p = 0.0003 but PSI 0.065: the difference is
  detectable and small. `ay_std` has PSI 1.83 and p = 5e-65: PSI shows the size of the change, the p-value mostly
  how sure we are of it, which depends on the sample size.
- **Data drift:** the rides are rougher (vibration x1.4): P(X) changed.
- **Concept drift:** expected before the run: no input test fires, since the signals are unchanged. Confirmed.
  Only the link between signals and labels changed, and only labels reveal it. In Prometheus,
  `RecklessRideMissed` fired and `FeedbackAccuracyDrop` went from pending to firing after 5 minutes (accuracy
  0.25).

**Unit bug.** Accelerometer sent in g: the gravity norm is 0.97 m/s² instead of about 9.4, and the three
accelerometer channels vary 0.10 times as usual. These are the same rides in another unit, so we do **not**
retrain: the model would learn the bug. Fix at the sender (send m/s²), then check that the gravity norm is back
and that `make drift` answers `ok` on new rides. The accuracy on the feedback went **up** (0.77): watching
performance alone would have missed this bug.

**Sensitivity to the bins.** Our verdict / the lab rule (max PSI > 0.25) on the same logs:

| Run | 5 bins | 10 bins | 20 bins |
|---|---|---|---|
| Baseline | ok / ok (0.160) | ok / **retrain** (0.304) | ok / **retrain** (0.469) |
| Data drift | drift / retrain | drift / retrain | drift / retrain |
| Concept drift | ok / ok (0.122) | ok / ok (0.241) | alert / **retrain** (0.484) |
| Unit bug | bug / retrain | bug / retrain | bug / retrain |

Here the lab rule changes its answer with the bins on inputs that did not drift. See
[monitoring.md](monitoring.md) for the calibration of our rule.

## 2. Trigger

`scripts/retrain_on_drift.sh`, run with `DRY_RUN=1` (nothing started):

| Log | Decision (exit code) |
|---|---|
| Baseline | `NO_ACTION` (0): stays silent |
| Data drift | `BLOCKED_NO_NEW_DATA` (6) |
| Concept drift | `BLOCKED_NO_NEW_DATA` (6): signal from the accuracy |
| Unit bug | `INVESTIGATE` (5) |

Unlike the lab script, it also refuses to retrain without 20 new labelled rides (simulated rides never count).
With `--allow-no-new-data`, the data drift log gives `RETRAIN`, and without `DRY_RUN` it would run
`gh workflow run train.yml -f reason="input drift on ... (max PSI 4.24); 0 new labelled rides"`.

## 3. Retraining run in GitHub Actions: **not done**

`train.yml` has no run on GitHub. Missing: a self-hosted runner (added by the repository owner), the MinIO
secrets, the `MLFLOW_TRACKING_URI` variable, the data in that machine's MinIO, and for the approval the
`production` environment plus `PRODUCTION_APPROVAL` = `on` (see [retraining.md](retraining.md)). The job
`deploy-champion` (manual approval) is written but has never run. Once ready:
`bash scripts/retrain_on_drift.sh --allow-no-new-data`, then `gh run watch`.

## 4. What the retraining changed, and what it did not

In place of the missing Actions run, the training step of `train.yml` (`dvc repro --force`) was run **on the
laptop**, with `RETRAIN_REASON` taken from the drift verdict. It is not the GitHub Actions run the lab asks for.

| | Champion v1 | New version v3 |
|---|---|---|
| Test macro-F1 | 0.2675 | 0.2675 (identical) |
| Quality gate | passed | passed |
| Promotion | champion | rejected: does not beat the champion |
| Reason recorded (tag `retrain_reason`) | none | input drift on the six `_std` features, max PSI 4.24 |

- **Changed:** one more version in the registry, with the reason of its run.
- **Not changed:** the test score, the champion and the model the API serves.

- **What did it learn from the drifted traffic? Nothing.** `train.py` reads `data/processed/train.parquet`, built
  by `prepare` from `data/raw/GBG500.parquet` and its labels, plus `data/feedback/` when it exists. The drifted
  rides are only in `logs/`, and they are simulated.
- **Which data should enter training?** Real rides with a reviewed feedback: the feedback names the ride
  (`ride_hash`), the API keeps its signal (`RIDE_STORE_DIR`), `make feedback-data` collects it, then `dvc add`
  and `dvc push`, with the pointer merged on `main`. The self-hosted runner reads it with `dvc pull`. These rides
  join the train split only: validation and test stay frozen.

## Known limits

- Same data, same model: the pipeline is exercised, not the adaptation.
- A verdict says drift, not cause. The plausibility check catches unit-like bugs and dead sensors, not every bug.
- The reference is a choice (above) and the thresholds are rules of thumb, calibrated by redrawing only 150 rides.
- All traffic is simulated.
- `@champion` moves before the approval: a pod restarted for another reason would load the new model.
- The optional parts of the lab (7, 9, 10) were not done.
