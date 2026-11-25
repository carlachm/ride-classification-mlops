# Retraining

Retraining only fixes something when **new data reaches the training set**: with the same data, code and
parameters, the pipeline gives the same test score and the new version is not promoted.

```
feedback + raw rides ──▶ make feedback-data ──▶ data/feedback ──▶ dvc add + push ──▶ PR on main
                                                                                        │
make retrain-check ── drift OR accuracy drop, and new labelled rides? ──▶ make retrain ─┤
                                                                                        ▼
                         train.yml (self-hosted): dvc pull ─▶ dvc repro ─▶ quality gate ─▶ @champion
                                                                                        │
                                        manual step: kubectl -n ride rollout restart deploy/ride-api
```

## 1. Collect the labelled production rides

The API keeps the raw signal of each ride when `RIDE_STORE_DIR` is set (`make api` uses `logs/rides`), and
the feedback names its ride with `ride_hash` (the web app does it).

```bash
make prepare          # the validation / test rides are needed by the leak guard
make feedback-data    # -> data/feedback/rides.parquet + labels.csv, with the reason of every rejection
```

Rejected rides: test traffic (`ride_id` starting with `sim-`, `demo`, `load-`, `smoke`, `deploy-check`),
rides already in the validation or test sets (**leak**), GBG500 training rides, feedback without
`ride_hash`, signal not stored. The dataset accumulates and the latest label of a ride wins.

**Look at the labels before using them**: nothing checks that a reviewer is right.

## 2. Version it

The CI runner only sees Git and the DVC remote.

```bash
uv run dvc add data/feedback && uv run dvc push
git switch -c data/feedback-<date>
git add data/feedback.dvc && git commit -m "data: add labelled production rides"
# then a pull request to main (main is protected)
```

`prepare` adds these rides to the **train split only** and drops again any copy of a validation or test
ride. Validation and test stay **frozen**, so a new model is compared with the champion on the same rides.
`dvc.yaml` does not list this optional folder as a dependency: use `dvc repro --force` locally
(`train.yml` already does).

## 3. Decide

```bash
make retrain-check      # decision only
make retrain            # same, then `gh workflow run train.yml` if the answer is RETRAIN
```

In order:

| Check | If it fails |
|---|---|
| No sign of an upstream bug | `INVESTIGATE` (5): never retrain on a bug |
| A signal: input drift (`make drift`) **or** accuracy < 0.35 on at least 50 feedbacks | `NO_ACTION` (0), or `WAIT` (4) if there is too little data |
| At least 20 labelled rides that the last training run has not seen | `BLOCKED_NO_NEW_DATA` (6) |

Then `RETRAIN` (3). With `--execute`, the command refuses to start the workflow while `data/feedback.dvc`
is not identical to `origin/main`. It does not check that `dvc push` was done: if not, the run fails at
`dvc pull`. Thresholds: `retraining` in `params.yaml`.

This is a **hybrid** trigger (drift + performance): drift is early but blind to concept drift, performance
sees it but only after the labels arrive. Nothing is scheduled: the checks are run by hand.

To show the pipeline without new data (it proves the plumbing, not an adaptation):

```bash
uv run python -m fraud_detection.retraining.trigger --last 200 --allow-no-new-data --execute
```

## 4. Train, gate, deploy

`train.yml` writes the reason in the run summary; the training run records it (tag `retrain_reason`) with
the number of production rides used (`n_train_feedback_rides`). The new version takes `@champion` only if it
passes the quality gate and beats the champion's test macro-F1. The running API keeps the old model until it
is restarted: that restart is the **manual approval**.

### One-time setup of the self-hosted runner

- a self-hosted runner registered on the repository and running, on the machine of the MLflow stack;
- repository secrets `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY`;
- repository variable `MLFLOW_TRACKING_URI` = `http://localhost:5050` on macOS (the workflow default is
  port 5000);
- the data pushed to that machine's MinIO (`make push`, `dvc push`).

## Limits

- **Concept drift against a frozen test set**: if the meaning of the signals really changed, the old test
  labels are partly outdated, and the gate may reject the model that adapted. The test set would then need to
  be rebuilt, as a reviewed decision.
- With 1 to 6 test rides per minority class, a few rides move the macro-F1 a lot: beating the champion is
  weak evidence. Look at the confusion matrix before restarting the API.
- Labels only come back for the rides someone reviews: they are not a random sample.
- The raw signals are kept without any retention policy.
