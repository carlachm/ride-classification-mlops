# Monitoring

Four layers, from the metrics of the API (`/metrics/`, see [api-contract.md](api-contract.md)) and from its
prediction log. What to do when an alert fires: [runbooks.md](runbooks.md).

```
ride-api (:8000) ──/metrics/──▶ Prometheus (:9090, alert rules) ──▶ Grafana (:3000, dashboard)
        │
        ├─ logs/predictions.jsonl ──▶ drift check (PSI, K-S, chi-squared, plausibility)
        └─ logs/rides/*.parquet   ──▶ retraining (retraining.md)
```

## Run it

```bash
make up                 # MLflow stack + Prometheus + Grafana
make repro              # a champion model in the registry
make drift-reference    # healthy reference for the drift check (data/monitoring/reference.csv)
make api                # terminal 2: logs/predictions.jsonl and raw rides in logs/rides
make simulate           # terminal 3: 200 normal rides, half of them with feedback
make drift              # drift report on the last 200 predictions -> reports/drift.json
```

Grafana: http://localhost:3000, dashboard *Ride classifier: 4-layer monitoring* (folder MLOps).
Alerts: http://localhost:9090/alerts.

| Scenario | Command | What should react |
|---|---|---|
| Normal traffic | `make simulate` | nothing |
| Data drift (rougher rides, vibration x1.4) | `make simulate-drift` | `InputShift`, `make drift` exits 3 |
| Concept drift (70 % of Safe rides now judged Reckless) | `make simulate-concept-drift` | feedback only: `RecklessRideMissed`, `FeedbackAccuracyDrop` |
| Upstream bug (accelerometer sent in g) | `make simulate-unit-bug` | `InputUnitSuspected`, `make drift` exits 5 |

The simulator replays validation and test rides (never used for training), with `sim-` ride ids.
Prometheus alerts look at 5 or 15 minutes of traffic: after a scenario, older traffic stays in the window and
dilutes the signal. `make drift LAST=200` looks at the last 200 rides only.

## Alerts

| Layer | Alert | Fires when | Normal value measured |
|---|---|---|---|
| Infrastructure | `ApiDown` | no answer for 1 min | |
| | `ModelNotLoaded` | API up without a model for 1 min | |
| | `LatencyP95High` | P95 of `/v1/predict` > 500 ms for 1 min | about 0.2 s, one client sending one ride at a time |
| | `ErrorRateHigh` | more than 5 % of 5xx for 1 min | 0 |
| Inputs | `RejectedRidesHigh` | more than 10 % of rides answered 422 | 0 |
| | `InputUnitSuspected` | a channel varies 4x more or less than in training (≥ 5 rides) | ratio 0.98 to 1.04 |
| | `InputShift` | ratio > 1.25 or < 0.8, below the unit-bug level (≥ 30 rides, 2 min) | same |
| | `MissingValuesHigh` | more than 5 % of values missing | 0.7 % |
| Predictions | `SafeShareShift` | share of rides predicted Safe < 30 % or > 75 % (≥ 50 rides) | 52 % on held-out rides |
| Outcome | `FeedbackAccuracyDrop` | accuracy on feedback < 35 % over 15 min (≥ 50 labels) | about 0.6 on simulated traffic, 0.52 on the test set |
| | `RecklessRideMissed` | a ride confirmed Reckless was predicted otherwise | see below |

How the thresholds were chosen (GBG500, champion with test macro-F1 0.27):

- **InputShift**: bootstrap on the 150 held-out rides. With 30 rides or more, the mean per-ride standard
  deviation of a healthy batch stays within about 0.85 to 1.25 of the training value.
- **SafeShareShift**: same bootstrap on the champion's predictions. With 50 rides, 99.8 % of healthy batches
  have between 30 % and 74 % of Safe.
- **FeedbackAccuracyDrop**: 0.35 is about 2.5 standard errors below 0.52 with 50 labels.
- **RecklessRideMissed** will be frequent: the champion's recall on Reckless is 0 on the single Reckless test
  ride. It is a ticket, not a page.

The rules have unit tests (`make alerts-test`). The reference values of `InputShift` are written in
`infra/prometheus/alerts.yml`: update them if the training data changes.

## Drift check

`make drift` compares, per ride, the mean and standard deviation of the 6 channels and the ride length
(13 features) with the **training rides**. The train / validation / test split is random and stratified by
ride, so the training rides describe the same population as healthy traffic. The predicted classes are
compared (chi-squared) with the champion's answers on validation + test rides; this is reported but never
triggers.

Verdicts and exit codes: `ok` / `alert` (0), `drift` (3), `insufficient_data` (4, fewer than 100 rides),
`upstream_bug_suspected` (5). Chain actions on 3 only: a crash (1) must never start a training run.

### Why not "one feature with PSI > 0.25"

Measured on 200 random batches drawn (with replacement) from the 150 held-out rides, which have **no**
drift. Share of batches declared drifted:

| Rides per batch | Bins | 1 feature PSI > 0.25 | 2 features PSI > 0.25 | 1 feature, PSI + K-S | 2 features, PSI + K-S |
|---|---|---|---|---|---|
| 100 | 10 | 99 % | 83 % | 10 % | 0 % |
| 100 | 5 | 27 % | 6 % | 6 % | **0 %** |
| 200 | 10 | 77 % | 14 % | 4 % | 0 % |
| 200 | 5 | 4 % | 0 % | 1 % | 0 % |

K-S is judged at 0.01 / 13 (one test per feature). With the rule kept (5 bins, PSI + K-S, 2 features, at
least 100 rides), a vibration increase of x1.3 on all channels was detected in 100 % of the batches. The
price: a shift of a **single** feature only gives `alert`. These batches reuse the same 150 rides, so real
false alarm rates may differ.

### Upstream bugs come first

Before any verdict, two plausibility checks: the norm of the mean acceleration must be close to gravity
(GBG500: about 9.4 m/s²; accepted 7 to 12), and no channel may vary 4x more or less than usual. An
accelerometer sent in g gives 0.97 m/s². Such a bug is fixed at the sender: retraining would teach the model
the bug. Note that the accuracy on feedback does **not** necessarily drop in that case.

## Limits

- Input tests cannot see **concept drift**: only the feedback can, after the label delay, and only on the
  rides that get a label.
- The checks look at one feature at a time: a change in the relation between channels is invisible.
- Prometheus scrapes the API on the host (:8000). On **Kubernetes** it is not monitored: a port-forward
  reaches a single pod, and the pod logs live in an `emptyDir` lost at each restart. It would need
  Prometheus in the cluster (the pods already carry the `prometheus.io/*` annotations) and a shared volume
  or log collector.
- No alert is sent anywhere (no Alertmanager): they are visible in the Prometheus UI.
