# Workflows (DAG) and API sizing

## 1. Workflow DAGs

### Data and training pipeline (DVC, `dvc.yaml`)

```mermaid
flowchart LR
    raw[("data/raw<br/>GBG500.parquet + labels<br/>(DVC, MinIO)")] --> prepare
    params[/params.yaml/] --> prepare
    params --> train
    prepare["prepare<br/>validate, clean, split by ride"] --> processed[("data/processed")]
    processed --> train["train<br/>1D-CNN, MLflow tracking"]
    train --> gate{"quality gate<br/>macro-F1 > baseline"}
    gate -- passed --> registry[("MLflow registry<br/>ride-classifier@champion")]
    gate -- failed --> stop([exit 1: no promotion])
    train --> metrics[/metrics.json/]
```

`dvc repro` only re-runs a stage when its inputs (data, code, parameters) changed.

### CI/CD (GitHub Actions)

```mermaid
flowchart TD
    pr([pull request / push]) --> lint
    pr --> test
    lint --> docker["docker<br/>build training image + smoke test"]
    test --> docker
    lint --> serving["serving<br/>build API + web app images + smoke test"]
    test --> serving
    docker -- main only --> hub[(Docker Hub<br/>tag = commit SHA)]
    serving -- main only --> hub
    hub --> deploy["deploy (self-hosted runner)<br/>kubectl apply, rollout, prediction check"]
    deploy -- rollout fails --> rollback([rollout undo])
    manual([manual / retraining trigger]) --> train["train (self-hosted runner)<br/>dvc pull, dvc repro, quality gate"]
    train --> registry[(MLflow registry)]
```

### Request path (serving)

```mermaid
flowchart LR
    user([user]) --> web["ride-webapp<br/>Streamlit"]
    web --> api["ride-api x2<br/>FastAPI"]
    api --> model[("ride-classifier@champion<br/>MLflow registry + MinIO")]
    api --> metrics["/metrics<br/>Prometheus"]
    api --> log[/predictions.jsonl/]
    log --> drift["drift detection<br/>(monitoring)"]
    web -- feedback --> api
```

## 2. API sizing

Measured with `make load` (Locust, rides of 3 000 samples ≈ 5 min of riding, 8-core Apple Silicon laptop, one
API process with the real model `ride-classifier@champion`, 40 s per level, 0 failed request in every run).

| Concurrent users | Requests | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|---|
| 1 | 232 | 5.9 req/s | 46 ms | 52 ms | 60 ms |
| 10 | 1 497 | 38 req/s | 120 ms | 240 ms | 290 ms |
| 20 | 1 650 | 42 req/s | 340 ms | 480 ms | 550 ms |

**Reading.** One process saturates around **40 req/s**: from 10 to 20 users the throughput no longer grows while the
latency triples, which is the signature of a CPU-bound service (one Python process, one core, the JSON parsing of
3 000 samples and the CNN inference). A single request costs about 25 ms of CPU.

**Sizing for Kubernetes.** The manifest `k8s/api.yaml` asks for 250 m CPU and 512 Mi per pod, with a limit of 1 CPU and
1 Gi. With one core per pod, 2 replicas serve about **80 req/s**, far above the expected traffic of this application
(a ride is classified once, after it ends). The two replicas are kept for **availability** (rolling update without
downtime, one pod can die), not for throughput. Scaling further is a matter of `replicas` (or a
HorizontalPodAutoscaler on CPU), since the pods are stateless.

**Limits of this measurement.** It was made on a laptop, outside Kubernetes, with synthetic rides (random noise): the
latency inside a pod limited to 1 CPU will be higher. It was not repeated on the cluster.

## 3. Check of the real model through the API

The real model trained by the pipeline (test macro-F1 0.2675) was served by the API and queried with 10 rides of the
GBG500 data (2 per class, all samples including the NaN sent as `null`): **7 of 10 correct**, latency 10 to 90 ms.
The mistakes are a *Reckless* ride predicted *Safe overall*, a *Safe overall* predicted *Safe* and a *Tandem*
predicted *Safe overall*, consistent with the confusion matrix in the README. This is a smoke check on rides that
include training data, not an evaluation: the evaluation is the test macro-F1.
