# RL Platform for Delayed-Reward Graph Navigation

A reinforcement learning codebase built around a single research idea: learning good policies in environments where rewards are sparse and arrive long after the actions that earned them. The core method combines Node2Vec graph embeddings with a reward-prediction network (InferNet) and tabular Q-learning. Around that core, the repository adds the pieces you normally end up writing when a research prototype grows: a configurable environment, several baseline agents, training and evaluation infrastructure, experiment tracking, telemetry, hyperparameter search, model serving, and container/Kubernetes deployment manifests.

The test domain is an N×N grid represented as a graph. A handful of nodes contain coins (reward), and the agent has to navigate to them. With only a few coins on a 64-node grid, most transitions return zero reward, which is what makes credit assignment hard and what the Node2Vec + InferNet approach is meant to address.

## Overview

The central agent works in three stages per training iteration:

1. **Structure learning** — Node2Vec learns dense node embeddings from the graph using a contrastive objective over co-occurring nodes in sampled trajectories. This captures topology independently of reward.
2. **Reward prediction** — InferNet maps those embeddings to predicted rewards using a dual loss (step-level pointwise MSE plus episode-level cumulative MSE), giving a denser learning signal than the raw sparse rewards.
3. **Value propagation** — Standard Q-learning updates a Q-table over the graph's states, and a greedy policy is extracted from it.

The rest of the repository is supporting infrastructure. There are baseline agents (DQN and its variants, PPO) for comparison, a vectorized environment, a distributed trainer, replay buffers, curriculum scheduling, an evaluation/benchmark suite, reward-shaping strategies, experiment tracking, a Prometheus-style telemetry server, a model registry with an HTTP inference service, and Docker/Kubernetes deployment files.

## Features

- Node2Vec + InferNet + Q-learning agent for delayed-reward graph navigation, with epsilon decay, learning-rate scheduling, and full checkpoint save/load.
- Baseline value-based agents: DQN, Double DQN, and Dueling DQN, each with a target network, Huber loss, gradient clipping, and experience replay.
- PPO agent with a clipped surrogate objective, Generalized Advantage Estimation, entropy bonus, and multiple update epochs per rollout.
- Configurable graph navigation environment with adjustable grid size, coin placement, reward values, step penalty, and episode length.
- Uniform and prioritized experience replay (the latter backed by a sum-tree with importance-sampling weights).
- Distributed training via a multiprocessing parameter-server: rollout workers collect trajectories into a shared queue while a central learner updates the models.
- Curriculum learning that advances through difficulty stages either on a fixed schedule or gated on success rate.
- Checkpoint management with atomic writes, versioned directories, best-model tracking, and automatic pruning of old checkpoints.
- Policy evaluation with bootstrap confidence intervals, random-baseline comparison, Mann-Whitney significance tests, and Monte Carlo value-landscape estimation.
- A benchmark suite that sweeps grid sizes, coin densities, and seeds, then aggregates statistics.
- File-based experiment tracking (JSON) with an optional MLflow backend.
- A telemetry service that exposes rolling metrics and system stats over an HTTP endpoint in Prometheus text format.
- Hyperparameter search via grid, random, or Optuna TPE.
- Reward-shaping strategies: potential-based, count-based exploration bonus, BFS temporal-distance shaping, InferNet densification, and a composite combiner.
- A model registry with staged deployment (development/staging/production) and a lightweight HTTP inference service.
- Docker, Docker Compose, and Kubernetes deployment manifests, plus a Prometheus scrape config.

## Tech Stack

- **Language:** Python 3.11
- **Deep learning:** PyTorch (`torch`, `torch.nn`, `torch.optim`, `torch.multiprocessing`)
- **Numerics / stats:** NumPy, SciPy (`scipy.stats` for bootstrap and Mann-Whitney tests)
- **Experiment tracking:** custom file-based tracker, optional MLflow backend
- **Hyperparameter optimization:** Optuna (optional; falls back to random search if not installed)
- **System metrics:** psutil
- **HTTP services:** Python standard library `http.server` (telemetry and inference)
- **Deployment:** Docker (multi-stage), Docker Compose, Kubernetes, Prometheus, Grafana

A note on dependencies: `requirements.txt` also lists `gymnasium`, `ray`, `fastapi`, `uvicorn`, and `hydra-core`. These are not currently used by the code — serving runs on the standard-library HTTP server, distributed training uses `torch.multiprocessing` rather than Ray, and configuration is handled through command-line arguments rather than Hydra. They can be removed if you want a leaner install.

## Project Structure

```
rl-platform/
├── src/
│   ├── agents/
│   │   ├── node2vec_rl.py      # Core agent: Node2Vec + InferNet + Q-learning
│   │   ├── dqn.py              # Deep Q-Network
│   │   ├── double_dqn.py       # Double DQN
│   │   ├── dueling_dqn.py      # Dueling DQN
│   │   └── ppo.py              # PPO with GAE
│   ├── models/
│   │   ├── node2vec.py         # Embedding table + contrastive loss
│   │   ├── infer_net.py        # Reward predictor (dual loss)
│   │   ├── actor_critic.py     # Shared-encoder actor-critic (PPO)
│   │   └── q_network.py        # Standard + dueling Q-networks
│   ├── envs/
│   │   ├── graph_nav_env.py    # Grid graph navigation environment
│   │   └── vectorized_env.py   # Synchronous multi-env wrapper
│   ├── training/
│   │   ├── trainer.py          # Main training loop
│   │   ├── distributed_trainer.py  # Parameter-server multiprocessing trainer
│   │   ├── replay_buffer.py    # Uniform + prioritized replay
│   │   ├── checkpoint_manager.py   # Versioned checkpoints
│   │   └── curriculum_learning.py  # Difficulty scheduling
│   ├── evaluation/
│   │   ├── evaluator.py        # Statistical policy evaluation
│   │   └── benchmark.py        # Multi-config benchmark suite
│   ├── utils/
│   │   ├── experiment_tracker.py   # JSON + optional MLflow tracking
│   │   ├── telemetry.py        # Prometheus-format HTTP metrics
│   │   ├── reward_shaping.py   # Auxiliary reward strategies
│   │   └── hyperparameter_search.py  # Grid/random/Optuna search
│   └── serving/
│       └── model_registry.py   # Model registry + HTTP inference service
├── notebooks/
│   ├── 01_original_implementation.ipynb
│   ├── 02_delayed_reward_optimization.ipynb
│   ├── 03_distributed_rl_training.ipynb
│   └── 04_benchmark_analysis.ipynb
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile          # Multi-stage build
│   │   └── docker-compose.yml  # Trainer + inference + monitoring stack
│   ├── k8s/
│   │   └── manifests.yaml      # Namespace, deployments, services, HPA
│   └── monitoring/
│       └── prometheus.yml      # Scrape configuration
├── scripts/
│   └── train.py            # CLI training entry point
├── requirements.txt
└── README.md
```

## Core Algorithm: Node2Vec + InferNet

The agent lives in `src/agents/node2vec_rl.py` and uses two models from `src/models/`.

**Node2Vec** (`node2vec.py`) is an embedding table of shape `(num_nodes, embed_dim)` trained with a contrastive objective. For each node in a sampled walk, nearby nodes (within a context window) are treated as positive pairs and randomly sampled nodes as negatives. The loss maximizes similarity for positive pairs and minimizes it for negatives using log-sigmoid of the dot product. The result is that structurally similar nodes get similar embeddings.

**InferNet** (`infer_net.py`) is a two-layer MLP (Linear → Tanh → Linear) that maps a node embedding to a scalar reward prediction. Its loss has two parts: a pointwise term that matches predicted to actual reward at each step, and a cumulative term that matches the predicted episode total to the actual total. The two are combined with a configurable auxiliary weight. The motivation is to give the agent a dense, learned reward signal at every step instead of only at the rare coin nodes.

**Q-learning** ties it together. Each iteration the agent samples one episode per starting node under the current policy, trains Node2Vec on the walk data, trains InferNet on the embeddings and observed rewards, applies tabular Bellman updates to the Q-table, and extracts a greedy policy. Exploration uses an epsilon value that decays over iterations, and both optimizers use `ReduceLROnPlateau` scheduling.

The agent exposes `train()`, `get_embeddings()`, `save_checkpoint()`, and a `load_checkpoint()` classmethod. Configuration is a dataclass (`AgentConfig`) covering embedding dimension, walk/context parameters, learning rates, discount factor, and the epsilon schedule.

## Implemented Algorithms

| Algorithm | File | Notes |
|---|---|---|
| Node2Vec + InferNet + Q-learning | `agents/node2vec_rl.py` | Core method for graph-structured, delayed-reward tasks |
| DQN | `agents/dqn.py` | Target network, replay buffer, Huber loss, gradient clipping |
| Double DQN | `agents/double_dqn.py` | Online net selects actions, target net evaluates them |
| Dueling DQN | `agents/dueling_dqn.py` | Separate value and advantage streams |
| PPO | `agents/ppo.py` | Clipped surrogate, GAE, entropy bonus, multi-epoch updates |

The value-based agents operate on node embeddings (they index into an embedding matrix passed to `update()`), so they're designed to consume the Node2Vec representations rather than raw state indices. PPO uses the shared-encoder actor-critic network in `models/actor_critic.py`.

## Environment

`GraphNavEnv` (`src/envs/graph_nav_env.py`) models an N×N grid as a graph. State is the current node index, actions are the four cardinal directions, and a move into a coin node returns `coin_reward` (default 1.0) while everything else returns zero (minus an optional step penalty). Invalid moves at the boundary leave the agent in place. Episodes end after `max_steps`.

Key methods:

- `reset(start_node=None)` — start a new episode (random or fixed start).
- `step(action)` — returns a `StepResult` with next state, reward, done flag, and an info dict.
- `sample_episode(policy, start_node)` — roll out a full episode under a `{state: action}` policy.
- `render_policy(policy)` — ASCII visualization of a policy with directional arrows and coin markers.
- `get_edge_index()` — edge tensor in the format used by graph libraries.

`VectorizedEnv` (`vectorized_env.py`) wraps several environments behind one interface for batch rollouts and aggregate statistics. It currently runs the environments sequentially rather than in separate processes.

## Training Infrastructure

**Trainer** (`training/trainer.py`) drives the main loop: it sets seeds, runs the agent's collect/update/evaluate cycle, performs periodic evaluation, checks early stopping, schedules checkpoints through the `CheckpointManager`, logs metrics, and writes a metrics JSON at the end. Configuration is via `TrainerConfig` (experiment name, iteration count, eval/checkpoint/log intervals, early-stopping patience, output paths, seed).

**DistributedTrainer** (`training/distributed_trainer.py`) implements a parameter-server pattern with `torch.multiprocessing`. Worker processes read a shared policy from an `mp.Manager().dict()`, sample episodes, and push trajectories onto an `mp.Queue`. The central learner drains the queue, updates Node2Vec and InferNet, and periodically recomputes and broadcasts a greedy policy back to the workers. It reports per-iteration loss and a rough throughput estimate.

**Replay buffers** (`training/replay_buffer.py`) include a uniform `ReplayBuffer` (a `deque` with random sampling) and a `PrioritizedReplayBuffer` built on a `SumTree` for O(log N) proportional sampling, with annealed importance-sampling weights and TD-error-based priority updates.

**CheckpointManager** (`training/checkpoint_manager.py`) writes checkpoints to timestamped run directories using a temp-file-and-rename pattern for atomicity, tracks the best model by evaluation return, stores metadata as JSON, and prunes old checkpoints beyond a configurable limit.

**CurriculumScheduler** (`training/curriculum_learning.py`) defines ordered difficulty stages (grid size, coin density, success threshold) and advances either on a linear schedule or when the rolling success rate clears a threshold. It can build an environment for the current stage on demand.

## Evaluation and Benchmarking

**PolicyEvaluator** (`evaluation/evaluator.py`) runs Monte Carlo rollouts and reports mean/std/min/max/median return, a bootstrap confidence interval, coin discovery rate, and mean episode length. It can evaluate a random baseline, compare several named policies with Mann-Whitney U tests, and estimate a per-state value landscape via discounted rollouts.

**BenchmarkSuite** (`evaluation/benchmark.py`) trains and evaluates the core agent across combinations of grid size, coin density, and seed, then aggregates mean/std return and improvement over random per configuration. Results are written to JSON and can be printed as a ranked leaderboard.

## Experiment Tracking, Telemetry, Search, and Reward Shaping

**ExperimentTracker** (`utils/experiment_tracker.py`) stores runs as JSON (parameters, metrics with steps and timestamps, tags, artifacts) and can list, load, compare, and find the best run by a chosen metric. If `use_mlflow=True` and MLflow is installed, it mirrors params and metrics to an MLflow server; otherwise it falls back to local-only tracking.

**TelemetryService** (`utils/telemetry.py`) keeps rolling windows of metrics and counters, collects CPU/memory (and GPU memory if CUDA is available) via psutil, and serves them on a background HTTP thread. Endpoints: `/metrics` (Prometheus text format), `/summary` (JSON), and `/health`.

**HyperparameterSearch** (`utils/hyperparameter_search.py`) takes an objective function and a parameter space and supports exhaustive grid search, random search, and Optuna TPE (with a graceful fall back to random search if Optuna isn't installed). Results are recorded per trial and can be saved or printed as a ranked summary.

**Reward shaping** (`utils/reward_shaping.py`) provides several strategies behind a common interface: potential-based shaping (policy-invariant), a count-based exploration bonus, BFS temporal-distance shaping toward goal nodes, InferNet-based densification, and a composite that combines shapers with weights.

## Serving

`src/serving/model_registry.py` contains two components.

**ModelRegistry** assigns semantic versions to registered models, tracks metadata and metrics, and supports transitioning versions through development/staging/production/archived stages. It can return the current production version and rank versions by a metric. The index is persisted as JSON.

**InferenceService** is a small HTTP server (standard-library `http.server`) that serves a policy. Given a state it returns the policy's action, and if a Q-table is provided it also returns the Q-values and a softmax-based confidence.

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/wittyswayam/rl-platform.git
cd rl-platform

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Optuna and MLflow are optional; the code degrades gracefully if they're absent (random search instead of TPE, local-only experiment tracking).

## Configuration

Training is configured through command-line flags in `scripts/train.py`:

| Flag | Default | Description |
|---|---|---|
| `--experiment_name` | `default_run` | Name used for tracking and checkpoints |
| `--num_iterations` | `100` | Training iterations |
| `--grid_size` | `8` | Grid dimension (N×N) |
| `--embed_dim` | `512` | Node2Vec embedding dimension |
| `--lr_node2vec` | `0.1` | Node2Vec learning rate |
| `--lr_infernet` | `0.01` | InferNet learning rate |
| `--seed` | `42` | Random seed |
| `--device` | `cpu` | Torch device |
| `--telemetry_port` | `9090` | Port for the telemetry HTTP server |
| `--no_checkpoint` | off | Write checkpoints to a temp dir instead of `checkpoints/` |
| `--config` | `None` | Accepted but not currently read (see Notes) |

The deployment manifests also reference environment variables. These are consumed by the container/orchestration layer rather than the Python code directly, but they document the intended runtime configuration:

```env
# Example .env for deployment
RL_DEVICE=cpu
RL_NUM_WORKERS=4
RL_EXPERIMENT_NAME=docker_run
RL_OUTPUT_DIR=/app/outputs
RL_CHECKPOINT_DIR=/app/checkpoints
RL_MODEL_PATH=/app/checkpoints/production/best_model.pt
MLFLOW_TRACKING_URI=http://mlflow:5000
```

## Running the Project

### CLI

```bash
# Default 8x8 grid, 100 iterations
python scripts/train.py

# Custom run
python scripts/train.py \
  --num_iterations 200 \
  --grid_size 8 \
  --embed_dim 512 \
  --lr_node2vec 0.1 \
  --seed 42 \
  --experiment_name my_run
```

The script builds the environment and agent, starts experiment tracking and the telemetry server, runs training, logs the final evaluation, and exits cleanly (marking the run FAILED if training raises).

### Programmatic

```python
from src.envs.graph_nav_env import GraphNavEnv, EnvConfig
from src.agents.node2vec_rl import Node2VecRLAgent, AgentConfig
from src.training.trainer import Trainer, TrainerConfig

env = GraphNavEnv(config=EnvConfig(grid_size=8, coin_nodes={10, 30, 50}, max_steps=128))
agent = Node2VecRLAgent(env, AgentConfig(embed_dim=512, num_iter=100))
trainer = Trainer(agent, env, TrainerConfig(experiment_name="my_experiment", num_iterations=100, seed=42))
results = trainer.train()
print(results["final_eval"]["eval/mean_return"])
```

### Distributed

```python
from src.training.distributed_trainer import DistributedTrainer, DistributedConfig
from src.envs.graph_nav_env import EnvConfig

trainer = DistributedTrainer(
    env_config=EnvConfig(grid_size=8, coin_nodes={10, 30, 50}),
    dist_config=DistributedConfig(num_workers=4, episodes_per_worker=16, num_iterations=100),
)
metrics = trainer.train()
```

### Evaluation

```python
from src.evaluation.evaluator import PolicyEvaluator

evaluator = PolicyEvaluator(env, n_eval_episodes=200, seed=0)
result = evaluator.evaluate_policy(agent.policy)
baseline = evaluator.evaluate_random_baseline()
print(result["mean_return"], result["ci_lower"], result["ci_upper"])
print("improvement over random:", result["mean_return"] - baseline["mean_return"])
```

## API Endpoints

Two HTTP services are included, both built on the standard-library HTTP server.

**Inference service** (`InferenceService`, default port 8080):

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Body `{"state": <int>}` returns the policy action, and Q-values + confidence if a Q-table is loaded |
| `GET` | `/health` | Returns status, uptime, and request count |

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"state": 42}'
```

**Telemetry service** (`TelemetryService`, default port 9090):

| Method | Path | Description |
|---|---|---|
| `GET` | `/metrics` | Metrics in Prometheus text exposition format |
| `GET` | `/summary` | Full metric summary as JSON |
| `GET` | `/health` | Returns `OK` |

## Deployment

### Docker

The `Dockerfile` is a two-stage build: a builder stage installs dependencies, and a slim runtime stage copies them in, runs as a non-root user, creates output directories, exposes port 9090, and defines a health check against the telemetry endpoint. The default entry point runs the training module.

```bash
docker build -f deployment/docker/Dockerfile -t rl-platform:2.0.0 .
docker run -p 9090:9090 -v $(pwd)/checkpoints:/app/checkpoints rl-platform:2.0.0
```

### Docker Compose

`deployment/docker/docker-compose.yml` defines a full stack: the trainer, an inference service, MLflow, Prometheus, Grafana, and Redis, wired together on a bridge network with named volumes for checkpoints, experiments, and monitoring data.

```bash
docker compose -f deployment/docker/docker-compose.yml up -d
```

### Kubernetes

`deployment/k8s/manifests.yaml` provisions a `rl-platform` namespace, a ConfigMap, two PersistentVolumeClaims (checkpoints and experiments), a trainer Deployment and an inference Deployment (with rolling-update strategy and liveness/readiness probes), a LoadBalancer service for inference, a ClusterIP service for telemetry, and a HorizontalPodAutoscaler that scales inference on CPU utilization.

```bash
kubectl apply -f deployment/k8s/manifests.yaml
kubectl get pods -n rl-platform
```

### Monitoring

`deployment/monitoring/prometheus.yml` scrapes the trainer and inference targets and Prometheus itself. Grafana is included in the compose stack for dashboards.

## Notebooks

The `notebooks/` directory contains four notebooks that trace the project's development: the original implementation, the delayed-reward optimization work, distributed training experiments, and benchmark analysis. They're useful for understanding how the packaged `src/` modules came together.

## Screenshots

_No images are included in the repository._ For a quick visual of agent behavior, `GraphNavEnv.render_policy()` prints the learned policy as an ASCII grid with directional arrows and coin markers, which is handy to drop into a notebook or terminal session.

```
[ Add a policy-grid render or a training-curve plot here ]
```

## Notes and Known Gaps

A few things in the repository are referenced but not yet present, which is worth knowing before relying on them:

- **No CI/CD pipeline is included** — there is no `.github/` directory or workflow file.
- **No `configs/` directory**, although the Docker and Kubernetes manifests pass `--config configs/default.yaml`. The `--config` flag in `scripts/train.py` is parsed but not read, so configuration currently comes from the other CLI flags. Add a config loader (or a `configs/default.yaml`) before using the container default command as-is.
- **No `scripts/serve.py`**, though the compose and k8s manifests launch inference with `python -m scripts.serve`. The serving logic exists in `src/serving/model_registry.py` (`InferenceService`); a thin entry-point script needs to be added to match the manifests.
- **No `tests/` directory** yet.
- **No `LICENSE` file** is present (see below).
- `requirements.txt` includes `gymnasium`, `ray`, `fastapi`, `uvicorn`, and `hydra-core`, which the current code does not import.
- `VectorizedEnv` runs environments sequentially despite its docstring mentioning multiprocessing.
- The inference service listens on port 8080 and does not expose a `/metrics` endpoint, while the Prometheus config scrapes inference on port 9090; align these if you want inference metrics scraped.

## Future Improvements

These follow directly from the gaps above and the existing structure:

- Add a `configs/` directory with YAML configs and wire up the `--config` flag (the dependencies already include a config library).
- Add `scripts/serve.py` as the inference entry point referenced by the deployment manifests.
- Add a test suite and a CI workflow.
- Expose inference metrics on the port Prometheus expects.
- Trim unused dependencies, or actually adopt them (e.g. make the environment Gymnasium-compatible, or move the experience queue to Redis).

## License

No license file is currently included in the repository, so usage and distribution terms are unspecified by default. If this is intended to be open source, add a `LICENSE` file (for example MIT) to make the terms explicit.
