# Knapsack Solver Benchmark: 3 Neural Approaches vs Classical Methods

A comprehensive benchmark for the **0/1 Knapsack problem** comparing:
- **3 Neural approaches**: GNN (supervised), GNN+REINFORCE (policy gradient), DQN (MLP), S2V-DQN (GNN+Q-learning)
- **3 Classical methods**: Dynamic Programming (DP), Greedy, Genetic Algorithm (GA)

## Novelty Directions

1. **Scalability** — train on small instances, test on larger ones (Hướng 1)
2. **Generalization** — train on uncorrelated, test on Pisinger strongly correlated (Hướng 2)

## Project Structure

```
GNNForKnapSack/
│
├── CORE LIBRARIES (shared utilities)
│   ├── decode_utils.py              # Greedy feasible decode (used by 10+ files)
│   ├── instance_loader.py           # NPZ loader (used by 9+ files)
│   ├── rl_training_logger.py        # Shared logger for DQN/S2V/REINFORCE
│   ├── Dp.py                        # DP solver (NumPy-accelerated)
│   ├── Greedy.py                    # Greedy v/w ratio solver
│   ├── GA.py                        # Genetic Algorithm
│   ├── Graph_builder.py             # kNN graph (6-dim features)
│   ├── dataset.py                   # PyG dataset
│   └── io_excel.py                  # Excel loader (legacy)
│
├── DATA GENERATION
│   ├── Generate_Data.py             # Random instances (PuLP/ILP)
│   ├── Generate_Hard.py             # Pisinger hard instances
│   ├── Genhard_fixed.c              # Pisinger generator source
│   └── Check_Data.py                # Dataset validator
│
├── GNN (supervised)
│   ├── model.py                     # KnapsackGNN (GINConv + GlobalContext)
│   ├── Run_train.py                 # Supervised training with Greedy comparison
│   ├── Train_eval.py                # Training utilities
│   └── Evaluate_GNN.py              # GNN evaluation
│
├── CLASSICAL BASELINES (standalone eval scripts)
│   ├── dp_baseline_eval.py
│   ├── greedy_baseline_eval.py
│   └── ga_baseline_eval.py
│
├── COMPARISON & PIPELINE
│   ├── Merge_results.py             # Merge all solver CSVs
│   ├── run_all_evaluations.py       # Master pipeline (one command)
│   ├── cross_scale_eval.py          # Cross-scale / cross-distribution eval
│   ├── Benchmark_Hard.py            # Pisinger benchmark
│   └── plot_results.py              # Visualization
│
├── TESTING & CONFIG
│   ├── test_solvers.py              # Unit tests (22 tests)
│   ├── requirements.txt             # Dependencies
│   └── configs/                     # Experiment configurations
│       ├── default.json
│       └── pisinger_hard.json
│
└── Reinforcement_Learning/
    ├── DQN/                         # Deep Q-Network (MLP-based)
    │   ├── dqn_config.py
    │   ├── dqn_env.py
    │   ├── dqn_model.py             # QNetwork (MLP)
    │   ├── dqn_replay.py
    │   ├── train_dqn.py
    │   └── Evaluate_DQN.py
    │
    ├── S2V_DQN/                     # GNN + Q-learning (Dai et al. 2017)
    │   ├── s2v_env.py               # Graph environment
    │   ├── s2v_model.py             # S2VQNetwork (GNN + Q-head)
    │   ├── s2v_replay.py            # Graph replay buffer
    │   ├── train_s2v_dqn.py
    │   └── Evaluate_S2V_DQN.py
    │
    └── REINFORCE/                   # GNN + Policy Gradient (Kool et al. 2019)
        ├── train_gnn_reinforce.py   # REINFORCE with greedy rollout baseline
        └── Evaluate_REINFORCE.py
```

## Solver Comparison

| Solver | Type | Labels needed | Approach |
|--------|------|---------------|----------|
| DP | Exact | — | O(n × C) bottom-up |
| Greedy | Heuristic | No | Sort by value/weight ratio |
| GA | Metaheuristic | No | Evolution + repair |
| **GNN (supervised)** | Neural | **Yes (DP)** | BCE loss against DP labels |
| **DQN (MLP)** | Neural RL | No | Q-learning on flat state |
| **S2V-DQN (GNN)** | Neural RL | No | GNN encoder + per-node Q-values |
| **GNN+REINFORCE** | Neural RL | No | Policy gradient with greedy baseline |

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 2. Generate data

```bash
# Three separate directories: train / val / test
python Generate_Data.py 1000 10 100 -p data/knapsack_ilp/train -s 0
python Generate_Data.py 200  10 100 -p data/knapsack_ilp/val   -s 100
python Generate_Data.py 200  10 100 -p data/knapsack_ilp/test  -s 200

# Validate
python Check_Data.py data/knapsack_ilp/train
```

### 3. Train all 4 neural models

```bash
# 3a. GNN supervised (baseline for REINFORCE warm-start)
python Run_train.py \
    --generated_dir data/knapsack_ilp/train \
    --val_dir data/knapsack_ilp/val \
    --test_dir data/knapsack_ilp/test \
    --conv_type gin --epochs 100

# 3b. GNN + REINFORCE (warm-start from supervised)
python Reinforcement_Learning/REINFORCE/train_gnn_reinforce.py \
    --generated_dir data/knapsack_ilp/train \
    --val_dir data/knapsack_ilp/val \
    --test_dir data/knapsack_ilp/test \
    --pretrained results/GNN/gnn.pt \
    --epochs 50

# 3c. DQN (MLP)
python Reinforcement_Learning/DQN/train_dqn.py \
    --dataset_dir data/knapsack_ilp/train \
    --val_dir data/knapsack_ilp/val \
    --test_dir data/knapsack_ilp/test \
    --train_steps 50000

# 3d. S2V-DQN (GNN)
python Reinforcement_Learning/S2V_DQN/train_s2v_dqn.py \
    --dataset_dir data/knapsack_ilp/train \
    --val_dir data/knapsack_ilp/val \
    --test_dir data/knapsack_ilp/test \
    --train_steps 50000
```

### 4. Evaluate everything

```bash
# Run all solvers on test set
python dp_baseline_eval.py      --dataset_dir data/knapsack_ilp/test
python greedy_baseline_eval.py  --dataset_dir data/knapsack_ilp/test
python ga_baseline_eval.py      --dataset_dir data/knapsack_ilp/test
python Evaluate_GNN.py          --dataset_dir data/knapsack_ilp/test --model_path results/GNN/gnn.pt
python Reinforcement_Learning/REINFORCE/Evaluate_REINFORCE.py \
    --dataset_dir data/knapsack_ilp/test \
    --model_path results/GNN_REINFORCE/gnn_reinforce.pt
python Reinforcement_Learning/DQN/Evaluate_DQN.py \
    --dataset_dir data/knapsack_ilp/test \
    --model_path results/DQN/dqn_best.pt
python Reinforcement_Learning/S2V_DQN/Evaluate_S2V_DQN.py \
    --dataset_dir data/knapsack_ilp/test \
    --model_path results/S2V_DQN/s2v_dqn_best.pt

# Merge all results
python Merge_results.py \
    --reinforce_csv results/GNN_REINFORCE/reinforce_eval_results.csv \
    --s2v_csv       results/S2V_DQN/s2v_dqn_eval_results.csv

# Or use master pipeline (runs everything + merge)
python run_all_evaluations.py --dataset_dir data/knapsack_ilp/test
```

### 5. Novelty experiments

**Experiment A — Cross-distribution (generalization)**:

```bash
# Generate Pisinger strongly correlated instances
python Generate_Hard.py --type 3 --n_items 100 --num_instances 200 \
    --out_dir data/pisinger/type_03/test

# Cross-evaluate trained models
python cross_scale_eval.py \
    --extra_models \
        gnn:results/GNN/gnn.pt \
        reinforce:results/GNN_REINFORCE/gnn_reinforce.pt \
        s2v:results/S2V_DQN/s2v_dqn_best.pt \
    --test_sets \
        data/knapsack_ilp/test \
        data/pisinger/type_03/test \
    --labels "uncorr_n100" "pisinger_n100" \
    --include_baselines
```

**Experiment B — Cross-scale (scalability)**:

```bash
# Generate larger instances
python Generate_Data.py 100 150 200 -p data/knapsack_ilp/test_n200 -s 500
python Generate_Data.py 100 400 500 -p data/knapsack_ilp/test_n500 -s 600

python cross_scale_eval.py \
    --extra_models \
        gnn:results/GNN/gnn.pt \
        reinforce:results/GNN_REINFORCE/gnn_reinforce.pt \
        s2v:results/S2V_DQN/s2v_dqn_best.pt \
    --test_sets \
        data/knapsack_ilp/test \
        data/knapsack_ilp/test_n200 \
        data/knapsack_ilp/test_n500 \
    --labels "n10-100" "n150-200" "n400-500" \
    --include_baselines
```

### 6. Plot everything

```bash
python plot_results.py \
    --results_dir results/compare \
    --cross_scale_csv results/cross_scale/cross_scale_results.csv \
    --training_logs \
        logs/training_log.csv \
        results/DQN/training_log.csv \
        results/S2V_DQN/training_log.csv \
        logs/reinforce_training_log.csv
```

## CSV Schema

All evaluation scripts output the same schema for `Merge_results.py`:

```
instance_file, n_items, capacity, total_weight, total_value,
feasible, inference_time_ms, selected_items
```

## Key Design Decisions

- **GINConv over SAGEConv**: SUM aggregation naturally models knapsack weight summation
- **3 layers (not 4)**: enough for kNN graphs with k=16, saves 19% params
- **Separate train/val/test directories**: prevents data leakage
- **REINFORCE with greedy rollout baseline**: Kool et al. 2019 standard for variance reduction
- **100% feasibility guaranteed**: enforced by `greedy_feasible_decode` (GNN) and action masking (DQN/S2V)

## Testing

```bash
python -m pytest test_solvers.py -v
```

22 unit tests covering DP, Greedy, GA, decode_utils, instance_loader, graph_builder, DQN env, and cross-solver consistency.

## Tips

**Disable laptop sleep during training** to get accurate timing:

```cmd
# Windows
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```

**Warm-start REINFORCE from supervised GNN** for faster convergence:
```bash
python train_gnn_reinforce.py --pretrained results/GNN/gnn.pt ...
```

## Expected Results (reference)

On uncorrelated instances n=10–100:
- DP: ratio 1.000 (exact)
- GA: ratio 0.999
- Greedy: ratio 0.993 (baseline to beat)
- GNN-supervised: ratio 0.97–0.98 (bounded by DP labels)
- GNN+REINFORCE: ratio ≥ Greedy (theoretical ceiling = DP)
- S2V-DQN: ratio varies (requires careful tuning)
- DQN-MLP: weakest (no graph structure)

On Pisinger type 3 strongly correlated:
- Greedy drops to ~0.987 → **window where neural approaches can win**