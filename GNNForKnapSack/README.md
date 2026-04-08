# Knapsack Solver Benchmark: GNN vs DQN vs Classical Methods

A comprehensive benchmark comparing **Graph Neural Networks (GNN)**, **Deep Q-Networks (DQN)**, and classical algorithms (DP, Greedy, GA, B&B) on the 0/1 Knapsack problem.

## Project Structure

```
├── CORE LIBRARIES
│   ├── decode_utils.py          # Centralized decode (greedy feasible, ratio-based)
│   ├── instance_loader.py       # NPZ instance loader (all solvers use this)
│   ├── Dp.py                    # DP solver (NumPy-accelerated)
│   ├── Greedy.py                # Greedy v/w ratio solver
│   ├── GA.py                    # Genetic Algorithm solver
│   ├── Graph_builder.py         # kNN graph builder (6-dim features)
│   ├── dataset.py               # PyG dataset classes
│   └── io_excel.py              # Excel data loader (legacy)
│
├── GNN
│   ├── model.py                 # KnapsackGNN (GINConv + GlobalContext)
│   ├── Run_train.py             # Training with Greedy comparison
│   └── Train_eval.py            # Training/eval utilities
│
├── DQN
│   ├── dqn_model.py             # Q-Network (MLP)
│   ├── dqn_env.py               # Sequential knapsack environment
│   ├── dqn_config.py            # Hyperparameters
│   ├── dqn_replay.py            # Replay buffer
│   └── train_dqn.py             # DQN training script
│
├── DATA GENERATION
│   ├── Generate_Data.py         # Random instances (PuLP/ILP solver)
│   ├── Generate_Hard.py         # Pisinger hard instances (Genhard.c)
│   ├── Genhard.c                # Pisinger generator (C source)
│   └── Check_Data.py            # Dataset validator
│
├── EVALUATION
│   ├── dp_baseline_eval.py      # DP evaluation
│   ├── greedy_baseline_eval.py  # Greedy evaluation
│   ├── ga_baseline_eval.py      # GA evaluation
│   ├── Evaluate_GNN.py          # GNN evaluation
│   └── Evaluate_DQN.py          # DQN evaluation
│
├── COMPARISON
│   ├── Merge_results.py         # Merge all solver CSVs
│   ├── run_all_evaluations.py   # Master pipeline (one command)
│   ├── Benchmark_Hard.py        # Pisinger benchmark
│   └── plot_results.py          # Visualization & charts
│
└── TESTING
    ├── test_solvers.py          # Unit tests for all solvers
    ├── requirements.txt         # Dependencies
    └── configs/                 # Experiment configurations
        └── default.json
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate data

```bash
Hãy đứng ở mục GNNForKnapSack
nd
# Training data (1000 instances, n=10-100 items)
python scripts/Generate_Data.py 1000 10 100 -p data/knapsack_ilp/train -s 0

# Validation data (200 instances)
python scripts/Generate_Data.py 200 10 100 -p data/knapsack_ilp/val -s 100

# Test data (200 instances)
python scripts/Generate_Data.py 200 10 100 -p data/knapsack_ilp/test -s 200

# Validate generated data
python script/Check_Data.py data/knapsack_ilp/train
```

### 3. Train GNN
```bash
python Run_train.py \
    --generated_dir data/knapsack_ilp/train \
    --val_dir data/knapsack_ilp/val \
    --test_dir data/knapsack_ilp/test \
    --conv_type gin \
    --epochs 100
```

### 4. Train DQN

```bash
Đứng ở thư mục DQN
python train_dqn.py \
    --dataset_dir ../../data/knapsack_ilp/train \
    --train_steps 50000
```

### 5. Evaluate all solvers

```bash
# One by one
Đứng ở thư mục chứa project
python -m GNNForKnapSack.solvers.DP.dp_baseline_eval     --dataset_dir GNNForKnapSack/data/knapsack_ilp/test
python -m GNNForKnapSack.solvers.Greedy.Greedy  --dataset_dir GNNForKnapSack/data/knapsack_ilp/test
python -m GNNForKnapSack.solvers.GA.GA      --dataset_dir GNNForKnapSack/data/knapsack_ilp/test
python Evaluate_GNN.py          --dataset_dir data/knapsack_ilp/test --model_path results/GNN/gnn.pt
python Evaluate_DQN.py          --dataset_dir data/knapsack_ilp/test --model_path results/DQN/dqn.pt

# Or all at once
python run_all_evaluations.py --dataset_dir data/test

# Merge results
python Merge_results.py

# Plot comparison
python plot_results.py --results_dir results/compare
```

### 6. Hard instances (Pisinger benchmark)

```bash
# Generate strongly correlated instances
python scripts/Generate_Hard.py --type 3 --n_items 100 --num_instances 200

# Full benchmark
python scripts/Benchmark_Hard.py --types 1 3 5 8 --n_items 100 --num_instances 200
```

## Solvers

| Solver | Type | Labels needed | Approach |
|--------|------|---------------|----------|
| DP | Exact | — | Dynamic programming O(nC) |
| Greedy | Heuristic | No | Sort by v/w ratio |
| GA | Metaheuristic | No | Evolutionary search + repair |
| GNN | Supervised | Yes (DP) | Node classification (BCE loss) |
| DQN | RL | No | Sequential decision (reward = value) |

## CSV Schema

All evaluation scripts output the same CSV schema:

```
instance_file, n_items, capacity, total_weight, total_value,
feasible, inference_time_ms, selected_items
```

This allows `Merge_results.py` to join any combination of solvers.



## Running Tests

```bash
python -m pytest test_solvers.py -v
```
