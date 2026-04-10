# GNN for 0/1 Knapsack — Benchmark nhiều solver

Dự án nghiên cứu giải bài toán **0/1 Knapsack** bằng **Graph Neural Network (GNN)**, và so sánh với các phương pháp cổ điển cùng reinforcement learning.

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Yêu cầu hệ thống](#2-yêu-cầu-hệ-thống)
3. [Cài đặt](#3-cài-đặt)
4. [Cấu trúc project](#4-cấu-trúc-project)
5. [Danh sách solver](#5-danh-sách-solver)
6. [Pipeline sinh dữ liệu](#6-pipeline-sinh-dữ-liệu)
7. [Huấn luyện mô hình neural](#7-huấn-luyện-mô-hình-neural)
8. [Đánh giá từng solver](#8-đánh-giá-từng-solver)
9. [Pipeline tự động](#9-pipeline-tự-động)
10. [Vẽ biểu đồ](#10-vẽ-biểu-đồ)
11. [Thí nghiệm nâng cao](#11-thí-nghiệm-nâng-cao)
12. [Troubleshooting](#12-troubleshooting)
13. [Ghi chú kỹ thuật](#13-ghi-chú-kỹ-thuật)

---

## 1. Tổng quan

### 1.1 Bài toán

**0/1 Knapsack**: cho $n$ item, mỗi item có trọng lượng $w_i$ và giá trị $v_i$, cho dung lượng $C$ của ba lô. Tìm tập con các item để tối đa hoá tổng giá trị sao cho tổng trọng lượng không vượt $C$:

$$\max \sum_{i=1}^{n} v_i x_i \quad \text{s.t.} \quad \sum_{i=1}^{n} w_i x_i \leq C, \quad x_i \in \{0, 1\}$$

### 1.2 Mục tiêu

So sánh nhiều phương pháp giải trên cùng một tập dữ liệu, đánh giá trade-off giữa **chất lượng lời giải** (approximation ratio vs DP) và **thời gian chạy** (inference time).

### 1.3 Metric đánh giá

- **Approximation ratio** (vs DP optimal): `ratio = V_solver / V_DP`
- **Inference time** (ms/instance)
- **Feasibility rate**: tỷ lệ lời giải hợp lệ (không vi phạm capacity)
- **Head-to-head**: số instance mà solver A thắng solver B

**Lưu ý**: DP là ground truth. Mọi solver khác được đánh giá bằng ratio so với DP (DP luôn = 1.0).

---

## 2. Yêu cầu hệ thống

### 2.1 Phần cứng

- **CPU**: 4 cores trở lên (project được thiết kế chạy được trên CPU-only)
- **RAM**: tối thiểu 8 GB; khuyến nghị 16 GB cho dataset lớn (n lên tới 200)
- **Disk**: khoảng 2 GB cho dataset + checkpoint + results
- **GPU** (tùy chọn): CUDA-enabled, giúp tăng tốc training GNN/RL

### 2.2 Hệ điều hành

Đã test trên:
- Windows 10/11
- Ubuntu 20.04/22.04

### 2.3 Python

- Python **3.10** hoặc mới hơn

### 2.4 Thư viện chính

Xem `requirements.txt` để cài đặt chính xác. Các dependency quan trọng:

| Package | Mục đích |
|---|---|
| `torch` | Framework deep learning |
| `torch-geometric` | GNN layers (GINConv, SAGEConv, global_pool) |
| `numpy` | Xử lý mảng |
| `pulp` | ILP solver dùng để sinh dữ liệu |
| `matplotlib` | Vẽ biểu đồ |

---

## 3. Cài đặt

### 3.1 Clone hoặc tải project

```bash
cd <thư_mục_của_bạn>
# Copy/clone project về đây
cd GNNForKnapSack
```

### 3.2 Tạo môi trường ảo

**Windows (CMD)**:
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3.3 Cài dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Lưu ý `torch-geometric`**: nếu `pip install torch-geometric` báo lỗi, cài bằng wheel chính thức theo phiên bản PyTorch và CUDA của bạn. Xem: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### 3.4 Verify cài đặt

Chạy test suite:
```bash
python test_solvers.py
```

Nếu mọi test pass → môi trường đã sẵn sàng.

---

## 4. Cấu trúc project

Một số module cốt lõi được đặt trong sub-folder `Graph_Neural_Network/Knapsack_GNN/` theo cấu trúc thực tế của project — xem import paths trong các file để nắm vị trí cụ thể.

```
GNNForKnapSack/
│
├── Core libraries (không gọi trực tiếp)
│   ├── decode_utils.py          # Decode logic dùng chung (greedy_feasible_decode)
│   ├── instance_loader.py       # Load NPZ instances (W, V, C)
│   ├── rl_training_logger.py    # Logger cho RL training
│   ├── dataset.py               # PyG dataset wrapper
│   ├── Graph_builder.py         # Xây graph từ instance Knapsack
│   ├── io_excel.py              # Đọc/ghi Excel (optional)
│   │
│   ├── Dp.py                    # Dynamic Programming
│   ├── Greedy.py                # Greedy v/w ratio
│   ├── GA.py                    # Genetic Algorithm
│   └── model.py                 # KnapsackGNN architecture
│
├── Data generation
│   ├── Generate_Data.py         # Sinh dữ liệu bằng PuLP ILP (CHÍNH)
│   ├── Generate_Hard.py         # Sinh dữ liệu Pisinger hard instances
│   ├── Genhard_fixed.c          # Source C cho Pisinger generator
│   └── Check_Data.py            # Verify dataset
│
├── Training (neural models)
│   ├── Run_train.py                     # Train GNN supervised
│   ├── Train_eval.py                    # Helper train/eval
│   ├── train_dqn.py                     # Train DQN
│   ├── train_s2v_dqn.py                 # Train S2V-DQN
│   └── train_gnn_reinforce.py           # Train GNN + REINFORCE
│
├── Evaluation scripts (mỗi solver 1 file, CSV schema giống nhau)
│   ├── dp_baseline_eval.py      # DP
│   ├── greedy_baseline_eval.py  # Greedy
│   ├── ga_baseline_eval.py      # GA
│   ├── bb_baseline_eval.py      # Branch-and-Bound
│   ├── Evaluate_GNN.py          # GNN supervised
│   ├── Evaluate_DQN.py          # DQN
│   ├── Evaluate_S2V_DQN.py      # S2V-DQN
│   └── Evaluate_REINFORCE.py    # GNN + REINFORCE
│
├── RL submodules
│   ├── dqn_config.py            # Config DQN
│   ├── dqn_env.py               # Environment DQN
│   ├── dqn_model.py             # MLP Q-network
│   ├── dqn_replay.py            # Replay buffer
│   ├── s2v_env.py               # Environment S2V-DQN
│   ├── s2v_model.py             # Structure2Vec GNN
│   └── s2v_replay.py            # Replay buffer S2V
│
├── Benchmark & pipeline
│   ├── run_all_evaluations.py   # Chạy toàn bộ solver trên 1 dataset
│   ├── Merge_results.py         # Merge CSV của các solver
│   ├── cross_scale_eval.py      # Đánh giá cross-scale/cross-distribution
│   ├── Benchmark_Hard.py        # Benchmark Pisinger hard instances
│   └── plot_results.py          # Vẽ biểu đồ
│
├── Testing & configs
│   ├── test_solvers.py          # Test suite
│   ├── requirements.txt
│   ├── README.md
│   ├── CHANGELOG.md
│   └── configs/
│       ├── default.json
│       └── pisinger_hard.json
│
├── Data directories (tự động tạo khi sinh dữ liệu)
│   └── data/
│       ├── knapsack_ilp/
│       │   ├── train/           # instance_*.npz
│       │   ├── val/
│       │   └── test/
│       └── pisinger/            # Hard instances (nếu dùng)
│
└── Results directories (tự động tạo khi chạy eval)
    ├── results/
    │   ├── DP/
    │   ├── Greedy/
    │   ├── GA/
    │   ├── BB/
    │   ├── GNN/
    │   ├── DQN/
    │   ├── S2V_DQN/
    │   ├── GNN_REINFORCE/
    │   ├── compare/             # Merged results
    │   └── cross_scale/
    ├── logs/                    # Training logs
    └── plots/                   # Biểu đồ
```

---

## 5. Danh sách solver

Project có **8 solver**, chia thành 2 nhóm:

### 5.1 Nhóm cổ điển (classical)

| Solver | File chính            | Đặc điểm | Độ phức tạp |
|---|-----------------------|---|---|
| **DP** | `Dp.py`, `dp_baseline_eval.py` | Exact — dùng làm ground truth | $O(nC)$ |
| **Greedy** | `greedy_baseline_eval.py` | Sắp xếp theo v/w ratio, lấy từ cao xuống thấp | $O(n \log n)$ |
| **GA** | `ga_baseline_eval.py` | Genetic Algorithm với crossover + mutation | phụ thuộc population/generations |
| **BB** | `bb_baseline_eval.py` | Branch-and-Bound (Horowitz-Sahni) với LP upper bound | worst case $O(2^n)$ |

### 5.2 Nhóm neural

| Solver | File chính | Phương pháp học |
|---|---|---|
| **GNN supervised** | `Run_train.py`, `Evaluate_GNN.py` | BCE loss so với nhãn DP optimal |
| **GNN + REINFORCE** | `train_gnn_reinforce.py`, `Evaluate_REINFORCE.py` | Policy gradient — tối ưu trực tiếp value, không cần DP label |
| **DQN** | `train_dqn.py`, `Evaluate_DQN.py` | Deep Q-learning với MLP Q-network |
| **S2V-DQN** | `train_s2v_dqn.py`, `Evaluate_S2V_DQN.py` | Structure2Vec embedding + Q-learning |

### 5.3 Thống nhất CSV schema

Tất cả eval script xuất ra CSV với **cùng schema**:

```
instance_file, n_items, capacity, total_weight, total_value,
feasible, inference_time_ms, selected_items
```

Điều này cho phép `Merge_results.py` hợp nhất mọi kết quả thành 1 bảng duy nhất.

---

## 6. Pipeline sinh dữ liệu

### 6.1 Sinh dữ liệu uncorrelated (chính)

Dùng `Generate_Data.py` (backend: PuLP/ILP solver). Đây là script **duy nhất** dùng để sinh dữ liệu chính.

**Syntax**:
```
python Generate_Data.py <num_instances> <n_min> <n_max> -p <output_dir> -s <seed>
```

**Sinh 3 split cho training**:
```bash
# Train: 1000 instances
python Generate_Data.py 1000 10 200 -p data/knapsack_ilp/train -s 0

# Val: 200 instances
python Generate_Data.py 200 10 200 -p data/knapsack_ilp/val -s 100

# Test: 200 instances
python Generate_Data.py 200 10 200 -p data/knapsack_ilp/test -s 200
```

**Parameters mặc định trong `Generate_Data.py`**:
- `WEIGHT_LOW = 10`, `WEIGHT_HIGH = 1200`
- `VALUE_MULT_LOW = 0.8`, `VALUE_MULT_HIGH = 1.3` (value = weight × mult)
- Capacity được sinh để phủ spectrum từ 1% đến 99% tổng trọng lượng

**Output**: Mỗi instance là 1 file `.npz` chứa:
- `weights`: mảng trọng lượng
- `values`: mảng giá trị
- `capacity`: dung lượng
- `solution`: nghiệm tối ưu DP (dùng làm label cho supervised training)

### 6.2 Sinh Pisinger hard instances (dùng cho novelty experiment)

Pisinger instances là các benchmark **khó** vì weight và value tương quan mạnh — Greedy không còn hiệu quả, tạo cơ hội cho neural solver. Type 3/5/6 là các loại khó nhất.

```bash
# Sinh hard instances type 3 (strongly correlated)
python Generate_Hard.py --type 3 --n_items 100 --num_instances 1000 --out_dir data/pisinger/type_03/train
python Generate_Hard.py --type 3 --n_items 100 --num_instances 200  --out_dir data/pisinger/type_03/val
python Generate_Hard.py --type 3 --n_items 100 --num_instances 200  --out_dir data/pisinger/type_03/test
```

### 6.3 Verify dữ liệu đã sinh

```bash
python Check_Data.py --data_dir data/knapsack_ilp/train
```

Script này kiểm tra:
- Các file NPZ đọc được
- Shape của weights/values khớp với n_items
- Solution là binary array cùng size
- Feasibility của solution so với capacity

---

## 7. Huấn luyện mô hình neural

### 7.1 GNN supervised

**Lệnh cơ bản** (dùng default hyperparameters):
```bash
python Run_train.py \
    --generated_dir data/knapsack_ilp/train \
    --val_dir data/knapsack_ilp/val \
    --test_dir data/knapsack_ilp/test
```

**Các CLI argument quan trọng**:

| Argument | Default | Ý nghĩa |
|---|---|---|
| `--generated_dir` | `data/knapsack_ilp/train` | Thư mục training instances |
| `--val_dir` | (none) | Thư mục validation (nếu không set → tự split từ train) |
| `--test_dir` | (none) | Thư mục test (nếu không set → tự split từ train) |
| `--epochs` | 150 | Số epoch |
| `--batch_size` | 16 | Batch size |
| `--lr` | 5e-4 | Learning rate |
| `--warmup_epochs` | 5 | Số epoch warmup LR |
| `--hidden_dim` | 256 | Chiều hidden trong GNN |
| `--num_layers` | 4 | Số tầng GNN |
| `--dropout` | 0.15 | Dropout rate |
| `--conv_type` | `gin` | Loại conv: `gin`, `sage`, `hybrid` |
| `--k` | 16 | Số neighbor base cho kNN graph (adaptive theo n) |
| `--early_stop_wait` | 30 | Patience cho early stopping |
| `--lambda_cap_max` | 0.0 | Capacity penalty (0 = tắt, decode đã đảm bảo feasible) |

**Output**:
- `results/GNN/gnn.pt` — model của epoch cuối
- `results/GNN/gnn_best.pt` — model của epoch có ratio tốt nhất trên val
- `logs/training_log.csv` — log chi tiết từng epoch

**Training log schema**:
```
epoch, train_loss, val_acc, gnn_ratio, gnn_std, greedy_ratio,
gnn_beats_greedy, gnn_loses_greedy, ties, advantage, lr, lambda_cap, time_sec
```

### 7.2 DQN

```bash
python train_dqn.py \
    --train_dir data/knapsack_ilp/train \
    --val_dir data/knapsack_ilp/val \
    --test_dir data/knapsack_ilp/test \
    --train_steps 50000
```

**Output**:
- `results/DQN/dqn.pt` (final)
- `results/DQN/dqn_best.pt` (best by val)
- `results/DQN/training_log.csv`

### 7.3 S2V-DQN

```bash
python train_s2v_dqn.py \
    --train_dir data/knapsack_ilp/train \
    --val_dir data/knapsack_ilp/val \
    --test_dir data/knapsack_ilp/test \
    --train_steps 50000
```

**Output**:
- `results/S2V_DQN/s2v_dqn.pt`
- `results/S2V_DQN/s2v_dqn_best.pt`
- `results/S2V_DQN/training_log.csv`

### 7.4 GNN + REINFORCE

REINFORCE có thể khởi tạo từ checkpoint GNN supervised để làm **warm-start**:

```bash
python train_gnn_reinforce.py \
    --generated_dir data/knapsack_ilp/train \
    --val_dir data/knapsack_ilp/val \
    --test_dir data/knapsack_ilp/test \
    --pretrained results/GNN/gnn_best.pt \
    --epochs 50 \
    --lr 1e-5
```

**Warm-start rất quan trọng**: learning rate của REINFORCE nên thấp hơn nhiều so với supervised (1e-5 thay vì 5e-4) để không phá hỏng weights đã học.

**Output**:
- `results/GNN_REINFORCE/gnn_reinforce.pt`
- `logs/reinforce_training_log.csv`

### 7.5 Lưu ý quan trọng khi training

1. **Không tắt máy / Ctrl+C giữa chừng** — có thể làm checkpoint corrupt (0 byte)
2. **Disable sleep trên Windows**:
   ```cmd
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   ```
3. **Backup checkpoint ngay sau khi train xong**:
   ```cmd
   copy results\GNN\gnn_best.pt results\GNN\gnn_backup.pt
   ```
4. **Xóa dataset cache** nếu thay đổi feature dim hoặc đổi dataset:
   ```cmd
   del data\knapsack_ilp\train\processed_dataset.pt
   del data\knapsack_ilp\val\processed_dataset.pt
   del data\knapsack_ilp\test\processed_dataset.pt
   ```

---

## 8. Đánh giá từng solver

Mỗi solver có file eval riêng. Tất cả đều có CLI giống nhau.

### 8.1 Các argument chung

| Argument | Ý nghĩa |
|---|---|
| `--dataset_dir` | Thư mục chứa `instance_*.npz` |
| `--out_csv` | File CSV output |
| `--n` | Giới hạn N instance đầu (để test nhanh) |

### 8.2 Chạy từng solver

**DP** (ground truth — chạy đầu tiên):
```bash
python dp_baseline_eval.py \
    --dataset_dir data/knapsack_ilp/test \
    --out_csv results/DP/dp_results.csv
```

**Greedy**:
```bash
python greedy_baseline_eval.py \
    --dataset_dir data/knapsack_ilp/test \
    --out_csv results/Greedy/greedy_eval_results.csv
```

**GA**:
```bash
python ga_baseline_eval.py \
    --dataset_dir data/knapsack_ilp/test \
    --out_csv results/GA/ga_eval_results.csv \
    --population 100 \
    --generations 500 \
    --mutation_rate 0.05
```

**BB**:
```bash
python bb_baseline_eval.py \
    --dataset_dir data/knapsack_ilp/test \
    --out_csv results/BB/bb_results.csv \
    --timeout_sec 60 \
    --max_nodes 2000000
```

**Lưu ý BB**: timeout mặc định 60s/instance. Với dataset hard (Pisinger), nên tăng lên 120s hoặc 300s. Nếu quá chậm, có thể dùng `--skip bb` trong pipeline tự động.

**GNN supervised**:
```bash
python Evaluate_GNN.py \
    --dataset_dir data/knapsack_ilp/test \
    --model_path results/GNN/gnn_best.pt \
    --out_csv results/GNN/gnn_eval_results.csv
```

**DQN**:
```bash
python Evaluate_DQN.py \
    --dataset_dir data/knapsack_ilp/test \
    --model_path results/DQN/dqn_best.pt \
    --out_csv results/DQN/dqn_eval_results.csv
```

**S2V-DQN**:
```bash
python Evaluate_S2V_DQN.py \
    --dataset_dir data/knapsack_ilp/test \
    --model_path results/S2V_DQN/s2v_dqn_best.pt \
    --out_csv results/S2V_DQN/s2v_dqn_eval_results.csv
```

**REINFORCE**:
```bash
python Evaluate_REINFORCE.py \
    --dataset_dir data/knapsack_ilp/test \
    --model_path results/GNN_REINFORCE/gnn_reinforce.pt \
    --out_csv results/GNN_REINFORCE/reinforce_eval_results.csv
```

### 8.3 Merge kết quả

Sau khi có các CSV, merge lại thành 1 bảng để so sánh:

```bash
python Merge_results.py \
    --dp_csv results/DP/dp_results.csv \
    --greedy_csv results/Greedy/greedy_eval_results.csv \
    --ga_csv results/GA/ga_eval_results.csv \
    --bb_csv results/BB/bb_results.csv \
    --gnn_csv results/GNN/gnn_eval_results.csv \
    --dqn_csv results/DQN/dqn_eval_results.csv \
    --s2v_csv results/S2V_DQN/s2v_dqn_eval_results.csv \
    --reinforce_csv results/GNN_REINFORCE/reinforce_eval_results.csv \
    --out_dir results/compare
```

**Output**:
- `results/compare/merged_results.csv` — per-instance comparison, mỗi row 1 instance, mỗi solver có cột `<solver>_value`, `<solver>_ratio`, `<solver>_time`
- `results/compare/summary.json` — summary statistics (avg ratio, avg time, feasibility rate cho từng solver)

---

## 9. Pipeline tự động

### 9.1 Chạy toàn bộ eval trong 1 lệnh

`run_all_evaluations.py` chạy tất cả solver (DP + Greedy + GA + BB + GNN + DQN + S2V + REINFORCE) rồi merge kết quả.

```bash
python run_all_evaluations.py --dataset_dir data/knapsack_ilp/test
```

**Các flag quan trọng**:

| Flag | Ý nghĩa |
|---|---|
| `--dataset_dir` | Dataset để eval |
| `--skip` | Skip solver nào đó (ví dụ: `--skip bb reinforce`) |
| `--only` | Chỉ chạy solver được liệt kê (ví dụ: `--only dp greedy gnn`) |
| `--model_path` | Đường dẫn model GNN (default: `results/GNN/gnn.pt`) |
| `--dqn_model` | Checkpoint DQN |
| `--s2v_model` | Checkpoint S2V-DQN |
| `--reinforce_model` | Checkpoint REINFORCE |
| `--bb_timeout` | Timeout BB (default 60s) |
| `--bb_max_nodes` | Max nodes BB (default 2M) |
| `--n` | Giới hạn N instance đầu |

**Ví dụ**:
```bash
# Chỉ chạy baseline classical (bỏ neural models)
python run_all_evaluations.py \
    --dataset_dir data/knapsack_ilp/test \
    --only dp greedy ga bb

# Chạy tất cả nhưng skip BB (vì quá chậm trên hard instances)
python run_all_evaluations.py \
    --dataset_dir data/knapsack_ilp/test \
    --skip bb
```

### 9.2 Điều kiện tiên quyết

Để `run_all_evaluations.py` chạy thành công với đầy đủ solver, cần có sẵn:
- Dataset đã sinh (`data/knapsack_ilp/test/instance_*.npz`)
- Model checkpoint cho các neural solver trong `results/*/`

Nếu thiếu checkpoint nào → script sẽ báo `[SKIP]` cho solver đó và tiếp tục.

---

## 10. Vẽ biểu đồ

Script `plot_results.py` sinh ra các biểu đồ so sánh.

### 10.1 Các loại biểu đồ

| Biểu đồ | Input cần có | Output file |
|---|---|---|
| Solver comparison bar chart | `summary.json` | `solver_comparison.png` |
| Ratio distribution histogram | `merged_results.csv` | `ratio_distribution.png` |
| Ratio theo problem size | `merged_results.csv` | `ratio_by_size.png` |
| GNN vs Greedy head-to-head scatter | `merged_results.csv` | `head_to_head.png` |
| Training curve | Training log CSV | `<parent>_<stem>_curve.png` |
| Cross-scale bar chart | `cross_scale_results.csv` | `cross_scale.png` |

### 10.2 Lệnh vẽ

**Vẽ từ kết quả đã merge**:
```bash
python plot_results.py \
    --results_dir results/compare \
    --out_dir plots/
```

**Vẽ training curve**:
```bash
python plot_results.py \
    --training_log logs/training_log.csv \
    --out_dir plots/
```

**Vẽ nhiều training log 1 lần**:
```bash
python plot_results.py \
    --training_logs \
        logs/training_log.csv \
        results/DQN/training_log.csv \
        results/S2V_DQN/training_log.csv \
    --out_dir plots/
```

**Vẽ tất cả cùng lúc**:
```bash
python plot_results.py \
    --results_dir results/compare \
    --training_log logs/training_log.csv \
    --cross_scale_csv results/cross_scale/cross_scale_results.csv \
    --out_dir plots/all
```

### 10.3 Debug khi chỉ gen được ít ảnh

Nếu lệnh trên chỉ sinh được ít biểu đồ hơn mong đợi, `plot_results.py` sẽ in rõ lý do skip từng chart. Thường do:
- Thiếu `merged_results.csv` → 3 chart ratio_distribution / ratio_by_size / head_to_head không chạy
- Thiếu cột `gnn_ratio` trong CSV → head_to_head skip
- Training log rỗng hoặc sai format

Xem stdout output để biết chính xác vấn đề.

---

## 11. Thí nghiệm nâng cao

### 11.1 Cross-scale evaluation

Train trên 1 size range, test trên nhiều size range khác để đánh giá **generalization**:

```bash
python cross_scale_eval.py \
    --extra_models gnn:results/GNN/gnn_best.pt reinforce:results/GNN_REINFORCE/gnn_reinforce.pt \
    --test_sets \
        data/knapsack_ilp/test \
        data/knapsack_n50/test \
        data/knapsack_n100/test \
        data/knapsack_n200/test \
    --labels "mixed" "n=50" "n=100" "n=200" \
    --include_baselines \
    --bb_timeout 120 \
    --out_dir results/cross_scale
```

**Output**: `results/cross_scale/cross_scale_results.csv` — bảng so sánh ratio của mỗi solver trên từng test set.

### 11.2 Cross-distribution (Pisinger hard)

Test mô hình đã train trên uncorrelated data lên Pisinger strongly correlated:

```bash
# Sinh data
python Generate_Hard.py --type 3 --n_items 100 --num_instances 200 \
    --out_dir data/pisinger/type_03/test

# Cross-scale eval
python cross_scale_eval.py \
    --extra_models gnn:results/GNN/gnn_best.pt \
    --test_sets \
        data/knapsack_ilp/test \
        data/pisinger/type_03/test \
    --labels "uncorrelated" "pisinger_t3" \
    --include_baselines
```

### 11.3 Benchmark nhiều hard types

```bash
python Benchmark_Hard.py \
    --types 3 5 6 \
    --n_items 100 \
    --num_instances 100
```

### 11.4 Tại sao Pisinger hard quan trọng

Trên dữ liệu uncorrelated (value và weight độc lập), **Greedy theo v/w ratio rất mạnh** — gần như tối ưu. Neural solver khó vượt qua Greedy trên loại này.

Trên Pisinger strongly correlated (type 3/5/6), v/w ratio gần như giống nhau cho mọi item → Greedy không còn heuristic tốt. Đây là nơi neural solver có cơ hội thể hiện ưu thế.

---

## 12. Troubleshooting

### 12.1 Lỗi thường gặp

**`EOFError: Ran out of input` khi load checkpoint**
- Nguyên nhân: file `.pt` rỗng (0 byte), thường do training bị Ctrl+C hoặc máy sleep lúc save
- Cách fix:
  1. Check size file: `dir results\GNN\`
  2. Nếu `gnn_best.pt` còn → dùng file đó thay cho `gnn.pt`
  3. Nếu cả hai đều rỗng → train lại, disable sleep trước khi train
  4. Kiểm tra PyCharm Local History có lưu version cũ không

**`FileNotFoundError: No instance_*.npz in <dir>`**
- Nguyên nhân: Chưa sinh dữ liệu hoặc đường dẫn sai
- Fix: Chạy `Generate_Data.py` trước

**Checkpoint load báo `Missing keys` hoặc `Unexpected keys`**
- Nguyên nhân: Model architecture thay đổi giữa lần train và lần load
- Fix: Hiện tại `load_checkpoint` dùng `strict=False`, các key thiếu được init lại. Nếu khác biệt quá lớn → train lại.

**Feature dim mismatch (6 vs 7)**
- Nguyên nhân: Dataset cache `processed_dataset.pt` được build với version cũ của `Graph_builder.py`
- Fix: Xóa cache và re-generate:
  ```cmd
  del data\knapsack_ilp\train\processed_dataset.pt
  del data\knapsack_ilp\val\processed_dataset.pt
  del data\knapsack_ilp\test\processed_dataset.pt
  ```

**Training log chỉ có vài epoch rồi dừng**
- Có thể do early stopping kích hoạt — kiểm tra cột `advantage` trong log xem có plateau không
- Hoặc do OOM — giảm `--batch_size`
- Hoặc do Ctrl+C / sleep — xem mục trên

**`plot_results.py` chỉ sinh được ít ảnh**
- Xem stdout output — script in rõ `[SKIP]` từng chart kèm lý do
- Phổ biến: thiếu `merged_results.csv` → cần chạy `Merge_results.py` trước

**BB chạy rất lâu trên 1 instance**
- Tăng `--timeout_sec` hoặc skip solver này: `--skip bb`
- BB worst case $O(2^n)$, với Pisinger type 3/5/6 có thể không kết thúc trong thời gian hợp lý

### 12.2 Chạy test suite

```bash
python test_solvers.py
```

Script test các solver cốt lõi với các instance test cases. Nếu pass hết → môi trường OK.

---

## 13. Ghi chú kỹ thuật

### 13.1 Kiến trúc GNN

- **Backbone**: GINConv (default) hoặc SAGEConv
- **Node features** (7-dim):
  1. `w_norm` — `w_i / max(w)`
  2. `v_norm` — `v_i / max(v)`
  3. `ratio_norm` — `(v_i/w_i) / max(v/w)`
  4. `cap_ratio` — `w_i / capacity`
  5. `cap_util` — `capacity / sum(w)` (global)
  6. `item_frac` — `1 / n_items` (global)
  7. `w_vs_mean` — `w_i / mean(w)`
- **Graph**: k-NN trong feature space, k adaptive theo $n$
- **Normalization**: LayerNorm (để ổn định với batch có nhiều kích thước graph khác nhau)
- **Global context injection**: attention pooling, vectorized bằng `scatter_softmax`
- **Residual connections** giữa các layer

### 13.2 Decode logic

File `decode_utils.py` cung cấp `greedy_feasible_decode` — **single source of truth** cho việc chuyển probability thành binary solution:
1. Sort items theo probability (descending)
2. Thêm từng item nếu còn capacity
3. Đảm bảo solution luôn feasible

Mọi eval script (GNN, DQN, S2V, REINFORCE) đều gọi function này để tránh inconsistency.

### 13.3 Training loop GNN supervised

- **Loss**: BCE giữa sigmoid(logits) và DP label
- **Optimizer**: AdamW với weight_decay 1e-5
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=25, T_mult=2)
- **Warmup**: Linear warmup 5 epochs đầu
- **Gradient clipping**: max_norm=1.0
- **Evaluation per epoch**: compute GNN ratio và so với Greedy baseline trên val set

### 13.4 Giới hạn của supervised learning

Supervised với BCE loss có **ceiling** lý thuyết: model học cách bắt chước DP. Nếu decode không hoàn hảo, kết quả có thể thua Greedy trên dữ liệu uncorrelated (nơi Greedy đã rất gần tối ưu).

Đây là động lực để dùng **REINFORCE** (tối ưu trực tiếp value, không cần DP label) hoặc **chuyển sang hard instances** (nơi Greedy không mạnh).

### 13.5 Capacity generation

Trong `Generate_Data.py`, capacity của instance thứ $i$ trong bộ $N$ instance được sinh như sau:

$$C_i = \frac{i+1}{N+1} \cdot \sum_j w_j$$

Điều này tạo ra spectrum capacity từ ~1% đến ~99% tổng trọng lượng, giúp đánh giá solver trên nhiều loại instance (easy → tight).

---

## Liên hệ và đóng góp

Nếu gặp bug hoặc có đề xuất cải tiến:
1. Check mục [Troubleshooting](#12-troubleshooting)
2. Chạy `python test_solvers.py` xem environment có OK không
3. Kiểm tra `CHANGELOG.md` để biết các thay đổi gần nhất

Chúc may mắn với nghiên cứu!