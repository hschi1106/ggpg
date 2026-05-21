# Batched GPU Evaluation for GP-GOMEA

This fork keeps `cpu_original` as the upstream GP-GOMEA baseline and adds two CUDA-enabled execution modes for symbolic regression:

- `gpu_exact_gom`: one individual and one GOM move are evaluated at a time on GPU. This preserves the original accept/reject ordering as closely as practical, but it can be launch-overhead limited.
- `gpu_batch_gom`: multiple trial candidates are serialized and evaluated in one CUDA launch. This improves GPU utilization but delays population commits until the current population batch is complete, so it is an approximate acceleration variant.

CUDA kernels never consume `Node*`, `Op*`, or tree pointers. Expressions are serialized on CPU into postfix `GpuToken` programs and copied to fixed-stride GPU buffers.

## Build

CPU-only:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=release
cmake --build build/release
```

CUDA:

```bash
GPG_USE_CUDA=1 cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=release -DGPG_USE_CUDA=ON
cmake --build build/cuda
```

If a GPU backend is requested from a CPU-only build, the program fails with:

```text
CUDA backend requested, but project was built without GPG_USE_CUDA.
```

## Run Backends

CPU baseline:

```bash
./build/release/gpg -train dataset/diabetes_train.csv -ff mse -backend cpu_original -disable_ims -g 10 -verbose
```

Exact GPU GOM:

```bash
./build/cuda/gpg -train dataset/diabetes_train.csv -ff mse -backend gpu_exact_gom -disable_ims -g 10 -verbose
```

Batched GPU GOM:

```bash
./build/cuda/gpg -train dataset/diabetes_train.csv -ff mse -backend gpu_batch_gom -gpu_batch_size 32 -disable_ims -g 10 -verbose
```

Optional correctness sampling:

```bash
./build/cuda/gpg -train dataset/diabetes_train.csv -ff mse -fset +,-,*,/ -backend gpu_exact_gom -gpu_check_correctness -disable_ims -g 2 -verbose
```

GPU backends currently support `-ff mse` and `-ff mae`. Use `cpu_original` for `ac`.

## Reproduction Harness

The editable paper-style settings live in:

```text
experiments/paper_reproduction_config.yaml
```

Prepare datasets:

```bash
python3 experiments/download_datasets.py
python3 experiments/prepare_paper_datasets.py
```

Run the fast development benchmark:

```bash
python3 experiments/run_paper_cpu_baseline.py --datasets "Yacht hydrodynamics" --seeds 0 1 --tree-heights 4 --time-limit 300
python3 experiments/run_gpu_exact.py --datasets "Yacht hydrodynamics" --seeds 0 1 --time-limit 300
python3 experiments/run_gpu_batch_ablation.py --datasets "Yacht hydrodynamics" --seeds 0 1 --gpu-batch-sizes 1 8 32 128 --time-limit 300
```

Inspect the full paper-style matrix before launching it:

```bash
python3 experiments/run_paper_full_suite.py --dry-run --gpu-batch-sizes 32
```

The default fixed-population matrix is 10 datasets x 30 seeds x 3 tree heights x 3 backends, with a 1000 second time budget per run. The script refuses to start long matrices unless `--confirm-long-run` is supplied:

```bash
python3 experiments/run_paper_full_suite.py --gpu-batch-sizes 32 --resume --confirm-long-run
```

IMS settings can be run separately with:

```bash
python3 experiments/run_paper_full_suite.py --include-ims --ims-g-values 4 6 8 --gpu-batch-sizes 32 --resume --confirm-long-run
```

Analyze results:

```bash
python3 experiments/analyze_results.py --input results/final_results.csv
python3 experiments/stats.py --input results/final_results.csv
```

## Comparison Policy

The primary valid comparison is same machine, same codebase:

```text
cpu_original vs gpu_exact_gom vs gpu_batch_gom
```

The paper is treated as an external reference and reproduction target. Do not claim direct superiority over the paper unless datasets, splits, time budgets, solution-size limits, repetitions, statistical tests, and hardware reporting all match.
