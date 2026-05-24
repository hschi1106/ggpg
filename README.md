# gpg
Re-implementation of GP-GOMEA (Python scikit-learn-compatible interface, C++ backend).
This version of the code features only GP-GOMEA and no other algorithms (differently from the [previous repo](https://github.com/marcovirgolin/GP-GOMEA)) and focuses on symbolic regression alone.
Also, this version uses dependencies that are easier and less finicky to install (see [environment.yml](environment.yml)).

This fork adds optional CUDA execution backends for exact and batched GP-GOMEA evaluation.

## Installation
Installation requires [git](https://github.com/git-guides/install-git) and [conda](https://www.anaconda.com/download).
Run the following bash commands from a folder of your choice:
```bash
git clone https://github.com/marcovirgolin/gpg.git
cd gpg
conda env create -f environment.yml
conda activate gpg
make
```

CPU-only CMake build:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=release
cmake --build build/release
```

CUDA CMake build:

```bash
GPG_USE_CUDA=1 cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=release -DGPG_USE_CUDA=ON
cmake --build build/cuda
```

CUDA-enabled Python package:

```bash
make cuda-release
```

## Usage
You can try `gpg` out with the following code snippet (or simply run `try.py` if you like):
```python
import numpy as np
from pygpg.sk import GPGRegressor
from pygpg.complexity import compute_complexity
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

X = np.random.randn(128, 3)*10

def grav_law(X : np.ndarray) -> np.ndarray:
    """Ground-truth function for the gravity law."""
    return 6.67 * X[:,0]*X[:,1]/(np.square(X[:,2])) + np.random.randn(X.shape[0])*0.1 # some noise

y = grav_law(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=RANDOM_SEED)

gpg = GPGRegressor(
  e=50_000,                   # 50,000 evaluations limit
  t=-1,                       # no time limit,
  g=-1,                       # no generation limit,
  d=3,                        # maximum tree depth
  verbose=True,               # print progress
  random_state=RANDOM_SEED,   # for reproducibility
)
gpg.fit(X_train,y_train)

print(
  gpg.model, 
  "(complexity: {})".format(compute_complexity(gpg.model, complexity_metric="node_count")))
print("Train\t\tR2: {}\t\tMSE: {}".format(
  np.round(r2_score(y_train, gpg.predict(X_train)), 3),
  np.round(mean_squared_error(y_train, gpg.predict(X_train)), 3),
))
print("Test\t\tR2: {}\t\tMSE: {}".format(
  np.round(r2_score(y_test, gpg.predict(X_test)), 3),
  np.round(mean_squared_error(y_test, gpg.predict(X_test)), 3),
))
```

## CUDA Backends

This fork keeps `cpu_original` as the upstream GP-GOMEA baseline and adds two CUDA-enabled execution modes for symbolic regression:

- `gpu_exact_gom`: one individual and one GOM move are evaluated at a time on GPU. This preserves the original accept/reject ordering as closely as practical, but it can be launch-overhead limited.
- `gpu_batch_gom`: multiple trial candidates are serialized and evaluated in one CUDA launch. This improves GPU utilization but delays population commits until the current population batch is complete, so it is an approximate acceleration variant.

CUDA kernels never consume `Node*`, `Op*`, or tree pointers. Expressions are serialized on CPU into postfix `GpuToken` programs and copied to fixed-stride GPU buffers.

The upstream division operator `/` is preserved as ordinary division. For paper-comparable experiments that use analytic quotient, select the explicit `aq` primitive:

```bash
-fset +,-,*,aq
```

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
./build/cuda/gpg -train dataset/diabetes_train.csv -ff mse -fset +,-,*,aq -backend gpu_exact_gom -gpu_check_correctness -disable_ims -g 2 -verbose
```

GPU backends currently support `-ff mse` and `-ff mae`. Use `cpu_original` for `ac`. If a GPU backend is requested from a CPU-only build, the program fails with:

```text
CUDA backend requested, but project was built without GPG_USE_CUDA.
```

## Python GPU Interface

The pybind extension exposes the same backend selector. A CUDA-enabled extension reports GPU availability through:

```python
from pygpg.sk import GPGRegressor

print(GPGRegressor.cuda_enabled())
print(GPGRegressor.available_backends())
```

Example GPU fit:

```python
gpg = GPGRegressor(
    backend="gpu_batch_gom",
    gpu_batch_size=128,
    ff="mse",
    fset="+,-,*,aq",
    disable_ims=True,
    g=10,
    pop=1024,
    feat_sel=-1,
)
gpg.fit(X_train, y_train)
```

Run a direct pybind smoke test from a CMake build directory:

```bash
PYTHONPATH=build/cuda-release python3 tests/python_pybind_smoke.py --backend gpu_batch_gom --gpu-batch-size 32
```

## Reproduction Harness

The editable paper-style settings live in:

```text
experiments/paper_reproduction_config.yaml
```

The default paper-style function set is `+,-,*,aq`, where `aq(a,b)=a/sqrt(1+b^2)`.

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

The primary valid comparison is same machine, same codebase:

```text
cpu_original vs gpu_exact_gom vs gpu_batch_gom
```

The paper is treated as an external reference and reproduction target. Do not claim direct superiority over the paper unless datasets, splits, time budgets, solution-size limits, repetitions, statistical tests, and hardware reporting all match.

## Differences w.r.t. previous version
This version has some differences compared to the code in the [previous repo](https://github.com/marcovirgolin/GP-GOMEA).
Here's a list:
- Protected operators are not used here (expressions that evaluate to NaN for some training points are assigned a worst-case fitness `INF`)
- Functions/variables/constants can be sampled with custom probabilities (by default, uniform with binary operators twice as likely as unary operators)
- Tournament selection can be used to speed up convergence within GOM.
- Models returned from the C++ code are simplified and (optionally) fine-tuned in Python
- Elite at multiple levels of complexity (expression size) are stored and returned to Python (a "best one" is selected using the `rci` parameter)
- If the IMS is disabled and the population converges before the budget is exhausted, then a new population is started which includes a random elite from those found before
- A simple feature selection mechanism is included (if desired)
- Models obtained from C++ are converted to `sympy` and can be further processed as such
- The scikit-learn interface includes imputation in case of incomplete data
- The scikit-learn interface includes coefficient fine-tuning with `sympy-torch` and L-BFGS


## Results on SRBench
Running this version on SRBench (`gpg`) leads to expressions that are as compact but more accurate than those of the original `GP-GOMEA`, in much less time!

<img src=pics/srbench.png alt="blackbox_results" width=800px />
<img src=pics/srbench_harmonic.png alt="harmonic_means" width=800px />


## Research
If you use our code for academic purposes, please support our research by citing:
```
@article{virgolin2021improving,
  title={Improving model-based genetic programming for symbolic regression of small expressions},
  author={Virgolin, Marco and Alderliesten, Tanja and Witteveen, Cees and Bosman, Peter A. N.},
  journal={Evolutionary Computation},
  volume={29},
  number={2},
  pages={211--237},
  year={2021},
  publisher={MIT Press}
}
```

## Branches
- `swig` and `pybind` are the same, with the exception that the first uses SWIG and the second uses pybind to realize the python interface. `pybind` is now default and, probably, `swig` will no longer be supported/updated.
- `vector_repr` represents an expression as a vector of strings instead of a tree of nodes. This version may be slightly faster (matters only when the number of observations in the data set is relatively small). However it needs to be [fixed](https://github.com/marcovirgolin/gpg/issues/10).
