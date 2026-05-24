# PLAN.md — GPU-Accelerated GP-GOMEA Based on `marcovirgolin/gpg`

## 0. Project Objective

Modify `marcovirgolin/gpg` as the base codebase and add GPU-accelerated GP-GOMEA execution modes for symbolic regression.

The final deliverable must support controlled comparison against the paper:

```text
Improving Model-based Genetic Programming for Symbolic Regression of Small Expressions
```

The paper PDF will be placed inside the repository by the user. Treat it as the authoritative experimental reference for:

- algorithm settings
- datasets
- train/validation/test protocol
- expression-size limits
- GP-GOMEA variants
- reported benchmark tables

The implementation must produce reproducible experiment outputs that can be compared against the paper's reported results.

---

## 1. Required Repositories and References

### 1.1 Base implementation

Use this repository as the base codebase:

```text
https://github.com/marcovirgolin/gpg
```

Relevant files to inspect and modify:

```text
src/variation.hpp
src/fitness.hpp
src/node.hpp
src/operator.hpp
src/fos.hpp
src/evolution.hpp
src/ims.hpp
src/globals.hpp
src/pypkg/pygpg/sk.py
```

Important existing structure:

- `Node` represents symbolic expression trees.
- Each `Node` owns an `Op*`.
- `Node::get_output(Mat& X)` recursively evaluates expressions.
- `Fitness::get_fitness(Node*)` computes MSE / MAE / other fitness over the current training batch.
- `efficient_gom(...)` in `variation.hpp` is the main GOM operator.
- Linkage learning and FOS construction are in `fos.hpp`.
- IMS-related logic is in `ims.hpp`.

### 1.2 GPU implementation reference

Use this repository as the reference for GPU parallelization ideas:

```text
https://github.com/hschi1106/Parallel_Programming_Final_Project/tree/main
```

Relevant implementations:

```text
parallel/src/fitness_cuda.cu
parallel/include/fitness_cuda.hpp
parallel/batch/src/fitness_cuda.cu
parallel/batch/include/fitness_cuda.hpp
parallel/batch/src/gomea.cpp
```

Reference ideas to reuse:

1. Single-program GPU evaluation
   - One expression/program is evaluated over many samples.
   - CUDA threads parallelize over sample dimension.

2. Batched GPU evaluation
   - Multiple expressions are flattened into one program buffer.
   - CUDA grid uses:
     - `grid.x = program index`
     - `grid.y = sample slice`
   - One kernel launch evaluates many candidate expressions.

3. Batched GOMEA approximation
   - Process multiple population individuals together.
   - Generate one candidate per individual per FOS step.
   - Batch-evaluate those candidates.
   - Accept/reject locally.
   - Commit the batch after finishing its local GOM process.

---

## 2. Required Execution Modes

Implement three execution modes.

```text
cpu_original
gpu_exact_gom
gpu_batch_gom
```

### 2.1 `cpu_original`

Preserve original `gpg` behavior.

Purpose:

- baseline correctness
- baseline performance
- baseline for paper-comparison experiments

This mode must not depend on CUDA.

### 2.2 `gpu_exact_gom`

Use GPU evaluation but preserve original GOM semantics as much as practical.

Behavior:

- GOM still processes one individual at a time.
- Each FOS move is evaluated immediately.
- Accept/reject happens immediately.
- The only intended change is replacing CPU fitness evaluation with GPU fitness evaluation.

Expected result:

- Semantically close to CPU original.
- May not be much faster because every meaningful GOM move still requires immediate GPU evaluation.

### 2.3 `gpu_batch_gom`

Use batched GPU evaluation inspired by the user's previous repo.

Behavior:

- Process a batch of population individuals together.
- For each FOS step, each individual produces one trial candidate.
- All candidates in the batch are serialized and evaluated together on GPU.
- Accept/reject happens per individual.
- Population updates are committed after the population batch is finished.

Important:

This mode is an approximate acceleration variant. It is not exactly equivalent to original GP-GOMEA because population updates are delayed inside each batch.

The experiment must explicitly report this approximation.

---

## 3. Critical Design Constraint: Do Not Send `Node*` / `Op*` to GPU

GPU kernels must not consume `Node*`, `Op*`, or any pointer-heavy tree structure.

Reasons:

- `Node*` / `Op*` use dynamic allocation.
- Operators may rely on virtual dispatch.
- Recursive tree traversal is inefficient and fragile on GPU.
- Device-side ownership would be error-prone.

Instead, every expression must be serialized on CPU into a GPU-friendly token sequence.

---

## 4. GPU-Friendly Token Representation

### 4.1 New files

Create:

```text
src/gpu/gpu_token.hpp
src/gpu/gpu_token.cpp
src/gpu/expression_serializer.hpp
src/gpu/expression_serializer.cpp
```

### 4.2 Token format

Recommended minimal token:

```cpp
enum class GpuOpCode : int {
    ADD = 1,
    SUB = 2,
    MUL = 3,
    DIV = 4,
    AQ = 5,
    SIN = 6,
    COS = 7,
    LOG = 9,
    SQRT = 10,
    VAR = 100,
    CONST = 101,
    INVALID = -1
};

struct GpuToken {
    int opcode;
    int var_index;
    double value;
};
```

Notes:

- Use `double` first for correctness.
- Later add `float` as an optional performance mode.
- `var_index` is only used by variable tokens.
- `value` is only used by constant tokens.
- `AQ` is the paper analytic quotient primitive, \(a / \sqrt{1 + b^2}\). It is intentionally distinct from upstream `/`.

### 4.3 Program layout for GPU

Use fixed-stride padded layout:

```cpp
std::vector<GpuToken> flat_programs(batch_size * max_program_len);
std::vector<int> program_lengths(batch_size);
```

Token `t` of program `b` is:

```cpp
flat_programs[b * max_program_len + t]
```

This layout is simple and matches the batch-evaluation pattern from the user's previous CUDA repo.

---

## 5. Expression Serialization

### 5.1 Required API

Implement:

```cpp
struct SerializedProgram {
    std::vector<GpuToken> tokens;
};

SerializedProgram serialize_active_tree_to_postfix(Node* root);

void serialize_active_tree_to_postfix_into(
    Node* root,
    std::vector<GpuToken>& out
);
```

### 5.2 Active-expression traversal

Serialize only semantically active children.

Pseudo-code:

```cpp
void serialize_node_postfix(Node* n, std::vector<GpuToken>& out) {
    int arity = n->op->arity();

    for (int i = 0; i < arity; ++i) {
        serialize_node_postfix(n->children[i], out);
    }

    out.push_back(op_to_gpu_token(n->op));
}
```

This matches the CPU behavior of `Node::get_output`, which only evaluates the first `arity` children.

Do not serialize introns.

### 5.3 Operator mapping

Implement:

```cpp
GpuToken op_to_gpu_token(Op* op);
```

Inspect `src/operator.hpp` and map all operators used in paper-style experiments.

Minimum required operators:

```text
+
-
*
division semantics matching the upstream CPU implementation
variables
constants / ERC
```

Likely optional operators depending on repository defaults:

```text
sin
cos
exp
log
sqrt
```

Important:

CPU and GPU operator semantics must match. Do not silently change division, overflow, NaN, or constant behavior.

### 5.4 Constants and ERC

Constants must be represented as `GpuToken{CONST, -1, value}`.

Do not treat all constants as one shared symbol for evaluation. Constant binning is only for linkage learning, not for expression evaluation.

### 5.5 Program length

Add utilities:

```cpp
int get_active_program_len(Node* root);
```

GPU evaluator must handle variable-length programs.

If active length exceeds capacity:

- grow GPU buffers, or
- fail loudly in debug mode.

Prefer dynamic growth.

---

## 6. CUDA Fitness Evaluation Layer

### 6.1 New files

Create:

```text
src/gpu/gpu_eval_context.hpp
src/gpu/gpu_eval_context.cpp
src/gpu/fitness_cuda.hpp
src/gpu/fitness_cuda.cu
src/gpu/fitness_cuda_kernels.cuh
```

### 6.2 Context

Suggested context:

```cpp
struct GpuEvalContext {
    bool initialized = false;

    int N = 0;
    int D = 0;
    int max_program_len = 0;
    int batch_cap = 0;

    double* d_X = nullptr;
    double* d_y = nullptr;

    GpuToken* d_programs = nullptr;
    int* d_lengths = nullptr;
    double* d_sums = nullptr;

    const Mat* host_X = nullptr;
    const Vec* host_y = nullptr;
};
```

### 6.3 Host API

Implement:

```cpp
void gpu_eval_init(
    GpuEvalContext& ctx,
    const Mat& X,
    const Vec& y,
    int initial_max_program_len,
    int initial_batch_cap
);

void gpu_eval_destroy(GpuEvalContext& ctx);

void gpu_eval_set_data(
    GpuEvalContext& ctx,
    const Mat& X,
    const Vec& y
);

void gpu_eval_ensure_capacity(
    GpuEvalContext& ctx,
    int batch_size,
    int max_program_len
);

double evaluate_fitness_gpu_single(
    GpuEvalContext& ctx,
    const GpuToken* tokens,
    int length
);

void evaluate_fitness_gpu_batch(
    GpuEvalContext& ctx,
    const GpuToken* flat_programs,
    const int* lengths,
    int batch_size,
    int max_program_len,
    double* out_fitness
);
```

### 6.4 Device-side program evaluator

Implement stack-machine evaluation:

```cpp
__device__ double eval_program_single_sample(
    const GpuToken* prog,
    int len,
    const double* x,
    int D
);
```

Use fixed-size local stack:

```cpp
double stack[128];
int sp = 0;
```

Handle:

- stack underflow
- stack overflow
- invalid operator
- invalid variable index
- non-finite result
- final stack size not equal to 1

Return penalty:

```cpp
const double PENALTY = 1e12;
```

### 6.5 Batch kernel

Implement:

```cpp
__global__ void fitness_kernel_batch(
    const GpuToken* programs,
    const int* lengths,
    int max_program_len,
    const double* X,
    const double* y,
    int N,
    int D,
    int batch_size,
    double* sums_out
);
```

Kernel design:

```cpp
int p = blockIdx.x;  // program index
int by = blockIdx.y; // sample slice
```

For each program `p`, threads iterate over samples assigned to `(p, by)`.

Use block reduction and `atomicAdd(&sums_out[p], partial_sum)`.

This follows the user's previous `parallel/batch/src/fitness_cuda.cu` design.

---

## 7. Fitness Integration

### 7.1 GPU single evaluation

For `gpu_exact_gom`, replace CPU fitness calls inside GOM with:

```cpp
SerializedProgram sp = serialize_active_tree_to_postfix(offspring);

double new_fitness = evaluate_fitness_gpu_single(
    gpu_ctx,
    sp.tokens.data(),
    static_cast<int>(sp.tokens.size())
);
```

Set:

```cpp
offspring->fitness = new_fitness;
```

### 7.2 GPU batch evaluation

For `gpu_batch_gom`, gather many candidate `Node*` expressions.

Process:

1. serialize candidates into `flat_programs`
2. fill `program_lengths`
3. call `evaluate_fitness_gpu_batch`
4. assign returned fitness to candidate state
5. accept/reject on CPU

### 7.3 Correctness checking

Add optional debug mode:

```cpp
g::gpu_check_correctness
```

If enabled, randomly sample some evaluated expressions and compute both CPU and GPU fitness:

```cpp
abs(cpu_fitness - gpu_fitness) < tolerance
```

Suggested tolerances:

```text
double mode: 1e-6 to 1e-8
float mode: 1e-4 to 1e-5
```

---

## 8. GOM Implementations

## 8.1 CPU original

Refactor existing GOM into:

```cpp
Node* efficient_gom_cpu_original(
    Node* parent,
    std::vector<Node*>& population,
    std::vector<std::vector<int>>& fos
);
```

This should preserve existing logic.

## 8.2 Exact GPU GOM

Create:

```cpp
Node* efficient_gom_gpu_exact(
    Node* parent,
    std::vector<Node*>& population,
    std::vector<std::vector<int>>& fos,
    GpuEvalContext& gpu_ctx
);
```

Behavior:

```text
for each parent:
    clone parent into offspring
    shuffle FOS
    for each FOS subset:
        apply donor subset
        optional coefficient mutation
        if meaningful change:
            serialize offspring
            evaluate on GPU immediately
            accept/reject immediately
        else:
            skip evaluation
    return offspring
```

Important:

- Use the same accept/reject rule as CPU original.
- Preserve neutral-move behavior if CPU original accepts equal fitness.
- Preserve coefficient mutation behavior if enabled.
- Preserve forced-improvement / tournament behavior if original uses it.

## 8.3 Batch GPU GOM

Create population-level function:

```cpp
void gomea_generation_gpu_batch(
    std::vector<Node*>& population,
    std::vector<std::vector<int>>& fos,
    GpuEvalContext& gpu_ctx,
    int pop_batch_size
);
```

This function is called once per generation instead of calling `efficient_gom(...)` once per individual.

### 8.3.1 Batch GOM pseudo-code

```cpp
void gomea_generation_gpu_batch(...) {
    order = shuffled population indices

    for each population batch:
        B = min(pop_batch_size, remaining individuals)

        create local candidates:
            cand[i] = population[idx[i]]->clone()
            cand_fitness[i] = population[idx[i]]->fitness
            fos_order[i] = shuffled FOS order

        for step in 0..fos.size()-1:
            candidate_nodes.clear()
            candidate_map.clear()
            backups.clear()

            for i in 0..B-1:
                subset = fos[fos_order[i][step]]
                donor = random donor from global population

                apply donor subset to cand[i]
                record backup for rollback

                if meaningful change:
                    candidate_nodes.push_back(cand[i])
                    candidate_map.push_back(i)
                else:
                    rollback immediately or treat fitness as unchanged

            serialize candidate_nodes into flat token buffer

            gpu_batch_evaluate(...)

            for each evaluated candidate:
                i = candidate_map[k]
                if accept(new_fitness[k], cand_fitness[i]):
                    cand_fitness[i] = new_fitness[k]
                    cand[i]->fitness = new_fitness[k]
                else:
                    rollback cand[i] using backup
                    cand[i]->fitness = cand_fitness[i]

        commit all cand[i] back to population
}
```

### 8.3.2 Approximation statement

Document clearly in comments:

```text
This mode delays population commits until the current population batch is complete.
Therefore donors inside the same batch do not see improvements made earlier in that batch.
This is an approximate batched GOM variant.
```

### 8.3.3 `gpu_batch_size`

Expose as a user parameter.

Suggested values:

```text
1, 8, 16, 32, 64, 128, 256
```

`gpu_batch_size = 1` should behave close to exact GOM except for minor implementation and RNG differences.

Expected speedup only starts when `gpu_batch_size > 1`.

---

## 9. Integration with Evolution Loop

Find the generational loop in `evolution.hpp` or equivalent.

Existing likely pattern:

```cpp
fos = fos_builder.build_linkage_tree(population);

for each individual:
    offspring = efficient_gom(parent, population, fos);

population = offspring;
```

Change to:

```cpp
fos = fos_builder.build_linkage_tree(population);

if (g::execution_backend == GPU_BATCH_GOM) {
    gomea_generation_gpu_batch(population, fos, *g::gpu_ctx, g::gpu_batch_size);
} else {
    for each individual:
        if (g::execution_backend == CPU_ORIGINAL) {
            offspring = efficient_gom_cpu_original(parent, population, fos);
        } else if (g::execution_backend == GPU_EXACT_GOM) {
            offspring = efficient_gom_gpu_exact(parent, population, fos, *g::gpu_ctx);
        }
    population = offspring;
}
```

Be careful:

- preserve memory ownership
- delete old `Node*` correctly
- do not leak cloned candidates
- maintain existing population replacement behavior

---

## 10. Paper-Aware Experimental Reproduction

The repo will contain the paper PDF. Add scripts and metadata so experiments can be compared to the paper.

### 10.1 Paper location

Assume the paper is placed at one of:

```text
paper/Improving_Model_based_GP_for_SR.pdf
docs/paper.pdf
paper/paper.pdf
```

Codex should not depend on an exact filename. Add a config variable:

```text
PAPER_PATH
```

or search for a PDF under:

```text
paper/
docs/
```

### 10.2 Experiment config derived from paper

Create:

```text
experiments/paper_reproduction_config.yaml
```

Include the paper-style settings:

```yaml
function_set:
  - "+"
  - "-"
  - "*"
  - aq

terminal_set:
  include_features: true
  include_erc: true

erc_bounds: "min_x_to_max_x"

train_validation_test_split:
  train: 0.50
  validation: 0.25
  test: 0.25

repetitions: 30

tree_heights:
  - 3
  - 4
  - 5

max_nodes:
  h3: 15
  h4: 31
  h5: 63

fixed_population_size: 1000

ims_g_values:
  - 4
  - 6
  - 8

time_limit_seconds: 1000

datasets:
  - Airfoil
  - Boston housing
  - Concrete compressive strength
  - Dow chemical
  - Energy cooling
  - Energy heating
  - Tower
  - Wine red
  - Wine white
  - Yacht hydrodynamics
```

This config must be editable.

### 10.3 Dataset handling

Add scripts:

```text
experiments/download_datasets.py
experiments/prepare_paper_datasets.py
```

The paper uses real-world regression datasets. The implementation should either:

1. download them from public sources, or
2. load them from a local `data/` directory if already provided.

Use deterministic train/validation/test splits by seed.

Directory layout:

```text
data/raw/
data/processed/
```

Processed dataset format:

```text
data/processed/{dataset_name}/X_train.csv
data/processed/{dataset_name}/y_train.csv
data/processed/{dataset_name}/X_val.csv
data/processed/{dataset_name}/y_val.csv
data/processed/{dataset_name}/X_test.csv
data/processed/{dataset_name}/y_test.csv
```

### 10.4 Paper baseline reproduction

Add:

```text
experiments/run_paper_cpu_baseline.py
```

It should run `cpu_original` with paper-like settings.

Outputs:

```text
results/paper_reproduction/cpu_original/*.csv
```

At minimum log:

```text
dataset
seed
tree_height
max_nodes
population_size
ims_g
backend
elapsed_sec
train_nmse
validation_nmse
test_nmse
best_expression
num_evaluations
```

### 10.5 GPU experiment scripts

Add:

```text
experiments/run_gpu_exact.py
experiments/run_gpu_batch_ablation.py
```

`run_gpu_batch_ablation.py` should run:

```text
gpu_batch_size = 1, 8, 16, 32, 64, 128
```

for selected datasets first.

### 10.6 Result comparison tables

Add:

```text
experiments/analyze_results.py
```

Generate tables:

```text
results/tables/table_cpu_vs_gpu_exact.csv
results/tables/table_batch_size_ablation.csv
results/tables/table_paper_style_nmse.csv
```

Generate plots:

```text
results/figures/validation_nmse_vs_time.png
results/figures/speedup_vs_batch_size.png
results/figures/test_nmse_vs_batch_size.png
results/figures/evals_per_sec_vs_batch_size.png
```

### 10.7 Comparison policy

Do not claim direct superiority over the paper unless:

- same datasets
- same splits
- same time budget
- same solution-size limits
- same number of repetitions
- same statistical tests
- hardware differences are reported

The primary valid comparison is:

```text
same machine, same codebase:
cpu_original vs gpu_exact_gom vs gpu_batch_gom
```

The paper results should be treated as:

```text
external reference / reproduction target
```

---

## 11. Statistical Evaluation

Implement the statistical tests used in the paper as closely as feasible.

Add:

```text
experiments/stats.py
```

Required:

- median NMSE over 30 runs
- Wilcoxon signed-rank test for paired runs
- Bonferroni correction when multiple comparisons are made
- optional Friedman test for comparing multiple algorithms over datasets

Output:

```text
results/stats/significance_summary.csv
```

Columns:

```text
dataset
metric
method_a
method_b
median_a
median_b
p_value
significant
winner
```

Metrics:

```text
train_nmse
validation_nmse
test_nmse
wall_clock_time
evals_per_sec
```

---

## 12. Logging Requirements

Every experiment must write structured CSV logs.

### 12.1 Per-generation log

```text
results/logs/{run_id}_generation.csv
```

Columns:

```text
run_id
dataset
backend
seed
generation
elapsed_sec
tree_height
max_nodes
population_size
ims_g
gpu_batch_size
best_train_fitness
best_validation_fitness
best_train_nmse
best_validation_nmse
num_evaluations
num_gpu_evaluations
num_cpu_evaluations
num_accepted_moves
num_rejected_moves
num_meaningful_candidates
num_nonmeaningful_candidates
avg_actual_gpu_batch_size
```

### 12.2 Final result log

```text
results/final_results.csv
```

Columns:

```text
run_id
dataset
backend
seed
tree_height
max_nodes
population_size
ims_g
gpu_batch_size
elapsed_sec
train_nmse
validation_nmse
test_nmse
num_evaluations
num_gpu_evaluations
evals_per_sec
best_expression
```

---

## 13. Build System

Add optional CUDA support.

### 13.1 CPU-only build

Must still work without CUDA.

If user selects GPU backend without CUDA support, fail with a clear error:

```text
CUDA backend requested, but project was built without GPG_USE_CUDA.
```

### 13.2 CUDA build macro

Use:

```cpp
#ifdef GPG_USE_CUDA
```

Suggested environment flag:

```text
GPG_USE_CUDA=1
```

### 13.3 Build integration

Inspect the current package build system.

Possibilities:

- `setup.py`
- `pyproject.toml`
- CMake
- custom Makefile

Add `.cu` compilation only when CUDA is enabled.

---

## 14. Correctness Tests

Add tests under:

```text
tests/
```

### 14.1 Serializer tests

Test simple expressions:

```text
x0
x0 + x1
(x0 + 2.0) * x1
sin(x0)
x0 / x1
```

Check postfix token order.

### 14.2 CPU vs GPU single evaluation

For randomly generated expressions and datasets:

```text
CPU fitness
GPU single fitness
```

Assert close numerical agreement.

### 14.3 GPU batch vs GPU single

For a batch of expressions:

```text
gpu_batch[i] ~= gpu_single(expr_i)
```

### 14.4 `gpu_exact_gom` sanity

Run with tiny population and short generation count.

Check:

- no crash
- fitness improves or stays stable
- expression serialization does not fail

### 14.5 `gpu_batch_gom` sanity

Run:

```text
gpu_batch_size=1
gpu_batch_size=8
```

Check:

- no crash
- final expression is valid
- fitness generally improves

---

## 15. Benchmark Plan

### 15.1 Fast development benchmark

Use this first:

```text
datasets:
  - Yacht hydrodynamics
  - Airfoil
  - Wine white
  - Dow chemical

tree_height: 4
max_nodes: 31
population_size: 1000
time_limit_seconds: 300
seeds: 10
backends:
  - cpu_original
  - gpu_exact_gom
  - gpu_batch_gom
gpu_batch_sizes:
  - 1
  - 8
  - 32
  - 128
```

Purpose:

- find bugs
- understand speedup
- find batch-size sweet spot

### 15.2 Paper-style benchmark

After the core implementation works:

```text
datasets: all 10 paper datasets
tree_height: 3, 4, 5
max_nodes: 15, 31, 63
time_limit_seconds: 1000
repetitions: 30
backends:
  - cpu_original
  - gpu_exact_gom
  - gpu_batch_gom
gpu_batch_sizes:
  - best values from fast benchmark
```

IMS:

- start with fixed population size 1000
- add IMS `g=8` next
- add IMS `g=4,6` only if time permits

---

## 16. Deliverables

Final repository should include:

```text
src/gpu/
experiments/
tests/
paper/ or docs/ containing the paper PDF
data/ instructions or scripts
results/ generated by experiments
PLAN.md
README.md
```

### 16.1 `README.md`

Must explain:

- how to build CPU-only
- how to build with CUDA
- how to use the Python pybind interface
- how to run each backend
- how to reproduce fast benchmark
- how to reproduce paper-style benchmark
- what approximation `gpu_batch_gom` introduces

### 16.2 Result artifacts

At minimum:

```text
results/final_results.csv
results/tables/table_cpu_vs_gpu_exact.csv
results/tables/table_batch_size_ablation.csv
results/figures/speedup_vs_batch_size.png
results/figures/test_nmse_vs_batch_size.png
```

---

## 17. Important Pitfalls

### 17.1 Operator mismatch

The most likely correctness bug is CPU/GPU mismatch in operator semantics.

Pay special attention to:

- division and invalid-value handling
- analytic quotient handling when reproducing paper-style settings
- exponent overflow
- NaN / Inf handling
- constant precision
- variable indexing
- linear scaling
- ERC handling

### 17.2 Batch GOM is approximate

Do not present `gpu_batch_gom` as an exact implementation.

It delays population commits and changes donor visibility inside a batch.

This is intentional and must be measured.

### 17.3 GPU exact may not speed up

`gpu_exact_gom` may be slow because it still launches many small GPU evaluations.

This is expected.

### 17.4 Small datasets may not benefit from GPU

For small datasets, launch overhead may dominate.

Report this honestly.

### 17.5 Raw pointers and memory leaks

The base repo uses raw `Node*` and `Op*`.

When cloning, replacing, or rolling back:

- avoid double-delete
- delete abandoned candidate trees
- preserve old ownership behavior

### 17.6 Randomness

CPU and GPU versions may diverge due to floating point and evaluation ordering.

For exact GPU mode, aim for close trajectories, not necessarily bit-identical final expressions.

For batch GPU mode, divergence is expected.

---

## 18. Minimal Implementation Path

If time is limited, implement in this order:

1. Add backend enum and keep `cpu_original` working.
2. Implement token representation.
3. Implement active-tree postfix serializer.
4. Implement GPU single evaluation.
5. Validate CPU vs GPU fitness.
6. Implement `gpu_exact_gom`.
7. Implement GPU batch evaluation.
8. Implement `gpu_batch_gom`.
9. Add fast benchmark script.
10. Add paper-style config and result tables.

Do not start with full paper reproduction before the GPU backends are stable.

---

## 19. Final Report Framing

Recommended title:

```text
Batched GPU Evaluation for GP-GOMEA in Symbolic Regression
```

Recommended claim:

```text
GP-GOMEA's GOM operator has sequential accept/reject dependencies that limit naive GPU acceleration.
We implement an exact GPU evaluation backend and an approximate batched GOM backend.
The exact backend preserves GOM semantics but may have limited speedup.
The batched backend improves GPU utilization by evaluating multiple candidate expressions per launch, at the cost of delayed population updates.
We compare speed, NMSE, and statistical significance against a CPU baseline and paper-style experimental settings.
```

Do not claim:

```text
gpu_batch_gom is identical to original GP-GOMEA
```

Instead claim:

```text
gpu_batch_gom is a controlled approximate acceleration variant whose speed-quality tradeoff is measured experimentally.
```
