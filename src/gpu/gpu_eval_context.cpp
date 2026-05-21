#include "fitness_cuda.hpp"

#ifndef GPG_USE_CUDA

#include <stdexcept>

using namespace std;

void gpu_eval_init(
  GpuEvalContext & ctx,
  const myeig::Mat & X,
  const myeig::Vec & y,
  int initial_max_program_len,
  int initial_batch_cap
) {
  ctx.host_X = &X;
  ctx.host_y = &y;
  ctx.N = X.rows();
  ctx.D = X.cols();
  ctx.max_program_len = initial_max_program_len;
  ctx.batch_cap = initial_batch_cap;
  throw runtime_error("CUDA backend requested, but project was built without GPG_USE_CUDA.");
}

void gpu_eval_destroy(GpuEvalContext & ctx) {
  ctx = GpuEvalContext{};
}

void gpu_eval_set_data(
  GpuEvalContext & ctx,
  const myeig::Mat & X,
  const myeig::Vec & y
) {
  ctx.host_X = &X;
  ctx.host_y = &y;
  ctx.N = X.rows();
  ctx.D = X.cols();
  throw runtime_error("CUDA backend requested, but project was built without GPG_USE_CUDA.");
}

void gpu_eval_ensure_capacity(
  GpuEvalContext &,
  int,
  int
) {
  throw runtime_error("CUDA backend requested, but project was built without GPG_USE_CUDA.");
}

double evaluate_fitness_gpu_single(
  GpuEvalContext &,
  const GpuToken *,
  int
) {
  throw runtime_error("CUDA backend requested, but project was built without GPG_USE_CUDA.");
}

void evaluate_fitness_gpu_batch(
  GpuEvalContext &,
  const GpuToken *,
  const int *,
  int,
  int,
  double *
) {
  throw runtime_error("CUDA backend requested, but project was built without GPG_USE_CUDA.");
}

#endif
