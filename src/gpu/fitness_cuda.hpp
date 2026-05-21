#ifndef FITNESS_CUDA_HPP
#define FITNESS_CUDA_HPP

#include "gpu_eval_context.hpp"

void gpu_eval_init(
  GpuEvalContext & ctx,
  const myeig::Mat & X,
  const myeig::Vec & y,
  int initial_max_program_len,
  int initial_batch_cap
);

void gpu_eval_destroy(GpuEvalContext & ctx);

void gpu_eval_set_data(
  GpuEvalContext & ctx,
  const myeig::Mat & X,
  const myeig::Vec & y
);

void gpu_eval_ensure_capacity(
  GpuEvalContext & ctx,
  int batch_size,
  int max_program_len
);

double evaluate_fitness_gpu_single(
  GpuEvalContext & ctx,
  const GpuToken * tokens,
  int length
);

void evaluate_fitness_gpu_batch(
  GpuEvalContext & ctx,
  const GpuToken * flat_programs,
  const int * lengths,
  int batch_size,
  int max_program_len,
  double * out_fitness
);

#endif
