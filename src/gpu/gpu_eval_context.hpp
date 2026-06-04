#ifndef GPU_EVAL_CONTEXT_HPP
#define GPU_EVAL_CONTEXT_HPP

#include "../myeig.hpp"
#include "gpu_token.hpp"

enum class GpuFitnessKind : int {
  MSE = 1,
  MAE = 2
};

struct GpuEvalContext {
  bool initialized = false;

  int N = 0;
  int D = 0;
  int max_program_len = 0;
  int batch_cap = 0;
  GpuFitnessKind fitness_kind = GpuFitnessKind::MSE;

  float * d_X = nullptr;
  float * d_y = nullptr;

  GpuToken * d_programs = nullptr;
  int * d_lengths = nullptr;
  float * d_sums = nullptr;

  void * cuda_stream = nullptr;

  GpuToken * h_programs = nullptr;
  int * h_lengths = nullptr;
  float * h_sums = nullptr;
  size_t h_program_token_cap = 0;
  int h_batch_cap = 0;

  void * single_graph_exec = nullptr;
  int single_graph_max_program_len = 0;
  int single_graph_blocks_y = 0;

  const myeig::Mat * host_X = nullptr;
  const myeig::Vec * host_y = nullptr;
};

#endif
