#ifndef GPU_EVAL_CONTEXT_HPP
#define GPU_EVAL_CONTEXT_HPP

#include <vector>

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

  double * d_X = nullptr;
  double * d_y = nullptr;

  GpuToken * d_programs = nullptr;
  int * d_lengths = nullptr;
  double * d_sums = nullptr;

  std::vector<double> host_sums;

  void * cuda_stream = nullptr;

  int * h_single_length = nullptr;
  double * h_single_sum = nullptr;

  const myeig::Mat * host_X = nullptr;
  const myeig::Vec * host_y = nullptr;
};

#endif
