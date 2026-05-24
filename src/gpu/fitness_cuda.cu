#define EIGEN_NO_CUDA

#include "fitness_cuda.hpp"
#include "fitness_cuda_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr double PENALTY = 1e12;
constexpr int STACK_CAP = 128;

__device__ __forceinline__ bool finite_dev(double x) {
  return isfinite(x);
}

__device__ double atomic_add_double(double * address, double val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  unsigned long long int * address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(
      address_as_ull,
      assumed,
      __double_as_longlong(val + __longlong_as_double(assumed))
    );
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

cudaStream_t eval_stream(GpuEvalContext & ctx) {
  return reinterpret_cast<cudaStream_t>(ctx.cuda_stream);
}

void ensure_stream(GpuEvalContext & ctx) {
  if (!ctx.cuda_stream) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ctx.cuda_stream = stream;
  }
}

void ensure_single_host_buffers(GpuEvalContext & ctx) {
  if (!ctx.h_single_length) {
    CUDA_CHECK(cudaMallocHost((void **) &ctx.h_single_length, sizeof(int)));
    *ctx.h_single_length = 0;
  }
  if (!ctx.h_single_sum) {
    CUDA_CHECK(cudaMallocHost((void **) &ctx.h_single_sum, sizeof(double)));
    *ctx.h_single_sum = 0.0;
  }
}

int sample_block_count(int N) {
  constexpr int threads = 256;
  int blocks_y = (N + threads - 1) / threads;
  return std::max(1, std::min(blocks_y, 64));
}

__device__ double eval_program_single_sample(
  const GpuToken * prog,
  int len,
  const double * x,
  int D
) {
  double stack[STACK_CAP];
  int sp = 0;

  for (int i = 0; i < len; ++i) {
    GpuToken tok = prog[i];
    GpuOpCode opcode = static_cast<GpuOpCode>(tok.opcode);

    if (opcode == GpuOpCode::VAR) {
      if (tok.var_index < 0 || tok.var_index >= D || sp >= STACK_CAP) {
        return PENALTY;
      }
      stack[sp++] = x[tok.var_index];
      continue;
    }

    if (opcode == GpuOpCode::CONST) {
      if (sp >= STACK_CAP) {
        return PENALTY;
      }
      stack[sp++] = tok.value;
      continue;
    }

    if (
      opcode == GpuOpCode::ADD ||
      opcode == GpuOpCode::SUB ||
      opcode == GpuOpCode::MUL ||
      opcode == GpuOpCode::DIV ||
      opcode == GpuOpCode::AQ
    ) {
      if (sp < 2) {
        return PENALTY;
      }
      double b = stack[--sp];
      double a = stack[--sp];
      double r = 0.0;

      if (opcode == GpuOpCode::ADD) {
        r = a + b;
      } else if (opcode == GpuOpCode::SUB) {
        r = a - b;
      } else if (opcode == GpuOpCode::MUL) {
        r = a * b;
      } else if (opcode == GpuOpCode::DIV) {
        if (b == 0.0) {
          return PENALTY;
        }
        r = a / b;
      } else {
        r = a / sqrt(1.0 + b * b);
      }

      if (!finite_dev(r) || sp >= STACK_CAP) {
        return PENALTY;
      }
      stack[sp++] = r;
      continue;
    }

    if (
      opcode == GpuOpCode::SIN ||
      opcode == GpuOpCode::COS ||
      opcode == GpuOpCode::LOG ||
      opcode == GpuOpCode::SQRT ||
      opcode == GpuOpCode::NEG ||
      opcode == GpuOpCode::INV ||
      opcode == GpuOpCode::SQUARE ||
      opcode == GpuOpCode::CUBE
    ) {
      if (sp < 1) {
        return PENALTY;
      }
      double a = stack[--sp];
      double r = 0.0;

      if (opcode == GpuOpCode::SIN) {
        r = sin(a);
      } else if (opcode == GpuOpCode::COS) {
        r = cos(a);
      } else if (opcode == GpuOpCode::LOG) {
        return PENALTY;
      } else if (opcode == GpuOpCode::SQRT) {
        return PENALTY;
      } else if (opcode == GpuOpCode::NEG) {
        r = -a;
      } else if (opcode == GpuOpCode::INV) {
        if (a == 0.0) {
          return PENALTY;
        }
        r = 1.0 / a;
      } else if (opcode == GpuOpCode::SQUARE) {
        r = a * a;
      } else {
        r = a * a * a;
      }

      if (!finite_dev(r) || sp >= STACK_CAP) {
        return PENALTY;
      }
      stack[sp++] = r;
      continue;
    }

    return PENALTY;
  }

  if (sp != 1 || !finite_dev(stack[0])) {
    return PENALTY;
  }
  return stack[0];
}

__global__ void fitness_kernel_batch(
  const GpuToken * programs,
  const int * lengths,
  int max_program_len,
  const double * X,
  const double * y,
  int N,
  int D,
  int batch_size,
  int fitness_kind,
  double * sums_out
) {
  int p = (int) blockIdx.x;
  if (p >= batch_size) {
    return;
  }

  __shared__ double sh[256];
  int tid = (int) threadIdx.x;
  int blocks_y = (int) gridDim.y;
  int by = (int) blockIdx.y;
  const GpuToken * prog = programs + (size_t) p * (size_t) max_program_len;
  int len = lengths[p];

  double local = 0.0;
  for (int s = by * (int) blockDim.x + tid; s < N; s += (int) blockDim.x * blocks_y) {
    const double * row = X + (size_t) s * (size_t) D;
    double y_hat = eval_program_single_sample(prog, len, row, D);
    double diff = y_hat - y[s];
    double loss = fitness_kind == static_cast<int>(GpuFitnessKind::MAE) ? fabs(diff) : diff * diff;
    if (!finite_dev(loss)) {
      loss = PENALTY;
    }
    local += loss;
    if (!finite_dev(local)) {
      local = PENALTY;
      break;
    }
  }

  sh[tid] = local;
  __syncthreads();

  for (int off = (int) blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) {
      sh[tid] += sh[tid + off];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomic_add_double(&sums_out[p], sh[0]);
  }
}

void copy_data_to_device(GpuEvalContext & ctx, const myeig::Mat & X, const myeig::Vec & y) {
  if (X.rows() != y.rows()) {
    throw std::runtime_error("GPU eval data has mismatched X/y row counts");
  }

  if (ctx.d_X) {
    CUDA_CHECK(cudaFree(ctx.d_X));
    ctx.d_X = nullptr;
  }
  if (ctx.d_y) {
    CUDA_CHECK(cudaFree(ctx.d_y));
    ctx.d_y = nullptr;
  }

  ctx.N = X.rows();
  ctx.D = X.cols();
  ctx.host_X = &X;
  ctx.host_y = &y;

  std::vector<double> hX((size_t) ctx.N * (size_t) ctx.D);
  std::vector<double> hy((size_t) ctx.N);
  for (int i = 0; i < ctx.N; ++i) {
    for (int j = 0; j < ctx.D; ++j) {
      hX[(size_t) i * (size_t) ctx.D + (size_t) j] = (double) X(i, j);
    }
    hy[(size_t) i] = (double) y(i);
  }

  CUDA_CHECK(cudaMalloc(&ctx.d_X, hX.size() * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&ctx.d_y, hy.size() * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(ctx.d_X, hX.data(), hX.size() * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx.d_y, hy.data(), hy.size() * sizeof(double), cudaMemcpyHostToDevice));
}

} // namespace

void gpu_eval_init(
  GpuEvalContext & ctx,
  const myeig::Mat & X,
  const myeig::Vec & y,
  int initial_max_program_len,
  int initial_batch_cap
) {
  ctx = GpuEvalContext{};
  copy_data_to_device(ctx, X, y);
  ctx.initialized = true;
  gpu_eval_ensure_capacity(ctx, std::max(1, initial_batch_cap), std::max(1, initial_max_program_len));
}

void gpu_eval_destroy(GpuEvalContext & ctx) {
  if (ctx.d_X) CUDA_CHECK(cudaFree(ctx.d_X));
  if (ctx.d_y) CUDA_CHECK(cudaFree(ctx.d_y));
  if (ctx.d_programs) CUDA_CHECK(cudaFree(ctx.d_programs));
  if (ctx.d_lengths) CUDA_CHECK(cudaFree(ctx.d_lengths));
  if (ctx.d_sums) CUDA_CHECK(cudaFree(ctx.d_sums));
  if (ctx.h_single_length) CUDA_CHECK(cudaFreeHost(ctx.h_single_length));
  if (ctx.h_single_sum) CUDA_CHECK(cudaFreeHost(ctx.h_single_sum));
  if (ctx.cuda_stream) CUDA_CHECK(cudaStreamDestroy(eval_stream(ctx)));
  ctx = GpuEvalContext{};
}

void gpu_eval_set_data(
  GpuEvalContext & ctx,
  const myeig::Mat & X,
  const myeig::Vec & y
) {
  if (!ctx.initialized) {
    gpu_eval_init(ctx, X, y, 64, 1);
    return;
  }
  copy_data_to_device(ctx, X, y);
}

void gpu_eval_ensure_capacity(
  GpuEvalContext & ctx,
  int batch_size,
  int max_program_len
) {
  if (batch_size <= 0 || max_program_len <= 0) {
    return;
  }

  int new_batch_cap = std::max(1, ctx.batch_cap);
  while (new_batch_cap < batch_size) {
    new_batch_cap <<= 1;
  }

  int new_max_len = std::max(1, ctx.max_program_len);
  while (new_max_len < max_program_len) {
    new_max_len <<= 1;
  }

  if (ctx.d_programs && ctx.d_lengths && ctx.d_sums &&
      ctx.batch_cap >= batch_size && ctx.max_program_len >= max_program_len) {
    ensure_single_host_buffers(ctx);
    return;
  }

  if (ctx.d_programs) CUDA_CHECK(cudaFree(ctx.d_programs));
  if (ctx.d_lengths) CUDA_CHECK(cudaFree(ctx.d_lengths));
  if (ctx.d_sums) CUDA_CHECK(cudaFree(ctx.d_sums));

  ctx.batch_cap = new_batch_cap;
  ctx.max_program_len = new_max_len;
  CUDA_CHECK(cudaMalloc(&ctx.d_programs, (size_t) ctx.batch_cap * (size_t) ctx.max_program_len * sizeof(GpuToken)));
  CUDA_CHECK(cudaMalloc(&ctx.d_lengths, (size_t) ctx.batch_cap * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&ctx.d_sums, (size_t) ctx.batch_cap * sizeof(double)));
  ensure_single_host_buffers(ctx);
}

void evaluate_fitness_gpu_batch(
  GpuEvalContext & ctx,
  const GpuToken * flat_programs,
  const int * lengths,
  int batch_size,
  int max_program_len,
  double * out_fitness
) {
  if (batch_size <= 0) {
    return;
  }
  if (!ctx.initialized || ctx.N <= 0 || ctx.D <= 0) {
    throw std::runtime_error("GPU eval context is not initialized");
  }

  gpu_eval_ensure_capacity(ctx, batch_size, max_program_len);

  ensure_stream(ctx);
  cudaStream_t stream = eval_stream(ctx);
  const size_t program_bytes = (size_t) batch_size * (size_t) max_program_len * sizeof(GpuToken);
  CUDA_CHECK(cudaMemcpyAsync(ctx.d_programs, flat_programs, program_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(ctx.d_lengths, lengths, (size_t) batch_size * sizeof(int), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemsetAsync(ctx.d_sums, 0, (size_t) batch_size * sizeof(double), stream));

  const int threads = 256;
  int blocks_y = sample_block_count(ctx.N);
  dim3 block(threads);
  dim3 grid((unsigned) batch_size, (unsigned) blocks_y);

  fitness_kernel_batch<<<grid, block, 0, stream>>>(
    ctx.d_programs,
    ctx.d_lengths,
    max_program_len,
    ctx.d_X,
    ctx.d_y,
    ctx.N,
    ctx.D,
    batch_size,
    static_cast<int>(ctx.fitness_kind),
    ctx.d_sums
  );
  CUDA_CHECK(cudaGetLastError());

  ctx.host_sums.resize((size_t) batch_size);
  CUDA_CHECK(cudaMemcpyAsync(ctx.host_sums.data(), ctx.d_sums, (size_t) batch_size * sizeof(double), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int i = 0; i < batch_size; ++i) {
    double value = ctx.host_sums[(size_t) i] / (double) ctx.N;
    if (!std::isfinite(value) || value < 0.0) {
      value = PENALTY;
    }
    out_fitness[i] = value;
  }
}

double evaluate_fitness_gpu_single(
  GpuEvalContext & ctx,
  const GpuToken * tokens,
  int length
) {
  if (length <= 0) {
    return PENALTY;
  }
  if (!ctx.initialized || ctx.N <= 0 || ctx.D <= 0) {
    throw std::runtime_error("GPU eval context is not initialized");
  }

  gpu_eval_ensure_capacity(ctx, 1, length);
  ensure_single_host_buffers(ctx);

  *ctx.h_single_length = length;
  *ctx.h_single_sum = 0.0;

  CUDA_CHECK(cudaMemcpy(
    ctx.d_programs,
    tokens,
    (size_t) length * sizeof(GpuToken),
    cudaMemcpyHostToDevice
  ));
  CUDA_CHECK(cudaMemcpy(
    ctx.d_lengths,
    ctx.h_single_length,
    sizeof(int),
    cudaMemcpyHostToDevice
  ));
  CUDA_CHECK(cudaMemset(ctx.d_sums, 0, sizeof(double)));

  const int threads = 256;
  dim3 block(threads);
  dim3 grid(1, (unsigned) sample_block_count(ctx.N));
  fitness_kernel_batch<<<grid, block>>>(
    ctx.d_programs,
    ctx.d_lengths,
    ctx.max_program_len,
    ctx.d_X,
    ctx.d_y,
    ctx.N,
    ctx.D,
    1,
    static_cast<int>(ctx.fitness_kind),
    ctx.d_sums
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(
    ctx.h_single_sum,
    ctx.d_sums,
    sizeof(double),
    cudaMemcpyDeviceToHost
  ));

  double value = *ctx.h_single_sum / (double) ctx.N;
  if (!std::isfinite(value) || value < 0.0) {
    value = PENALTY;
  }
  return value;
}
