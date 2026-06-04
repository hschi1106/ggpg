#define EIGEN_NO_CUDA

#include "fitness_cuda.hpp"
#include "fitness_cuda_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr double PENALTY = 1e12;
constexpr float PENALTY_F = 1e12f;
constexpr int STACK_CAP = 128;

__device__ __forceinline__ bool finite_dev(float x) {
  return isfinite(x);
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

void destroy_single_graph(GpuEvalContext & ctx) {
  if (ctx.single_graph_exec) {
    CUDA_CHECK(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(ctx.single_graph_exec)));
    ctx.single_graph_exec = nullptr;
  }
  ctx.single_graph_max_program_len = 0;
  ctx.single_graph_blocks_y = 0;
}

void release_host_buffers(GpuEvalContext & ctx) {
  destroy_single_graph(ctx);
  if (ctx.h_programs) CUDA_CHECK(cudaFreeHost(ctx.h_programs));
  if (ctx.h_lengths) CUDA_CHECK(cudaFreeHost(ctx.h_lengths));
  if (ctx.h_sums) CUDA_CHECK(cudaFreeHost(ctx.h_sums));
  ctx.h_programs = nullptr;
  ctx.h_lengths = nullptr;
  ctx.h_sums = nullptr;
  ctx.h_program_token_cap = 0;
  ctx.h_batch_cap = 0;
}

void ensure_host_buffers(GpuEvalContext & ctx, int batch_size, int max_program_len) {
  const size_t required_tokens = (size_t) std::max(1, batch_size) * (size_t) std::max(1, max_program_len);
  const int required_batch = std::max(1, batch_size);
  if (
    ctx.h_programs && ctx.h_lengths && ctx.h_sums &&
    ctx.h_program_token_cap >= required_tokens &&
    ctx.h_batch_cap >= required_batch
  ) {
    return;
  }

  release_host_buffers(ctx);
  CUDA_CHECK(cudaMallocHost((void **) &ctx.h_programs, required_tokens * sizeof(GpuToken)));
  CUDA_CHECK(cudaMallocHost((void **) &ctx.h_lengths, (size_t) required_batch * sizeof(int)));
  CUDA_CHECK(cudaMallocHost((void **) &ctx.h_sums, (size_t) required_batch * sizeof(float)));
  ctx.h_program_token_cap = required_tokens;
  ctx.h_batch_cap = required_batch;
}

int sample_block_count(int N) {
  constexpr int threads = 256;
  int blocks_y = (N + threads - 1) / threads;
  return std::max(1, std::min(blocks_y, 128));
}

__device__ float eval_program_single_sample(
  const GpuToken * prog,
  int len,
  const float * x,
  int D
) {
  float stack[STACK_CAP];
  int sp = 0;

  for (int i = 0; i < len; ++i) {
    GpuToken tok = prog[i];
    GpuOpCode opcode = static_cast<GpuOpCode>(tok.opcode);

    if (opcode == GpuOpCode::VAR) {
      if (tok.var_index < 0 || tok.var_index >= D || sp >= STACK_CAP) {
        return PENALTY_F;
      }
      stack[sp++] = x[tok.var_index];
      continue;
    }

    if (opcode == GpuOpCode::CONST) {
      if (sp >= STACK_CAP) {
        return PENALTY_F;
      }
      stack[sp++] = (float) tok.value;
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
        return PENALTY_F;
      }
      float b = stack[--sp];
      float a = stack[--sp];
      float r = 0.0f;

      if (opcode == GpuOpCode::ADD) {
        r = a + b;
      } else if (opcode == GpuOpCode::SUB) {
        r = a - b;
      } else if (opcode == GpuOpCode::MUL) {
        r = a * b;
      } else if (opcode == GpuOpCode::DIV) {
        if (b == 0.0f) {
          return PENALTY_F;
        }
        r = a / b;
      } else {
        r = a / sqrtf(1.0f + b * b);
      }

      if (!finite_dev(r) || sp >= STACK_CAP) {
        return PENALTY_F;
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
        return PENALTY_F;
      }
      float a = stack[--sp];
      float r = 0.0f;

      if (opcode == GpuOpCode::SIN) {
        r = sinf(a);
      } else if (opcode == GpuOpCode::COS) {
        r = cosf(a);
      } else if (opcode == GpuOpCode::LOG) {
        return PENALTY_F;
      } else if (opcode == GpuOpCode::SQRT) {
        return PENALTY_F;
      } else if (opcode == GpuOpCode::NEG) {
        r = -a;
      } else if (opcode == GpuOpCode::INV) {
        if (a == 0.0f) {
          return PENALTY_F;
        }
        r = 1.0f / a;
      } else if (opcode == GpuOpCode::SQUARE) {
        r = a * a;
      } else {
        r = a * a * a;
      }

      if (!finite_dev(r) || sp >= STACK_CAP) {
        return PENALTY_F;
      }
      stack[sp++] = r;
      continue;
    }

    return PENALTY_F;
  }

  if (sp != 1 || !finite_dev(stack[0])) {
    return PENALTY_F;
  }
  return stack[0];
}

__global__ void fitness_kernel_batch(
  const GpuToken * programs,
  const int * lengths,
  int max_program_len,
  const float * X,
  const float * y,
  int N,
  int D,
  int batch_size,
  int fitness_kind,
  float * sums_out
) {
  int p = (int) blockIdx.x;
  if (p >= batch_size) {
    return;
  }

  __shared__ float sh[256];
  int tid = (int) threadIdx.x;
  int blocks_y = (int) gridDim.y;
  int by = (int) blockIdx.y;
  const GpuToken * prog = programs + (size_t) p * (size_t) max_program_len;
  int len = lengths[p];

  float local = 0.0f;
  for (int s = by * (int) blockDim.x + tid; s < N; s += (int) blockDim.x * blocks_y) {
    const float * row = X + (size_t) s * (size_t) D;
    float y_hat = eval_program_single_sample(prog, len, row, D);
    float diff = y_hat - y[s];
    float loss = fitness_kind == static_cast<int>(GpuFitnessKind::MAE) ? fabsf(diff) : diff * diff;
    if (!finite_dev(loss)) {
      loss = PENALTY_F;
    }
    local += loss;
    if (!finite_dev(local)) {
      local = PENALTY_F;
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
    atomicAdd(&sums_out[p], sh[0]);
  }
}

void ensure_single_graph(GpuEvalContext & ctx) {
  ensure_stream(ctx);
  ensure_host_buffers(ctx, 1, ctx.max_program_len);

  const int blocks_y = sample_block_count(ctx.N);
  if (
    ctx.single_graph_exec &&
    ctx.single_graph_max_program_len == ctx.max_program_len &&
    ctx.single_graph_blocks_y == blocks_y
  ) {
    return;
  }

  destroy_single_graph(ctx);

  cudaStream_t stream = eval_stream(ctx);
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

  CUDA_CHECK(cudaMemcpyAsync(
    ctx.d_programs,
    ctx.h_programs,
    (size_t) ctx.max_program_len * sizeof(GpuToken),
    cudaMemcpyHostToDevice,
    stream
  ));
  CUDA_CHECK(cudaMemcpyAsync(
    ctx.d_lengths,
    ctx.h_lengths,
    sizeof(int),
    cudaMemcpyHostToDevice,
    stream
  ));
  CUDA_CHECK(cudaMemsetAsync(ctx.d_sums, 0, sizeof(float), stream));

  const int threads = 256;
  dim3 block(threads);
  dim3 grid(1, (unsigned) blocks_y);
  fitness_kernel_batch<<<grid, block, 0, stream>>>(
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
  CUDA_CHECK(cudaMemcpyAsync(
    ctx.h_sums,
    ctx.d_sums,
    sizeof(float),
    cudaMemcpyDeviceToHost,
    stream
  ));

  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaGraphUpload(graph_exec, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ctx.single_graph_exec = graph_exec;
  ctx.single_graph_max_program_len = ctx.max_program_len;
  ctx.single_graph_blocks_y = blocks_y;
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

  std::vector<float> hX((size_t) ctx.N * (size_t) ctx.D);
  std::vector<float> hy((size_t) ctx.N);
  for (int i = 0; i < ctx.N; ++i) {
    for (int j = 0; j < ctx.D; ++j) {
      hX[(size_t) i * (size_t) ctx.D + (size_t) j] = X(i, j);
    }
    hy[(size_t) i] = y(i);
  }

  CUDA_CHECK(cudaMalloc(&ctx.d_X, hX.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&ctx.d_y, hy.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(ctx.d_X, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx.d_y, hy.data(), hy.size() * sizeof(float), cudaMemcpyHostToDevice));
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
  release_host_buffers(ctx);
  if (ctx.d_X) CUDA_CHECK(cudaFree(ctx.d_X));
  if (ctx.d_y) CUDA_CHECK(cudaFree(ctx.d_y));
  if (ctx.d_programs) CUDA_CHECK(cudaFree(ctx.d_programs));
  if (ctx.d_lengths) CUDA_CHECK(cudaFree(ctx.d_lengths));
  if (ctx.d_sums) CUDA_CHECK(cudaFree(ctx.d_sums));
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
  destroy_single_graph(ctx);
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
    ensure_host_buffers(ctx, ctx.batch_cap, ctx.max_program_len);
    return;
  }

  destroy_single_graph(ctx);
  if (ctx.d_programs) CUDA_CHECK(cudaFree(ctx.d_programs));
  if (ctx.d_lengths) CUDA_CHECK(cudaFree(ctx.d_lengths));
  if (ctx.d_sums) CUDA_CHECK(cudaFree(ctx.d_sums));

  ctx.batch_cap = new_batch_cap;
  ctx.max_program_len = new_max_len;
  CUDA_CHECK(cudaMalloc(&ctx.d_programs, (size_t) ctx.batch_cap * (size_t) ctx.max_program_len * sizeof(GpuToken)));
  CUDA_CHECK(cudaMalloc(&ctx.d_lengths, (size_t) ctx.batch_cap * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&ctx.d_sums, (size_t) ctx.batch_cap * sizeof(float)));
  ensure_host_buffers(ctx, ctx.batch_cap, ctx.max_program_len);
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
  std::memcpy(ctx.h_programs, flat_programs, program_bytes);
  std::memcpy(ctx.h_lengths, lengths, (size_t) batch_size * sizeof(int));
  CUDA_CHECK(cudaMemcpyAsync(ctx.d_programs, ctx.h_programs, program_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(ctx.d_lengths, ctx.h_lengths, (size_t) batch_size * sizeof(int), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemsetAsync(ctx.d_sums, 0, (size_t) batch_size * sizeof(float), stream));

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

  CUDA_CHECK(cudaMemcpyAsync(ctx.h_sums, ctx.d_sums, (size_t) batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int i = 0; i < batch_size; ++i) {
    double value = ctx.h_sums[(size_t) i] / (double) ctx.N;
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
  ensure_single_graph(ctx);
  std::memcpy(ctx.h_programs, tokens, (size_t) length * sizeof(GpuToken));
  ctx.h_lengths[0] = length;

  cudaStream_t stream = eval_stream(ctx);
  CUDA_CHECK(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(ctx.single_graph_exec), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  double value = ctx.h_sums[0] / (double) ctx.N;
  if (!std::isfinite(value) || value < 0.0) {
    value = PENALTY;
  }
  return value;
}
