#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu/gpu_token.hpp"

#define CUDA_CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    std::exit(1); \
  } \
} while (0)

namespace {

constexpr int kStackCap = 64;
constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / 32;

__device__ __forceinline__ bool finite_dev(float x) {
  return isfinite(x);
}

__device__ float eval_program_single_sample(
  const GpuToken * prog,
  int len,
  const float * x,
  int D
) {
  float stack[kStackCap];
  int sp = 0;

  for (int i = 0; i < len; ++i) {
    GpuToken tok = prog[i];
    GpuOpCode opcode = static_cast<GpuOpCode>(tok.opcode);

    if (opcode == GpuOpCode::VAR) {
      if (tok.var_index < 0 || tok.var_index >= D || sp >= kStackCap) {
        return 1.0e12f;
      }
      stack[sp++] = x[tok.var_index];
      continue;
    }

    if (opcode == GpuOpCode::CONST) {
      if (sp >= kStackCap) {
        return 1.0e12f;
      }
      stack[sp++] = tok.value;
      continue;
    }

    if (
      opcode == GpuOpCode::ADD ||
      opcode == GpuOpCode::SUB ||
      opcode == GpuOpCode::MUL ||
      opcode == GpuOpCode::AQ
    ) {
      if (sp < 2) {
        return 1.0e12f;
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
      } else {
        r = a / sqrtf(1.0f + b * b);
      }
      if (!finite_dev(r) || sp >= kStackCap) {
        return 1.0e12f;
      }
      stack[sp++] = r;
      continue;
    }

    return 1.0e12f;
  }

  if (sp != 1 || !finite_dev(stack[0])) {
    return 1.0e12f;
  }
  return stack[0];
}

__global__ void block_program_kernel(
  const GpuToken * programs,
  const int * lengths,
  int max_program_len,
  const float * X,
  const float * y,
  int N,
  int D,
  int batch_size,
  float * sink
) {
  int p = static_cast<int>(blockIdx.x);
  if (p >= batch_size) {
    return;
  }
  int tid = static_cast<int>(threadIdx.x);
  int by = static_cast<int>(blockIdx.y);
  int blocks_y = static_cast<int>(gridDim.y);
  const GpuToken * prog = programs + static_cast<size_t>(p) * static_cast<size_t>(max_program_len);
  int len = lengths[p];
  float local = 0.0f;
  for (int s = by * blockDim.x + tid; s < N; s += blockDim.x * blocks_y) {
    const float * row = X + static_cast<size_t>(s) * static_cast<size_t>(D);
    float diff = eval_program_single_sample(prog, len, row, D) - y[s];
    local += diff * diff;
  }
  size_t out = (
    (static_cast<size_t>(p) * static_cast<size_t>(gridDim.y) + static_cast<size_t>(by)) *
    static_cast<size_t>(blockDim.x)
  ) + static_cast<size_t>(tid);
  sink[out] = local;
}

__global__ void lane_program_kernel(
  const GpuToken * programs,
  const int * lengths,
  int max_program_len,
  const float * X,
  const float * y,
  int N,
  int D,
  int batch_size,
  float * sink
) {
  int lane = static_cast<int>(threadIdx.x) & 31;
  int warp = static_cast<int>(threadIdx.x) >> 5;
  int p = static_cast<int>(blockIdx.x) * 32 + lane;
  if (p >= batch_size) {
    return;
  }
  int by = static_cast<int>(blockIdx.y);
  int blocks_y = static_cast<int>(gridDim.y);
  const GpuToken * prog = programs + static_cast<size_t>(p) * static_cast<size_t>(max_program_len);
  int len = lengths[p];
  float local = 0.0f;
  for (int s = by * kWarpsPerBlock + warp; s < N; s += blocks_y * kWarpsPerBlock) {
    const float * row = X + static_cast<size_t>(s) * static_cast<size_t>(D);
    float diff = eval_program_single_sample(prog, len, row, D) - y[s];
    local += diff * diff;
  }
  size_t out = (
    (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(gridDim.y) + static_cast<size_t>(by)) *
    static_cast<size_t>(blockDim.x)
  ) + static_cast<size_t>(threadIdx.x);
  sink[out] = local;
}

std::vector<int> base_ops(int op_count, bool mixed_ops) {
  std::vector<int> ops;
  for (int i = 0; i < op_count; ++i) {
    if (!mixed_ops) {
      ops.push_back(static_cast<int>(GpuOpCode::ADD));
      continue;
    }
    switch (i % 4) {
      case 0: ops.push_back(static_cast<int>(GpuOpCode::ADD)); break;
      case 1: ops.push_back(static_cast<int>(GpuOpCode::SUB)); break;
      case 2: ops.push_back(static_cast<int>(GpuOpCode::MUL)); break;
      default: ops.push_back(static_cast<int>(GpuOpCode::AQ)); break;
    }
  }
  return ops;
}

void make_program(
  GpuToken * out,
  int D,
  int len,
  const std::vector<int> & ops
) {
  int pos = 0;
  out[pos++] = GpuToken{static_cast<int>(GpuOpCode::VAR), 0, 0.0f};
  out[pos++] = GpuToken{static_cast<int>(GpuOpCode::VAR), 1 % D, 0.0f};
  out[pos++] = GpuToken{ops[0], -1, 0.0f};
  for (int i = 1; i < static_cast<int>(ops.size()); ++i) {
    out[pos++] = GpuToken{static_cast<int>(GpuOpCode::VAR), (i + 1) % D, 0.0f};
    out[pos++] = GpuToken{ops[i], -1, 0.0f};
  }
  while (pos < len) {
    out[pos++] = GpuToken{static_cast<int>(GpuOpCode::INVALID), -1, 0.0f};
  }
}

std::vector<GpuToken> make_programs(
  int batch_size,
  int D,
  int max_len,
  bool variable_lengths,
  bool mixed_ops,
  std::vector<int> & lengths
) {
  std::vector<GpuToken> programs(static_cast<size_t>(batch_size) * static_cast<size_t>(max_len));
  std::mt19937 rng(20260613);
  std::vector<int> length_choices = {7, 15, 23, 31};
  lengths.resize(batch_size);
  for (int p = 0; p < batch_size; ++p) {
    int len = variable_lengths ? length_choices[p % static_cast<int>(length_choices.size())] : max_len;
    int op_count = (len - 1) / 2;
    std::vector<int> local_ops = base_ops(op_count, mixed_ops);
    if (mixed_ops) {
      std::shuffle(local_ops.begin(), local_ops.end(), rng);
    }
    lengths[p] = len;
    make_program(&programs[static_cast<size_t>(p) * static_cast<size_t>(max_len)], D, len, local_ops);
  }
  return programs;
}

template <typename Kernel>
float time_kernel(
  Kernel kernel,
  dim3 grid,
  dim3 block,
  int reps
) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < 10; ++i) {
    kernel();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < reps; ++i) {
    kernel();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(reps);
}

struct Result {
  std::string mapping;
  std::string scenario;
  float ms = 0.0f;
  double token_sample_per_s = 0.0;
};

} // namespace

int main(int argc, char ** argv) {
  int batch_size = 1024;
  int N = 32768;
  int D = 16;
  int reps = 80;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next_int = [&](int & out) {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
      out = std::stoi(argv[++i]);
    };
    if (arg == "--batch") next_int(batch_size);
    else if (arg == "--samples") next_int(N);
    else if (arg == "--features") next_int(D);
    else if (arg == "--reps") next_int(reps);
    else throw std::runtime_error("unknown argument: " + arg);
  }
  if (batch_size % 32 != 0) {
    throw std::runtime_error("--batch must be a multiple of 32");
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  constexpr int len = 31;
  std::vector<int> lengths(batch_size, len);
  std::vector<float> X(static_cast<size_t>(N) * static_cast<size_t>(D));
  std::vector<float> y(N);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float & v : X) v = dist(rng);
  for (float & v : y) v = dist(rng);

  GpuToken * d_programs = nullptr;
  int * d_lengths = nullptr;
  float * d_X = nullptr;
  float * d_y = nullptr;
  float * d_sink_block = nullptr;
  float * d_sink_lane = nullptr;
  int block_blocks_y = std::max(1, std::min((N + kThreads - 1) / kThreads, 128));
  int lane_blocks_y = block_blocks_y * 32;
  size_t program_bytes = static_cast<size_t>(batch_size) * static_cast<size_t>(len) * sizeof(GpuToken);
  size_t length_bytes = static_cast<size_t>(batch_size) * sizeof(int);
  size_t x_bytes = static_cast<size_t>(N) * static_cast<size_t>(D) * sizeof(float);
  size_t y_bytes = static_cast<size_t>(N) * sizeof(float);
  size_t sink_block_elems = static_cast<size_t>(batch_size) * static_cast<size_t>(block_blocks_y) * kThreads;
  size_t sink_lane_elems = static_cast<size_t>(batch_size / 32) * static_cast<size_t>(lane_blocks_y) * kThreads;

  CUDA_CHECK(cudaMalloc(&d_programs, program_bytes));
  CUDA_CHECK(cudaMalloc(&d_lengths, length_bytes));
  CUDA_CHECK(cudaMalloc(&d_X, x_bytes));
  CUDA_CHECK(cudaMalloc(&d_y, y_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink_block, sink_block_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sink_lane, sink_lane_elems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_X, X.data(), x_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, y.data(), y_bytes, cudaMemcpyHostToDevice));

  std::vector<Result> results;
  struct Scenario {
    const char * name;
    bool variable_lengths;
    bool mixed_ops;
  };
  std::vector<Scenario> scenarios = {
    {"uniform_length", false, false},
    {"variable_length", true, false},
    {"variable_length_mixed_ops", true, true},
  };
  for (const Scenario & scenario : scenarios) {
    std::vector<GpuToken> programs = make_programs(
      batch_size,
      D,
      len,
      scenario.variable_lengths,
      scenario.mixed_ops,
      lengths
    );
    long long active_tokens_per_population = 0;
    for (int value : lengths) {
      active_tokens_per_population += value;
    }
    CUDA_CHECK(cudaMemcpy(d_programs, programs.data(), program_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, lengths.data(), length_bytes, cudaMemcpyHostToDevice));
    std::string scenario_name = scenario.name;

    dim3 block(kThreads);
    dim3 grid_block(static_cast<unsigned>(batch_size), static_cast<unsigned>(block_blocks_y));
    auto block_launch = [&]() {
      block_program_kernel<<<grid_block, block>>>(
        d_programs, d_lengths, len, d_X, d_y, N, D, batch_size, d_sink_block
      );
    };
    CUDA_CHECK(cudaGetLastError());
    float block_ms = time_kernel(block_launch, grid_block, block, reps);
    CUDA_CHECK(cudaGetLastError());
    results.push_back(Result{
      "program_per_block",
      scenario_name,
      block_ms,
      static_cast<double>(active_tokens_per_population) * static_cast<double>(N) /
        (static_cast<double>(block_ms) * 1.0e-3)
    });

    dim3 grid_lane(static_cast<unsigned>(batch_size / 32), static_cast<unsigned>(lane_blocks_y));
    auto lane_launch = [&]() {
      lane_program_kernel<<<grid_lane, block>>>(
        d_programs, d_lengths, len, d_X, d_y, N, D, batch_size, d_sink_lane
      );
    };
    CUDA_CHECK(cudaGetLastError());
    float lane_ms = time_kernel(lane_launch, grid_lane, block, reps);
    CUDA_CHECK(cudaGetLastError());
    results.push_back(Result{
      "program_per_lane",
      scenario_name,
      lane_ms,
      static_cast<double>(active_tokens_per_population) * static_cast<double>(N) /
        (static_cast<double>(lane_ms) * 1.0e-3)
    });
  }

  std::cout << "device," << prop.name << "\n";
  std::cout << "batch," << batch_size << "\n";
  std::cout << "samples," << N << "\n";
  std::cout << "features," << D << "\n";
  std::cout << "program_len," << len << "\n";
  std::cout << "reps," << reps << "\n";
  std::cout << "mapping,scenario,kernel_ms,active_token_sample_per_s\n";
  std::cout << std::fixed << std::setprecision(6);
  for (const Result & r : results) {
    std::cout << r.mapping << "," << r.scenario << "," << r.ms << "," << r.token_sample_per_s << "\n";
  }

  CUDA_CHECK(cudaFree(d_programs));
  CUDA_CHECK(cudaFree(d_lengths));
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_sink_block));
  CUDA_CHECK(cudaFree(d_sink_lane));
  return 0;
}
