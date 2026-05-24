#ifndef GPU_TOKEN_HPP
#define GPU_TOKEN_HPP

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
  NEG = 11,
  INV = 12,
  SQUARE = 13,
  CUBE = 14,
  VAR = 100,
  CONST = 101,
  INVALID = -1
};

struct GpuToken {
  int opcode = static_cast<int>(GpuOpCode::INVALID);
  int var_index = -1;
  double value = 0.0;
};

const char * gpu_opcode_name(int opcode);

#endif
