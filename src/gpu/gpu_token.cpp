#include "gpu_token.hpp"

const char * gpu_opcode_name(int opcode) {
  switch (static_cast<GpuOpCode>(opcode)) {
    case GpuOpCode::ADD: return "ADD";
    case GpuOpCode::SUB: return "SUB";
    case GpuOpCode::MUL: return "MUL";
    case GpuOpCode::DIV: return "DIV";
    case GpuOpCode::SIN: return "SIN";
    case GpuOpCode::COS: return "COS";
    case GpuOpCode::LOG: return "LOG";
    case GpuOpCode::SQRT: return "SQRT";
    case GpuOpCode::NEG: return "NEG";
    case GpuOpCode::INV: return "INV";
    case GpuOpCode::SQUARE: return "SQUARE";
    case GpuOpCode::CUBE: return "CUBE";
    case GpuOpCode::VAR: return "VAR";
    case GpuOpCode::CONST: return "CONST";
    case GpuOpCode::INVALID: return "INVALID";
  }
  return "UNKNOWN";
}
