#ifndef EXPRESSION_SERIALIZER_HPP
#define EXPRESSION_SERIALIZER_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

#include "../node.hpp"
#include "gpu_token.hpp"

struct SerializedProgram {
  std::vector<GpuToken> tokens;
};

inline GpuToken op_to_gpu_token(Op * op) {
  if (!op) {
    return GpuToken{static_cast<int>(GpuOpCode::INVALID), -1, 0.0};
  }

  if (op->type() == OpType::otFeat) {
    return GpuToken{static_cast<int>(GpuOpCode::VAR), ((Feat*) op)->id, 0.0};
  }

  if (op->type() == OpType::otConst) {
    Const * c = (Const*) op;
    if (std::isnan(c->c)) {
      c->sym();
    }
    return GpuToken{static_cast<int>(GpuOpCode::CONST), -1, (float) c->c};
  }

  const std::string sym = op->sym();
  if (sym == "+") {
    return GpuToken{static_cast<int>(GpuOpCode::ADD), -1, 0.0};
  }
  if (sym == "-") {
    return GpuToken{static_cast<int>(GpuOpCode::SUB), -1, 0.0};
  }
  if (sym == "*") {
    return GpuToken{static_cast<int>(GpuOpCode::MUL), -1, 0.0};
  }
  if (sym == "/") {
    return GpuToken{static_cast<int>(GpuOpCode::DIV), -1, 0.0};
  }
  if (sym == "aq") {
    return GpuToken{static_cast<int>(GpuOpCode::AQ), -1, 0.0};
  }
  if (sym == "¬") {
    return GpuToken{static_cast<int>(GpuOpCode::NEG), -1, 0.0};
  }
  if (sym == "1/") {
    return GpuToken{static_cast<int>(GpuOpCode::INV), -1, 0.0};
  }
  if (sym == "**2") {
    return GpuToken{static_cast<int>(GpuOpCode::SQUARE), -1, 0.0};
  }
  if (sym == "**3") {
    return GpuToken{static_cast<int>(GpuOpCode::CUBE), -1, 0.0};
  }
  if (sym == "sin") {
    return GpuToken{static_cast<int>(GpuOpCode::SIN), -1, 0.0};
  }
  if (sym == "cos") {
    return GpuToken{static_cast<int>(GpuOpCode::COS), -1, 0.0};
  }
  if (sym == "log") {
    return GpuToken{static_cast<int>(GpuOpCode::LOG), -1, 0.0};
  }
  if (sym == "sqrt") {
    return GpuToken{static_cast<int>(GpuOpCode::SQRT), -1, 0.0};
  }
  throw std::runtime_error("Unsupported operator for GPU serialization: " + sym);
}

inline int get_active_program_len(Node * root);

inline void serialize_active_tree_to_postfix_into(
  Node * root,
  std::vector<GpuToken> & out
) {
  if (!root) {
    throw std::runtime_error("Cannot serialize a null expression tree");
  }

  int arity = root->op->arity();
  if ((int) root->children.size() < arity) {
    throw std::runtime_error("Expression tree has fewer children than its active arity");
  }

  for (int i = 0; i < arity; ++i) {
    serialize_active_tree_to_postfix_into(root->children[i], out);
  }
  out.push_back(op_to_gpu_token(root->op));
}

inline SerializedProgram serialize_active_tree_to_postfix(Node * root) {
  SerializedProgram result;
  result.tokens.reserve(get_active_program_len(root));
  serialize_active_tree_to_postfix_into(root, result.tokens);
  return result;
}

inline int get_active_program_len(Node * root) {
  if (!root) {
    return 0;
  }
  int len = 1;
  int arity = root->op->arity();
  for (int i = 0; i < arity; ++i) {
    len += get_active_program_len(root->children[i]);
  }
  return len;
}

#endif
