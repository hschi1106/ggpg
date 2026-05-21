#include "expression_serializer.hpp"

// The base project defines many non-inline helpers in headers. Keeping the
// serializer implementation inline avoids duplicate symbols when the header is
// included by both the CLI and pybind targets.
