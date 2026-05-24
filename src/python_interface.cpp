#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <memory>
#include <vector>

#include "util.hpp"
#include "myeig.hpp"
#include "globals.hpp"
#include "ims.hpp"

namespace py = pybind11; 
using namespace std;

namespace {

struct RunStateGuard {
  ~RunStateGuard() {
    try {
      g::reset();
    } catch (...) {
    }
  }
};

bool has_lib_option(const vector<string> & opts) {
  for (const string & opt : opts) {
    if (opt == "-lib" || opt == "--call_as_lib") {
      return true;
    }
  }
  return false;
}

} // namespace

py::list evolve(string options, myeig::Mat &X, myeig::Vec &y) {
  // 1. SETUP
  RunStateGuard run_state_guard;
  auto opts = split_string(options, " ");
  if (!has_lib_option(opts)) {
    opts.push_back("-lib");
  }

  vector<char*> argv;
  argv.reserve(opts.size() + 1);
  string title = "gpg";
  argv.push_back((char*) title.c_str());
  for (string & opt : opts) {
    argv.push_back((char*) opt.c_str());
  }
  g::read_options((int) argv.size(), argv.data());

  // initialize evolution handler 
  std::unique_ptr<IMS> ims = std::make_unique<IMS>();

  // set training set
  g::fit_func->set_Xy(X, y);
  // set terminals
  g::set_terminals(g::lib_tset);
  g::apply_feature_selection(g::lib_feat_sel_number);
  g::set_terminal_probabilities(g::lib_tset_probs);
  print("terminal set: ",g::str_terminal_set()," (probs: ",g::lib_tset_probs,")");
  // set batch size
  g::set_batch_size(g::lib_batch_size);
  print("batch size: ", g::batch_size);
  g::sync_gpu_context();

  // 2. RUN
  ims->run();

  // 3. OUTPUT
  if (ims->elites_per_complexity.empty()) {
    throw runtime_error("Not models found, something went wrong");
  }
  py::list models;
  for (auto it = ims->elites_per_complexity.begin(); it != ims->elites_per_complexity.end(); it++) {
    string model_repr = it->second->human_repr();
    models.append(model_repr);
  }

  // 4. CLEANUP
  ims.reset();

  return models;
}

bool cuda_enabled() {
#ifdef GPG_USE_CUDA
  return true;
#else
  return false;
#endif
}

py::list available_backends() {
  py::list result;
  result.append("cpu_original");
#ifdef GPG_USE_CUDA
  result.append("gpu_exact_gom");
  result.append("gpu_batch_gom");
#endif
  return result;
}

PYBIND11_MODULE(_pb_gpg, m) {
  m.doc() = "pybind11-based interface for gpg"; // optional module docstring
  m.def("evolve", &evolve, "Runs gpg evolution in C++");
  m.def("cuda_enabled", &cuda_enabled, "Returns true when the extension was built with CUDA support");
  m.def("available_backends", &available_backends, "Returns execution backends available in this extension build");
}
