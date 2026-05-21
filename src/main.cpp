#include <iostream>

#include "globals.hpp"
#include "myeig.hpp"
#include "util.hpp"
#include "evolution.hpp"
#include "ims.hpp"
#include "node.hpp"
#include "tests.hpp"

using namespace myeig;

int main(int argc, char** argv){
  try {
    g::read_options(argc, argv);

    auto t = Test();
    t.run_all();

    auto start_time = tick();
    //auto evo = Evolution();
    //evo.run();
    auto * ims = new IMS();
    ims->run();
    delete ims;
    print("Evaluation counters: total=", g::fit_func->evaluations,
          ", gpu=", g::num_gpu_evaluations,
          ", cpu=", g::fit_func->evaluations - g::num_gpu_evaluations,
          ", accepted_moves=", g::num_accepted_moves,
          ", rejected_moves=", g::num_rejected_moves,
          ", meaningful_candidates=", g::num_meaningful_candidates,
          ", nonmeaningful_candidates=", g::num_nonmeaningful_candidates);
    print("Runtime: ",tock(start_time),"s");

    g::clear_globals();
    return 0;
  } catch (const std::exception & e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

}
