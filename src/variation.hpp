#ifndef VARIATION_H
#define VARIATION_H

#include "globals.hpp"
#include "node.hpp"
#include "operator.hpp"
#include "util.hpp"
#include "selection.hpp"
#include "fos.hpp"
#include "rng.hpp"
#include "gpu/expression_serializer.hpp"
#include "gpu/fitness_cuda.hpp"

#include <cmath>
#include <vector>

using namespace std;

Op * _sample_operator(vector<Op *> & operators, Vec & cumul_probs) {
  double r = Rng::randu();
  for(int i = 0; i<operators.size();i++){
    if(r <= (double) cumul_probs[i]){
      return operators[i]->clone(); 
    }
  }
  return operators[operators.size() - 1]->clone();  
}

Op * _sample_function() {
  return _sample_operator(g::functions, g::cumul_fset_probs);
}

Op * _sample_terminal() {
  return _sample_operator(g::terminals, g::cumul_tset_probs);
}

Node * _grow_tree_recursive(int max_arity, int max_depth_left, int actual_depth_left, int curr_depth, float terminal_prob=.25) {
  Node * n = NULL;

  if (max_depth_left > 0) {
    if (actual_depth_left > 0 && Rng::randu() < 1.0-terminal_prob) {
      n = new Node(_sample_function());
    } else {
      n = new Node(_sample_terminal());
    }

    for (int i = 0; i < max_arity; i++) {
      Node * c = _grow_tree_recursive(max_arity,
        max_depth_left - 1, actual_depth_left - 1, curr_depth + 1, terminal_prob);
      n->append(c);
    }
  } else {
    n = new Node(_sample_terminal());
  }

  assert(n != NULL);

  return n;
}

Node * generate_tree(int max_depth, string init_type="hh") {

  int max_arity = 0;
  for(Op * op : g::functions) {
    int op_arity = op->arity();
    if (op_arity > max_arity)
      max_arity = op_arity;
  }

  Node * tree = NULL;
  int actual_depth = max_depth;

  if (init_type == "rhh" || init_type == "hh") {
    if (init_type == "rhh")
      actual_depth = Rng::randu() * max_depth;
    
    bool is_full = Rng::randu() < .5;

    if (is_full)
      tree = _grow_tree_recursive(max_arity, max_depth, actual_depth, -1, 0.0);
    else
      tree = _grow_tree_recursive(max_arity, max_depth, actual_depth, -1);

  } else {
    throw runtime_error("Unrecognized init_type "+init_type);
  }

  assert(tree);

  return tree;
}

Node * coeff_mut(Node * parent, bool return_copy=true, vector<int> * changed_indices = NULL, vector<Op*> * backup_ops = NULL) {
  Node * tree = parent;
  if (return_copy) {
    tree = parent->clone();
  }
  
  if (g::cmut_prob > 0 && g::cmut_temp > 0) {
    // apply coeff mut to all nodes that are constants
    vector<Node*> nodes = tree->subtree();
    for(int i = 0; i < nodes.size(); i++) {
      Node * n = nodes[i];
      if (
        n->op->type() == OpType::otConst &&
        Rng::randu() < g::cmut_prob
      ) {
        float prev_c = ((Const*)n->op)->c;
        float std = g::cmut_temp*abs(prev_c);
        if (std < g::cmut_eps)
          std = g::cmut_eps;
        float mutated_c = roundd(prev_c + Rng::randn()*std, NUM_PRECISION); 
        ((Const*)n->op)->c = mutated_c;
        // case in which we are going through GOM
        if (changed_indices != NULL) {
          // add the backup node only if this wasn't changed previously
          if (find(changed_indices->begin(), changed_indices->end(), i) == changed_indices->end()) {
            changed_indices->push_back(i);
            backup_ops->push_back(new Const(prev_c));
          };
        }
      }
    }
  }
  return tree;
}

vector<int> _sample_crossover_mask(int num_nodes) {
  auto crossover_mask = Rng::rand_perm(num_nodes);
  int k = 1+sqrt(num_nodes)*abs(Rng::randn());
  k = min(k, num_nodes);
  crossover_mask.erase(crossover_mask.begin() + k, crossover_mask.end());
  assert(crossover_mask.size() == k);
  return crossover_mask;
}

Node * crossover(Node * parent, Node * donor) {
  Node * offspring = parent->clone();
  auto nodes = offspring->subtree();
  auto d_nodes = donor->subtree();

  // sample a crossover mask
  auto crossover_mask = _sample_crossover_mask(nodes.size());
  for(int i : crossover_mask) {
    delete nodes[i]->op;
    nodes[i]->op = d_nodes[i]->op->clone();
  }

  return offspring;
}

Node * mutation(Node * parent, vector<Op*> & functions, vector<Op*> & terminals, float prob_fun = 0.75) {
  Node * offspring = parent->clone();
  auto nodes = offspring->subtree();

  // sample a crossover mask
  auto crossover_mask = _sample_crossover_mask(nodes.size());
  for(int i : crossover_mask) {
    delete nodes[i]->op;
    if (nodes[i]->children.size() > 0 && Rng::randu() < prob_fun) {
      nodes[i]->op = _sample_function();
    }
    else {
      nodes[i]->op = _sample_terminal();
    }
  }

  return offspring;
}

/*Node * gom(Node * parent, vector<Node*> & population, vector<vector<int>> & fos) {
  Node * offspring = parent->clone();
  Node * backup = parent->clone();

  float backup_fitness = backup->fitness;

  vector<Node*> offspring_nodes = offspring->subtree();

  auto random_fos_order = rand_perm(fos.size());

  for(int i = 0; i < fos.size(); i++) {
    auto crossover_mask = fos[random_fos_order[i]];
    // fetch donor
    Node * donor = population[randu()*population.size()];
    vector<Node*> donor_nodes = donor->subtree();

    for(const int idx : crossover_mask) {
      delete offspring_nodes[idx]->op;
      offspring_nodes[idx]->op = donor_nodes[idx]->op->clone();
    }

    // apply coeff mut
    coeff_mut(offspring, false);

    // check is not worse
    float new_fitness = g::fit_func->get_fitness(offspring);
    if (new_fitness > backup_fitness) {
      // undo
      offspring->clear();
      offspring = backup->clone();
      offspring_nodes = offspring->subtree();
    } else {
      // retain
      backup->clear();
      backup = offspring->clone();
      backup_fitness = new_fitness;
    }
  }
  
  return offspring;
}*/


float evaluate_node_gpu(Node * tree, vector<GpuToken> * token_scratch = NULL) {
  if (!g::gpu_ctx) {
    throw runtime_error("GPU context is not initialized");
  }

  vector<GpuToken> local_tokens;
  vector<GpuToken> & tokens = token_scratch ? *token_scratch : local_tokens;
  tokens.clear();
  serialize_active_tree_to_postfix_into(tree, tokens);

  double raw_fitness = evaluate_fitness_gpu_single(
    *g::gpu_ctx,
    tokens.data(),
    (int) tokens.size()
  );

  g::fit_func->evaluations += 1;
  g::fit_func->node_evaluations += tree->get_num_nodes(true);
  g::num_gpu_evaluations += 1;

  float fitness = (float) raw_fitness;
  if (!isfinite(fitness) || fitness < 0) {
    fitness = INF;
  }

  float rounded = roundd(fitness, NUM_PRECISION + 2);

  if (g::gpu_check_correctness) {
    float saved = tree->fitness;
    float cpu_fitness = g::fit_func->get_fitness(tree, g::fit_func->X_batch, g::fit_func->y_batch);
    tree->fitness = rounded;
    float fitness_scale = max(abs(cpu_fitness), abs(fitness));
    float relative_tolerance = fitness_scale > (float) 1e6 ? (float) 1e-2 : (fitness_scale > (float) 1e5 ? (float) 5e-3 : (float) 1e-3);
    float tolerance = max((float) 1e-4, abs(cpu_fitness) * relative_tolerance);
    if (isfinite(cpu_fitness) && isfinite(fitness) && cpu_fitness < 1e11 && fitness < 1e11 && abs(cpu_fitness - fitness) > tolerance) {
      throw runtime_error(
        "CPU/GPU fitness mismatch: cpu=" + to_string(cpu_fitness) +
        ", gpu=" + to_string(fitness)
      );
    }
    (void) saved;
  }

  tree->fitness = rounded;
  return fitness;
}

void discard_backup_ops(vector<Op*> & backup_ops) {
  for(Op * op : backup_ops) {
    if (op) {
      delete op;
    }
  }
  backup_ops.clear();
}

Node * efficient_gom_cpu_original(Node * parent, vector<Node*> & population, vector<vector<int>> & fos) {
  Node * offspring = parent->clone();
  float backup_fitness = parent->fitness;
  vector<Node*> offspring_nodes = offspring->subtree();

  auto random_fos_order = Rng::rand_perm(fos.size());

  bool ever_improved = false;
  for(int fos_idx = 0; fos_idx < fos.size(); fos_idx++) {
    
    const vector<int> & crossover_mask = fos[random_fos_order[fos_idx]];
    bool change_is_meaningful = false;
    vector<Op*> backup_ops; backup_ops.reserve(crossover_mask.size());
    vector<int> effectively_changed_indices; effectively_changed_indices.reserve(crossover_mask.size());

    Node * donor = population[Rng::randi(population.size())];
    vector<Node*> donor_nodes = donor->subtree();

    for(const int idx : crossover_mask) {
      // check if swap is not necessary
      if (offspring_nodes[idx]->op->sym() == donor_nodes[idx]->op->sym()) {
        // might need to swap if the node is a constant that might be optimized
        if (g::cmut_prob <= 0 || g::cmut_temp <= 0 || donor_nodes[idx]->op->type() != OpType::otConst)
          continue;
      }

      // then execute the swap
      Op * replaced_op = offspring_nodes[idx]->op;
      offspring_nodes[idx]->op = donor_nodes[idx]->op->clone();
      backup_ops.push_back(replaced_op);
      effectively_changed_indices.push_back(idx);
    }

    // apply coeff mut
    coeff_mut(offspring, false, &effectively_changed_indices, &backup_ops);

    // check if at least one change was meaningful
    for(int i : effectively_changed_indices) {
      Node * n = offspring_nodes[i];
      if (!n->is_intron()) {
        change_is_meaningful = true;
        break;
      }
    }

    // assume nothing changed
    float new_fitness = backup_fitness;
    if (change_is_meaningful) {
      // gotta recompute
      new_fitness = g::fit_func->get_fitness(offspring);
    }

    // check is not worse
    if (new_fitness > backup_fitness) {
      // undo
      for(int i = 0; i < effectively_changed_indices.size(); i++) {
        int changed_idx = effectively_changed_indices[i];
        Node * off_n = offspring_nodes[changed_idx];
        Op * back_op = backup_ops[i];
        delete off_n->op;
        off_n->op = back_op;
        backup_ops[i] = NULL;
        offspring->fitness = backup_fitness;
      }
    } else if (new_fitness < backup_fitness) {
      // it improved
      backup_fitness = new_fitness;
      ever_improved = true;
    }

    // discard backup
    for(Op * op : backup_ops) {
      delete op;
    }

  }

  // variant of forced improvement that is potentially less aggressive, & less expensive to carry out
  if(g::tournament_size > 1 && !ever_improved) {
    // make a tournament between tournament size - 1 candidates + offspring
    vector<Node*> tournament_candidates; tournament_candidates.reserve(g::tournament_size - 1);
    for(int i = 0; i < g::tournament_size - 1; i++) {
      tournament_candidates.push_back(population[Rng::randi(population.size())]);
    }
    tournament_candidates.push_back(offspring);
    Node * winner = tournament(tournament_candidates, g::tournament_size);
    offspring->clear();
    offspring = winner;
  }
  
  return offspring;
}

vector<vector<Node*>> collect_population_subtrees(vector<Node*> & population) {
  vector<vector<Node*>> population_nodes;
  population_nodes.reserve(population.size());
  for(Node * tree : population) {
    population_nodes.push_back(tree->subtree());
  }
  return population_nodes;
}

Node * efficient_gom_gpu_exact(
  Node * parent,
  vector<Node*> & population,
  vector<vector<Node*>> & population_nodes,
  vector<vector<int>> & fos,
  GpuEvalContext & gpu_ctx
) {
  (void) gpu_ctx;
  Node * offspring = parent->clone();
  float backup_fitness = parent->fitness;
  vector<Node*> offspring_nodes = offspring->subtree();
  vector<GpuToken> token_scratch;

  auto random_fos_order = Rng::rand_perm(fos.size());

  bool ever_improved = false;
  for(int fos_idx = 0; fos_idx < fos.size(); fos_idx++) {
    
    const vector<int> & crossover_mask = fos[random_fos_order[fos_idx]];
    bool change_is_meaningful = false;
    vector<Op*> backup_ops; backup_ops.reserve(crossover_mask.size());
    vector<int> effectively_changed_indices; effectively_changed_indices.reserve(crossover_mask.size());

    int donor_idx = Rng::randi(population.size());
    vector<Node*> & donor_nodes = population_nodes[donor_idx];

    for(const int idx : crossover_mask) {
      if (offspring_nodes[idx]->op->sym() == donor_nodes[idx]->op->sym()) {
        if (g::cmut_prob <= 0 || g::cmut_temp <= 0 || donor_nodes[idx]->op->type() != OpType::otConst)
          continue;
      }

      Op * replaced_op = offspring_nodes[idx]->op;
      offspring_nodes[idx]->op = donor_nodes[idx]->op->clone();
      backup_ops.push_back(replaced_op);
      effectively_changed_indices.push_back(idx);
    }

    coeff_mut(offspring, false, &effectively_changed_indices, &backup_ops);

    for(int i : effectively_changed_indices) {
      Node * n = offspring_nodes[i];
      if (!n->is_intron()) {
        change_is_meaningful = true;
        break;
      }
    }

    float new_fitness = backup_fitness;
    if (change_is_meaningful) {
      g::num_meaningful_candidates += 1;
      new_fitness = evaluate_node_gpu(offspring, &token_scratch);
    } else {
      g::num_nonmeaningful_candidates += 1;
    }

    if (new_fitness > backup_fitness) {
      for(int i = 0; i < effectively_changed_indices.size(); i++) {
        int changed_idx = effectively_changed_indices[i];
        Node * off_n = offspring_nodes[changed_idx];
        Op * back_op = backup_ops[i];
        delete off_n->op;
        off_n->op = back_op;
        backup_ops[i] = NULL;
        offspring->fitness = backup_fitness;
      }
      g::num_rejected_moves += 1;
    } else if (new_fitness < backup_fitness) {
      backup_fitness = new_fitness;
      ever_improved = true;
      g::num_accepted_moves += 1;
    } else if (change_is_meaningful) {
      g::num_accepted_moves += 1;
    }

    discard_backup_ops(backup_ops);
  }

  if(g::tournament_size > 1 && !ever_improved) {
    vector<Node*> tournament_candidates; tournament_candidates.reserve(g::tournament_size - 1);
    for(int i = 0; i < g::tournament_size - 1; i++) {
      tournament_candidates.push_back(population[Rng::randi(population.size())]);
    }
    tournament_candidates.push_back(offspring);
    Node * winner = tournament(tournament_candidates, g::tournament_size);
    offspring->clear();
    offspring = winner;
  }
  
  return offspring;
}

Node * efficient_gom(Node * parent, vector<Node*> & population, vector<vector<int>> & fos) {
  if (g::execution_backend == g::ExecutionBackend::GPU_EXACT_GOM) {
    vector<vector<Node*>> population_nodes = collect_population_subtrees(population);
    return efficient_gom_gpu_exact(parent, population, population_nodes, fos, *g::gpu_ctx);
  }
  return efficient_gom_cpu_original(parent, population, fos);
}

struct PendingBatchMove {
  int candidate_idx;
  vector<int> changed_indices;
  vector<Op*> backup_ops;
};

struct GpuBatchEvalScratch {
  vector<int> lengths;
  vector<GpuToken> flat_programs;
  vector<GpuToken> token_buffer;
  vector<double> gpu_fitness;
};

void rollback_pending_move(Node * candidate, vector<Node*> & candidate_nodes, PendingBatchMove & move, float backup_fitness) {
  for(int i = 0; i < move.changed_indices.size(); i++) {
    int changed_idx = move.changed_indices[i];
    Node * n = candidate_nodes[changed_idx];
    delete n->op;
    n->op = move.backup_ops[i];
    move.backup_ops[i] = NULL;
  }
  candidate->fitness = backup_fitness;
}

void clear_pending_move(PendingBatchMove & move) {
  discard_backup_ops(move.backup_ops);
  move.changed_indices.clear();
}

void evaluate_nodes_gpu_batch(vector<Node*> & nodes, vector<float> & out_fitness, GpuBatchEvalScratch & scratch) {
  int batch_size = (int) nodes.size();
  out_fitness.resize(batch_size);
  if (batch_size == 0) {
    return;
  }

  scratch.lengths.resize(batch_size);
  int max_program_len = 1;
  for(int i = 0; i < batch_size; i++) {
    int len = get_active_program_len(nodes[i]);
    scratch.lengths[i] = len;
    max_program_len = max(max_program_len, len);
  }

  size_t required_tokens = (size_t) batch_size * (size_t) max_program_len;
  if (scratch.flat_programs.size() < required_tokens) {
    scratch.flat_programs.resize(required_tokens);
  }
  if ((int) scratch.token_buffer.capacity() < max_program_len) {
    scratch.token_buffer.reserve(max_program_len);
  }

  for(int i = 0; i < batch_size; i++) {
    scratch.token_buffer.clear();
    serialize_active_tree_to_postfix_into(nodes[i], scratch.token_buffer);
    for(int j = 0; j < scratch.lengths[i]; j++) {
      scratch.flat_programs[(size_t) i * (size_t) max_program_len + (size_t) j] = scratch.token_buffer[j];
    }
  }

  scratch.gpu_fitness.resize(batch_size);
  evaluate_fitness_gpu_batch(
    *g::gpu_ctx,
    scratch.flat_programs.data(),
    scratch.lengths.data(),
    batch_size,
    max_program_len,
    scratch.gpu_fitness.data()
  );

  for(int i = 0; i < batch_size; i++) {
    g::fit_func->evaluations += 1;
    g::fit_func->node_evaluations += nodes[i]->get_num_nodes(true);
    g::num_gpu_evaluations += 1;

    float fitness = (float) scratch.gpu_fitness[i];
    if (!isfinite(fitness) || fitness < 0) {
      fitness = INF;
    }

    if (g::gpu_check_correctness) {
      double gpu_single = evaluate_fitness_gpu_single(
        *g::gpu_ctx,
        scratch.flat_programs.data() + (size_t) i * (size_t) max_program_len,
        scratch.lengths[i]
      );
      double gpu_tolerance = max(1e-6, abs(gpu_single) * 1e-8);
      if (isfinite(gpu_single) && isfinite(scratch.gpu_fitness[i]) && abs(gpu_single - scratch.gpu_fitness[i]) > gpu_tolerance) {
        throw runtime_error(
          "GPU batch/single fitness mismatch: single=" + to_string(gpu_single) +
          ", batch=" + to_string(scratch.gpu_fitness[i])
        );
      }

      float cpu_fitness = g::fit_func->get_fitness(nodes[i], g::fit_func->X_batch, g::fit_func->y_batch);
      float fitness_scale = max(abs(cpu_fitness), abs(fitness));
      float relative_tolerance = fitness_scale > (float) 1e6 ? (float) 1e-2 : (fitness_scale > (float) 1e5 ? (float) 5e-3 : (float) 1e-3);
      float tolerance = max((float) 1e-4, abs(cpu_fitness) * relative_tolerance);
      if (isfinite(cpu_fitness) && isfinite(fitness) && cpu_fitness < 1e6 && fitness < 1e6 && abs(cpu_fitness - fitness) > tolerance) {
        throw runtime_error(
          "CPU/GPU batch fitness mismatch: cpu=" + to_string(cpu_fitness) +
          ", gpu=" + to_string(fitness)
        );
      }
    }

    out_fitness[i] = fitness;
    nodes[i]->fitness = roundd(fitness, NUM_PRECISION + 2);
  }
}

void gomea_generation_gpu_batch(vector<Node*> & population, vector<vector<int>> & fos, GpuEvalContext & gpu_ctx, int pop_batch_size) {
  (void) gpu_ctx;
  if (!g::gpu_ctx) {
    throw runtime_error("GPU context is not initialized");
  }

  // This mode delays population commits until the current population batch is complete.
  // Donors inside the same batch do not see improvements made earlier in that batch.
  // It is therefore an approximate batched GOM variant.
  vector<int> order = Rng::rand_perm(population.size());
  pop_batch_size = max(1, pop_batch_size);
  GpuBatchEvalScratch eval_scratch;
  vector<float> new_fitnesses;
  vector<Node*> eval_nodes;
  vector<PendingBatchMove> pending_moves;

  for(int batch_start = 0; batch_start < order.size(); batch_start += pop_batch_size) {
    vector<vector<Node*>> population_nodes = collect_population_subtrees(population);
    int B = min(pop_batch_size, (int) order.size() - batch_start);
    vector<int> pop_indices(B);
    vector<Node*> candidates(B);
    vector<vector<Node*>> candidate_nodes(B);
    vector<float> candidate_fitness(B);
    vector<vector<int>> fos_orders(B);
    vector<bool> ever_improved(B, false);

    for(int i = 0; i < B; i++) {
      int pop_idx = order[batch_start + i];
      pop_indices[i] = pop_idx;
      candidates[i] = population[pop_idx]->clone();
      candidate_nodes[i] = candidates[i]->subtree();
      candidate_fitness[i] = population[pop_idx]->fitness;
      fos_orders[i] = Rng::rand_perm(fos.size());
    }

    for(int step = 0; step < fos.size(); step++) {
      eval_nodes.clear();
      pending_moves.clear();
      eval_nodes.reserve(B);
      pending_moves.reserve(B);

      for(int i = 0; i < B; i++) {
        const vector<int> & crossover_mask = fos[fos_orders[i][step]];
        bool change_is_meaningful = false;
        PendingBatchMove move;
        move.candidate_idx = i;
        move.backup_ops.reserve(crossover_mask.size());
        move.changed_indices.reserve(crossover_mask.size());

        int donor_idx = Rng::randi(population.size());
        vector<Node*> & donor_nodes = population_nodes[donor_idx];

        for(const int idx : crossover_mask) {
          if (candidate_nodes[i][idx]->op->sym() == donor_nodes[idx]->op->sym()) {
            if (g::cmut_prob <= 0 || g::cmut_temp <= 0 || donor_nodes[idx]->op->type() != OpType::otConst)
              continue;
          }

          Op * replaced_op = candidate_nodes[i][idx]->op;
          candidate_nodes[i][idx]->op = donor_nodes[idx]->op->clone();
          move.backup_ops.push_back(replaced_op);
          move.changed_indices.push_back(idx);
        }

        coeff_mut(candidates[i], false, &move.changed_indices, &move.backup_ops);

        for(int changed_idx : move.changed_indices) {
          if (!candidate_nodes[i][changed_idx]->is_intron()) {
            change_is_meaningful = true;
            break;
          }
        }

        if (change_is_meaningful) {
          g::num_meaningful_candidates += 1;
          eval_nodes.push_back(candidates[i]);
          pending_moves.push_back(move);
        } else {
          g::num_nonmeaningful_candidates += 1;
          clear_pending_move(move);
        }
      }

      evaluate_nodes_gpu_batch(eval_nodes, new_fitnesses, eval_scratch);

      for(int k = 0; k < pending_moves.size(); k++) {
        PendingBatchMove & move = pending_moves[k];
        int i = move.candidate_idx;
        float new_fitness = new_fitnesses[k];

        if (new_fitness > candidate_fitness[i]) {
          rollback_pending_move(candidates[i], candidate_nodes[i], move, candidate_fitness[i]);
          g::num_rejected_moves += 1;
        } else {
          if (new_fitness < candidate_fitness[i]) {
            candidate_fitness[i] = new_fitness;
            ever_improved[i] = true;
          }
          candidates[i]->fitness = roundd(candidate_fitness[i], NUM_PRECISION + 2);
          g::num_accepted_moves += 1;
        }
        clear_pending_move(move);
      }
    }

    for(int i = 0; i < B; i++) {
      if(g::tournament_size > 1 && !ever_improved[i]) {
        vector<Node*> tournament_candidates; tournament_candidates.reserve(g::tournament_size - 1);
        for(int j = 0; j < g::tournament_size - 1; j++) {
          tournament_candidates.push_back(population[Rng::randi(population.size())]);
        }
        tournament_candidates.push_back(candidates[i]);
        Node * winner = tournament(tournament_candidates, g::tournament_size);
        candidates[i]->clear();
        candidates[i] = winner;
      }

      int pop_idx = pop_indices[i];
      population[pop_idx]->clear();
      population[pop_idx] = candidates[i];
    }
  }
}

Node * append_linear_scaling(Node * tree) {
  // compute intercept and scaling coefficients, append them to the root
  Node * add_n, * mul_n, * slope_n, * interc_n;

  Vec p = tree->get_output(g::fit_func->X_train);

  pair<float,float> intc_slope = linear_scaling_coeffs(g::fit_func->y_train, p);
  
  if (intc_slope.second == 0){
    add_n = new Node(new Add());
    interc_n = new Node(new Const(intc_slope.first));
    add_n->append(interc_n);
    add_n->append(tree);
    return add_n;
  }

  mul_n = new Node(new Mul());
  slope_n = new Node(new Const(intc_slope.second));
  add_n = new Node(new Add());
  interc_n = new Node(new Const(intc_slope.first));
  mul_n->append(slope_n);
  mul_n->append(tree);
  add_n->append(interc_n);
  add_n->append(mul_n);

  // bring fitness info to new root
  add_n->fitness = tree->fitness;

  return add_n;

}



#endif
