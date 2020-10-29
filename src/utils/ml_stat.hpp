#pragma once

#include "coordinator/ml_sst.hpp"
#include "ml_model.hpp"
#include "log_reg.hpp"
#include "dnn.hpp"
#include <atomic>
#include <chrono>
#include <queue>
#include <string>

#define COMPUTE_THREAD 0
#define NETWORK_THREAD 1
namespace utils {
class ml_timer_t {
public:
  ml_timer_t();
  void set_start_time();
  void set_wait_start();  
  void set_wait_end();
  void set_compute_start();
  void set_compute_end();
  void set_compute_end(int thread);
  void set_push_start();
  void set_push_end();
  void set_push_end(int thread);
  void set_train_start();
  void set_train_end();
  void set_wait_end(int thread);

  struct timespec start_time, end_time;
  uint64_t relay_start, relay_end;
  uint64_t compute_start, compute_end;
  uint64_t push_start, push_end;
  uint64_t wait_start, wait_end;
  uint64_t relay_total, compute_total, push_total, wait_total;
  struct timespec train_start_time, train_end_time;
  double train_time_taken;
  std::queue<std::pair<std::pair<uint64_t, uint64_t>, uint32_t>> op_time_log_q;
  std::queue<std::pair<std::pair<uint64_t, uint64_t>, uint32_t>> op_time_log_q_compute;
  std::queue<std::pair<std::pair<uint64_t, uint64_t>, uint32_t>> op_time_log_q_network;
};
  
class ml_stat_t {
public:
    // This dummy constructor just below is for std and err in ml_stats_t.
    ml_stat_t(uint32_t num_nodes, uint32_t num_epochs);
    ml_stat_t(uint32_t trial_num, uint32_t num_nodes,
	       uint32_t num_epochs, double alpha,
	      double decay, double batch_size, const uint32_t node_rank,
              const sst::MLSST& ml_sst,
              ml_model::ml_model* ml_model);

    void set_epoch_parameters(uint32_t epoch_num, double* model, const sst::MLSST& ml_sst);
    void collect_results(uint32_t epoch_num, ml_model::ml_model* ml_model, std::string ml_model_name);
    void print_results();
    void fout_log_per_epoch();
    void fout_analysis_per_epoch();
    void fout_gradients_per_epoch();
    void fout_op_time_log(bool is_server, bool is_fully_async);
  
    uint32_t trial_num;
    uint32_t num_nodes;
    uint32_t num_epochs;
    double alpha;
    double decay;
    double batch_size;
    const uint32_t node_rank;
    size_t model_size;

    std::vector<double*> intermediate_models;
    std::vector<double> cumulative_num_broadcasts;
    std::vector<double> num_model_updates;
    // The first row of num_gradients_received is used for the sum of each worker's num_pushed gradients.
    std::vector<std::vector<double>> num_gradients_received;
    std::vector<std::vector<double>> num_lost_gradients;
    std::vector<uint64_t> num_lost_gradients_per_node;
  
    std::vector<double> cumulative_time;
    std::vector<double> training_error;
    std::vector<double> test_error;
    std::vector<double> loss_gap;
    std::vector<double> dist_to_opt;
    std::vector<double> grad_norm;

    ml_timer_t timer;
    uint64_t num_broadcasts;
};

class ml_stats_t {
public:
  ml_stats_t(uint32_t num_nodes, uint32_t num_epochs);
  void push_back(ml_stat_t ml_stat);
  void compute_mean();
  void compute_std();
  void compute_err();
  void grid_search_helper(std::string target_dir);
  void fout_log_mean_per_epoch();
  void fout_log_err_per_epoch();
  void fout_analysis_mean_per_epoch();
  void fout_analysis_err_per_epoch();
  void fout_gradients_mean_per_epoch();
  void fout_gradients_err_per_epoch();
  
  std::vector<ml_stat_t> ml_stat_vec;
  ml_stat_t mean;
  ml_stat_t std;
  ml_stat_t err;
  
private:
  double get_loss_opt(std::string target_dir);
};
}  // namespace utils
