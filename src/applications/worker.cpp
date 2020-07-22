#include <bits/stdc++.h>
#include <cblas.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <utility>
#include <unistd.h>
#include <queue>
#include <map>
#include <sys/time.h>
#include <thread>

#include "worker.hpp"
#include "utils/numpy_reader.hpp"

int main(int argc, char* argv[]) {
    if(argc < 10) {
        std::cerr << "Usage: " << argv[0]
		  << " <data_directory> <syn/mnist/rff> <alpha> <decay> <aggregate_batch_size> <num_epochs> <node_rank> <num_nodes> <num_trials>"
                  << std::endl;
        return 1;
    }
    std::string data_directory(argv[1]);
    std::string data(argv[2]);
    const double gamma = 0.0001;
    const double alpha = std::stod(argv[3]);
    double decay = std::stod(argv[4]);
    uint32_t aggregate_batch_size = std::stod(argv[5]);
    const uint32_t num_epochs = atoi(argv[6]);
    const uint32_t node_rank = atoi(argv[7]);
    const uint32_t num_nodes = atoi(argv[8]);
    const uint32_t num_trials = atoi(argv[9]);
    openblas_set_num_threads(1);
    
    std::map<uint32_t, std::string> ip_addrs_static;
    ip_addrs_static[0] = "192.168.99.16";
    ip_addrs_static[1] = "192.168.99.17";
    ip_addrs_static[2] = "192.168.99.18";
    ip_addrs_static[3] = "192.168.99.20";
    ip_addrs_static[4] = "192.168.99.30";
    ip_addrs_static[5] = "192.168.99.31";
    ip_addrs_static[6] = "192.168.99.32";
    ip_addrs_static[7] = "192.168.99.24";
    ip_addrs_static[8] = "192.168.99.25";
    ip_addrs_static[9] = "192.168.99.27";
    ip_addrs_static[10] = "192.168.99.23";
    ip_addrs_static[11] = "192.168.99.105";
    ip_addrs_static[12] = "192.168.99.29";
    ip_addrs_static[13] = "192.168.99.26";
    ip_addrs_static[14] = "192.168.99.106";
    ip_addrs_static[15] = "192.168.99.28";

    std::map<uint32_t, std::string> ip_addrs;
    ip_addrs[0] = ip_addrs_static.at(0);
    ip_addrs[node_rank] = ip_addrs_static.at(node_rank);
    sst::verbs_initialize(ip_addrs, node_rank);
    
    const size_t batch_size = aggregate_batch_size / (num_nodes - 1);

    for(uint32_t trial_num = 0; trial_num < num_trials; ++trial_num) {
      std::cout << "trial_num " << trial_num << std::endl;
      log_reg::multinomial_log_reg m_log_reg(
					     [&]() {
					       return (utils::dataset)numpy::numpy_dataset(
											   data_directory + "/" + data, (num_nodes - 1), node_rank - 1);
					     },
					     alpha, gamma, decay, batch_size);

      sst::MLSST ml_sst(std::vector<uint32_t>{0, node_rank},
			node_rank, m_log_reg.get_model_size(), num_nodes);

      m_log_reg.set_model_mem((double*)std::addressof(ml_sst.model_or_gradient[0][0]));
      m_log_reg.push_back_to_grad_vec((double*)std::addressof(ml_sst.model_or_gradient[1][0]));
      
      worker::async_worker worker(m_log_reg, ml_sst, node_rank);
      worker.train(num_epochs);
      
      if (trial_num == num_trials - 1) {
	ml_sst.sync_with_members();
      }
    }
}

worker::async_worker::async_worker(log_reg::multinomial_log_reg& m_log_reg,
					sst::MLSST& ml_sst, const uint32_t node_rank)
  : m_log_reg(m_log_reg), ml_sst(ml_sst), node_rank(node_rank) {
}

void worker::async_worker::train(const size_t num_epochs) {
  ml_sst.sync_with_members();
  std::queue<std::pair<std::pair<uint64_t, uint64_t>, uint32_t>> q;
  struct timespec start_time, end_time;
  uint64_t relay_start, relay_end, grad_start, grad_end, push_start, push_end, wait_start, wait_end;
  uint64_t relay_total=0, grad_total=0, push_total=0, wait_total=0;
  clock_gettime(CLOCK_REALTIME, &start_time);
  
  const size_t num_batches = m_log_reg.get_num_batches();
  for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
    for(size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      clock_gettime(CLOCK_REALTIME, &end_time);
      wait_start = (end_time.tv_sec - start_time.tv_sec) * 1e9
      	    + (end_time.tv_nsec - start_time.tv_nsec);

      while (ml_sst.last_round[0][node_rank] != ml_sst.round[1]) {
      }
      
      clock_gettime(CLOCK_REALTIME, &end_time);
      wait_end = (end_time.tv_sec - start_time.tv_sec) * 1e9
      	    + (end_time.tv_nsec - start_time.tv_nsec);
      q.push({{wait_start, wait_end}, 0});
      wait_total += wait_end - wait_start;

      
      clock_gettime(CLOCK_REALTIME, &end_time);
      grad_start = (end_time.tv_sec - start_time.tv_sec) * 1e9
	+ (end_time.tv_nsec - start_time.tv_nsec);

      m_log_reg.compute_gradient(batch_num, m_log_reg.get_model());

      clock_gettime(CLOCK_REALTIME, &end_time);
      grad_end = (end_time.tv_sec - start_time.tv_sec) * 1e9
	+ (end_time.tv_nsec - start_time.tv_nsec);
      q.push({{grad_start, grad_end}, 2});
      grad_total += grad_end - grad_start;

      clock_gettime(CLOCK_REALTIME, &end_time);
      push_start = (end_time.tv_sec - start_time.tv_sec) * 1e9
	+ (end_time.tv_nsec - start_time.tv_nsec);
      
      ml_sst.round[1]++;
      ml_sst.put_with_completion();

      clock_gettime(CLOCK_REALTIME, &end_time);
      push_end = (end_time.tv_sec - start_time.tv_sec) * 1e9
	    + (end_time.tv_nsec - start_time.tv_nsec);
      
      q.push({{push_start, push_end}, 3});
      push_total += push_end - push_start;
    }

    ml_sst.sync_with_members();
    ml_sst.sync_with_members();
  }
  std::ofstream worker_log_file("worker" + std::to_string(node_rank) + ".log", std::ofstream::trunc);
  while (!q.empty()) {
    std::pair<std::pair<uint64_t, uint64_t>, uint32_t> item = q.front();
    q.pop();
    worker_log_file <<  item.first.first << " " << item.first.second << " " << item.second << std::endl;
  }
  std::ofstream worker_stat_file("worker" + std::to_string(node_rank) + ".stat", std::ofstream::trunc);
  uint64_t total = relay_total + grad_total + push_total + wait_total;
  worker_stat_file <<  "relay" << " " << "grad" << " " << "push" <<  " " <<  "wait" << " " << "total" << std::endl;
  worker_stat_file <<  relay_total << " " << grad_total << " " << push_total << " " << wait_total << " " << total << std::endl;
  worker_stat_file <<  float(relay_total)/total << " " << float(grad_total)/total << " "
		   << float(push_total)/total << " " << float(wait_total)/total << " " << total << std::endl;

  ml_sst.sync_with_members();
}
