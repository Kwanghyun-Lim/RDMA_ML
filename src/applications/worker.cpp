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
    if(argc < 11) {
        std::cerr << "Usage: " << argv[0]
		  << " <data_directory> <syn/mnist/rff> <sync/async> \
                       <alpha> <decay> <aggregate_batch_size> \
                       <num_epochs> <node_rank> <num_nodes> <num_trials>"
                  << std::endl;
        return 1;
    }
    
    // Initialize parameters
    std::string data_directory(argv[1]);
    std::string data(argv[2]);
    const double gamma = 0.0001;
    std::string algorithm(argv[3]);
    const double alpha = std::stod(argv[4]);
    double decay = std::stod(argv[5]);
    uint32_t aggregate_batch_size = std::stod(argv[6]);
    const uint32_t num_epochs = atoi(argv[7]);
    const uint32_t node_rank = atoi(argv[8]);
    const uint32_t num_nodes = atoi(argv[9]);
    const uint32_t num_trials = atoi(argv[10]);
    const size_t batch_size = aggregate_batch_size / (num_nodes - 1);
    openblas_set_num_threads(1);

    // Network setup
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
    
    // multiple trials for statistics
    for(uint32_t trial_num = 0; trial_num < num_trials; ++trial_num) {
      std::cout << "trial_num " << trial_num << std::endl;
      // Initialize m_log_reg, ml_sst, ml_stat, and worker for training
      log_reg::multinomial_log_reg m_log_reg([&]() {
	                                     return (utils::dataset)numpy::numpy_dataset(
					     data_directory + "/" + data,
					     (num_nodes - 1), node_rank - 1);
					     },
					     alpha, gamma, decay, batch_size);
      sst::MLSST ml_sst(std::vector<uint32_t>{0, node_rank}, node_rank,
			m_log_reg.get_model_size(), num_nodes);
      m_log_reg.set_model_mem((double*)std::addressof(ml_sst.model_or_gradient[0][0]));
      m_log_reg.push_back_to_grad_vec((double*)std::addressof(
				       ml_sst.model_or_gradient[1][0]));
      utils::ml_stat_t ml_stat(trial_num, num_nodes, num_epochs,
			       alpha, decay, batch_size, node_rank,
			       ml_sst, m_log_reg);
      worker::worker* wrk;
      // Destructor will be called but do nothing and the object will still be alive.
      // TODO: find a better method
      if(algorithm == "sync") {
	worker::sync_worker sync(m_log_reg, ml_sst, ml_stat, node_rank);
	wrk = &sync;
      } else if(algorithm == "async") {
	worker::async_worker async(m_log_reg, ml_sst, ml_stat, node_rank);
	wrk = &async;
      } else {
	std::cerr << "Wrong algorithm input: " << algorithm << std::endl;
	exit(1);
      }

      // Train
      wrk->train(num_epochs);
      
      std::cout << "trial_num " << trial_num << " done." << std::endl;
      std::cout << "Collecting results..." << std::endl;
      ml_stat.fout_op_time_log(false); // is_server == false
      std::cout << "Collecting results done." << std::endl;
      ml_sst.sync_with_members(); // barrier pair with server #5
      if (trial_num == num_trials - 1) {
	std::cout << "All trainings and loggings done." << std::endl;
	ml_sst.sync_with_members(); // barrier pair with server #6
      }
    }
}

worker::worker::worker(log_reg::multinomial_log_reg& m_log_reg,
		 sst::MLSST& ml_sst,
		 utils::ml_stat_t& ml_stat,
		 const uint32_t node_rank)
  : m_log_reg(m_log_reg), ml_sst(ml_sst), ml_stat(ml_stat), node_rank(node_rank) {
}

void worker::worker::train(const size_t num_epochs) {
  // virtual function
}

worker::sync_worker::sync_worker(log_reg::multinomial_log_reg& m_log_reg,
				   sst::MLSST& ml_sst,
				   utils::ml_stat_t& ml_stat,
				   const uint32_t node_rank)
  : worker(m_log_reg, ml_sst, ml_stat, node_rank) {
}

worker::sync_worker::~sync_worker() {
  std::cout << "sync_worker deconstructor does nothing." << std::endl;
}

void worker::sync_worker::train(const size_t num_epochs) {
  ml_sst.sync_with_members(); // barrier pair with server #1
  ml_stat.timer.set_start_time();
  const size_t num_batches = m_log_reg.get_num_batches();
  for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
    for(size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      ml_stat.timer.set_wait_start();
      while(ml_sst.round[0] < ml_sst.round[1]) {
      }
      ml_stat.timer.set_wait_end();
      
      ml_stat.timer.set_compute_start();
      m_log_reg.compute_gradient(batch_num, m_log_reg.get_model());
      ml_stat.timer.set_compute_end();
      ml_sst.round[1]++;
      
      ml_stat.timer.set_push_start();
      ml_sst.put_with_completion();
      ml_stat.timer.set_push_end();
    }
    ml_sst.sync_with_members(); // barrier pair with server #2
    // Between those two, server stores intermidiate models and parameters for statistics
    ml_sst.sync_with_members(); // barrier pair with server #3
  }
  ml_sst.sync_with_members(); // barrier pair with server #4
}

worker::async_worker::async_worker(log_reg::multinomial_log_reg& m_log_reg,
				   sst::MLSST& ml_sst,
				   utils::ml_stat_t& ml_stat,
				   const uint32_t node_rank)
  : worker(m_log_reg, ml_sst, ml_stat, node_rank) {
}

worker::async_worker::~async_worker() {
  std::cout << "async_worker deconstructor does nothing." << std::endl;
}

void worker::async_worker::train(const size_t num_epochs) {
  ml_sst.sync_with_members(); // barrier pair with server #1
  ml_stat.timer.set_start_time();
  const size_t num_batches = m_log_reg.get_num_batches();
  for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
    for(size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      ml_stat.timer.set_wait_start();
      while (ml_sst.last_round[0][node_rank] != ml_sst.round[1]) {
      }
      ml_stat.timer.set_wait_end();

      ml_stat.timer.set_compute_start();
      m_log_reg.compute_gradient(batch_num, m_log_reg.get_model());
      ml_stat.timer.set_compute_end();

      ml_sst.round[1]++;
      ml_stat.timer.set_push_start();
      ml_sst.put_with_completion();
      ml_stat.timer.set_push_end();
    }
    ml_sst.sync_with_members(); // barrier pair with server #2
    // Between those two, server stores intermidiate models and parameters for statistics
    ml_sst.sync_with_members(); // barrier pair with server #3
  }
  ml_sst.sync_with_members(); // barrier pair with server #4
}
