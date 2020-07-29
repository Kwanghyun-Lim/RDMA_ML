#include <bits/stdc++.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <string>
#include <sys/time.h>
#include <thread>
#include <vector>

#include "server.hpp"
#include "coordinator/ml_sst.hpp"
#include "coordinator/tcp.hpp"
#include "coordinator/verbs.h"
#include "log_reg.hpp"
#include "utils/numpy_reader.hpp"
#include "utils/ml_stat.hpp"

int main(int argc, char* argv[]) {
    if(argc < 10) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_directory> <syn/mnist/rff> <sync/async> \
                       <alpha> <decay> <aggregate_batch_size> <num_epochs> \
                       <num_nodes> <num_trials>"
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
    const uint32_t node_rank = 0;
    const uint32_t num_nodes = atoi(argv[8]);
    const uint32_t num_trials = atoi(argv[9]);
    const size_t batch_size = aggregate_batch_size / (num_nodes - 1);

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
    ip_addrs_static[12] = "192.168.99.28";
    ip_addrs_static[13] = "192.168.99.29";
    ip_addrs_static[14] = "192.168.99.26";
    ip_addrs_static[15] = "192.168.99.106";

    std::map<uint32_t, std::string> ip_addrs;
    for(uint32_t i = 0; i < num_nodes; ++i) {
        ip_addrs[i] = ip_addrs_static.at(i);
    }
    sst::verbs_initialize(ip_addrs, 0);
    std::vector<uint32_t> members(num_nodes);
    std::iota(members.begin(), members.end(), 0);

    // multiple trials for statistics
    utils::ml_stats_t ml_stats(num_nodes, num_epochs);
    for(uint32_t trial_num = 0; trial_num < num_trials; ++trial_num) {
      std::cout << "trial_num " << trial_num << std::endl;
      // Initialize m_log_reg, ml_sst, ml_stat, and worker for training
      log_reg::multinomial_log_reg m_log_reg(
	 [&]() {return (utils::dataset)numpy::numpy_dataset(
		                       data_directory + "/" + data,
		                       (num_nodes - 1), node_rank);},
                                       alpha, gamma, decay, batch_size);
      std::cout << "before ml_sst" << std::endl;
      sst::MLSST ml_sst(members, node_rank,
			m_log_reg.get_model_size(), num_nodes);
      std::cout << "after ml_sst" << std::endl;
      m_log_reg.set_model_mem(
	      (double*)std::addressof(ml_sst.model_or_gradient[0][0]));
      for (uint row = 1; row < ml_sst.get_num_rows(); ++row) {
	m_log_reg.push_back_to_grad_vec(
	      (double*)std::addressof(ml_sst.model_or_gradient[row][0]));
      }
      utils::ml_stat_t ml_stat(trial_num, num_nodes, num_epochs,
			       alpha, decay, batch_size, node_rank,
			       ml_sst, m_log_reg);
      server::server* srv;
      if(algorithm == "sync") {
	srv = new server::sync_server(m_log_reg, ml_sst, ml_stat);
      } else if(algorithm == "async") {
	srv = new server::async_server(m_log_reg, ml_sst, ml_stat);
      } else {
	std::cerr << "Wrong algorithm input: " << algorithm << std::endl;
	exit(1);
      }

      // Train
      srv->train(num_epochs);
      
      std::cout << "trial_num " << trial_num << " done." << std::endl;
      std::cout << "Collecting results..." << std::endl;
      for(size_t epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
	ml_stat.collect_results(epoch_num, m_log_reg);
      }
      ml_stat.print_results();
      ml_stat.fout_log_per_epoch();
      ml_stat.fout_analysis_per_epoch();
      ml_stat.fout_gradients_per_epoch();
      ml_stat.fout_op_time_log(true); // is_server == true
      ml_stats.push_back(ml_stat);
      std::cout << "Collecting results done." << std::endl;
      ml_sst.sync_with_members(); // barrier pair with worker #5

      if (trial_num == num_trials - 1) {
	// Compute statistics, log files, and store model
	std::cout << "Compute statistics..." << std::endl;
	ml_stats.compute_mean();
	ml_stats.compute_std();
	ml_stats.compute_err();
	ml_stats.fout_log_mean_per_epoch();
	ml_stats.fout_log_err_per_epoch();
	ml_stats.fout_analysis_mean_per_epoch();
	ml_stats.fout_analysis_err_per_epoch();
	ml_stats.fout_gradients_mean_per_epoch();
	ml_stats.fout_gradients_err_per_epoch();
	std::string target_dir = data_directory + "/" + data;
	ml_stats.grid_search_helper(target_dir);
	std::cout << "Compute statistics done." << std::endl;
	std::cout << "All trainings and loggings done." << std::endl;
	ml_sst.sync_with_members(); // barrier pair with worker #6
      }
    }
}

server::server::server(log_reg::multinomial_log_reg& m_log_reg,
				   sst::MLSST& ml_sst, utils::ml_stat_t& ml_stat)
  : m_log_reg(m_log_reg), ml_sst(ml_sst), ml_stat(ml_stat) {
}

void server::server::train(const size_t num_epochs) {
  // virtual function
}

server::sync_server::sync_server(log_reg::multinomial_log_reg& m_log_reg,
				   sst::MLSST& ml_sst, utils::ml_stat_t& ml_stat)
  : server(m_log_reg, ml_sst, ml_stat) {
}

server::sync_server::~sync_server() {
  std::cout << "sync_server destructor does nothing." << std::endl;
}

void server::sync_server::train(const size_t num_epochs) {
  ml_sst.sync_with_members(); // barrier pair with server #1
  ml_stat.timer.set_start_time();
  const uint32_t num_nodes = ml_sst.get_num_rows();
  const size_t num_batches = m_log_reg.get_num_batches();
  std::vector<bool> done(num_nodes, false);
  std::vector<uint32_t> receivers;
  for (int i = 1; i < num_nodes; ++i) {
    receivers.push_back(i);
  }

  for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
    ml_stat.timer.set_train_start();
    for(size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      std::fill(done.begin(), done.end(), false);
      uint32_t num_done = 0;
      while(num_done < num_nodes - 1) {
	for(uint32_t row = 1; row < num_nodes; ++row) {
	  if(!done[row] && ml_sst.round[row] > ml_sst.round[0]) {
	    ml_stat.timer.set_compute_start();
	    m_log_reg.update_model(row);
	    ml_stat.timer.set_compute_end();
	    done[row] = true;
	    num_done++;
	  }
	}
      }
      ml_sst.round[0]++;
      ml_stat.timer.set_push_start();
      ml_sst.put_with_completion();
      ml_stat.timer.set_push_end();
      // ml_sst.put_with_completion(receivers, ALL_FIELDS);
    }
    ml_stat.timer.set_train_end();
    ml_sst.sync_with_members(); // barrier pair with worker #2
    ml_stat.set_epoch_parameters(epoch_num + 1, m_log_reg.get_model(),
				 ml_sst);
    ml_sst.sync_with_members(); // barrier pair with worker #3
  }
  ml_sst.sync_with_members(); // barrier pair with worker #4
}

server::async_server::async_server(log_reg::multinomial_log_reg& m_log_reg,
				   sst::MLSST& ml_sst, utils::ml_stat_t& ml_stat)
  : server(m_log_reg, ml_sst, ml_stat) {
}

server::async_server::~async_server() {
  std::cout << "async_server destructor does nothing." << std::endl;
}

void server::async_server::train(const size_t num_epochs) {
  std::atomic<bool> training = true;
  auto model_update_broadcast_loop =
    [this, num_epochs, &training]() mutable {
      pthread_setname_np(pthread_self(), ("update"));
      const uint32_t num_nodes = ml_sst.get_num_rows();
      const size_t num_batches = m_log_reg.get_num_batches();
      std::vector<uint32_t> receivers;
      while(training) {
	for (uint row = 1; row < num_nodes; ++row) {
	  // If new gradients arrived, update model
	  if(ml_sst.last_round[0][row] < num_epochs * num_batches &&
	     ml_sst.round[row] > ml_sst.last_round[0][row]) {
	    // std::cout << "updated: row=" << row << "ml_sst.round=" << ml_sst.round[row] << std::endl;
	    // Counts # of lost gradients.
	    if(ml_sst.round[row] - ml_sst.last_round[0][row] > 1) {
	      ml_stat.num_lost_gradients_per_node[row] +=
		ml_sst.round[row] - ml_sst.last_round[0][row] - 1;
	    }
	    ml_sst.last_round[0][row] = ml_sst.round[row];
	    ml_stat.timer.set_compute_start();
	    m_log_reg.update_model(row);
	    ml_stat.timer.set_compute_end();
	    receivers.push_back(row);
	    ml_sst.round[0]++;
	  }
	}
	
	if(!receivers.empty()) {
	  ml_stat.timer.set_push_start();
	  // std::cout << "receivers= ";
	  // for(int i = 0; i < receivers.size(); ++i) {
	  //   std::cout << i << " ";
	  // }
	  // std::cout << std::endl;
	  // std::cout << "before put_with_completion(receivers)" << std::endl;
	  ml_sst.put_with_completion(receivers);
	  // std::cout << "after put_with_completion(receivers)" << std::endl;
	  ml_stat.timer.set_push_end();
	  ml_stat.num_broadcasts++;
	  receivers.clear();
	}
      }
    };
  std::thread model_update_broadcast_thread = std::thread(model_update_broadcast_loop);
  ml_sst.sync_with_members(); // barrier pair with worker #1
  ml_stat.timer.set_start_time();
  for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
    // Between the barrier #1 above and #2 below, workers train.
    ml_stat.timer.set_train_start();
    ml_sst.sync_with_members(); // barrier pair with worker #2
    ml_stat.timer.set_train_end();
    ml_stat.set_epoch_parameters(epoch_num + 1, m_log_reg.get_model(),
				 ml_sst);
    ml_sst.sync_with_members(); // barrier pair with worker #3
  }
  training = false;
  model_update_broadcast_thread.join();
  ml_sst.sync_with_members(); // barrier pair with worker #4
}
