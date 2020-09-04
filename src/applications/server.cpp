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
#include "utils/numpy_reader.hpp"
#include "utils/ml_stat.hpp"

int main(int argc, char* argv[]) {
    if(argc < 11) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_directory> <syn/mnist/rff> <sync/async/fully_async> \
                       <log_reg/dnn> <alpha> <decay> <aggregate_batch_size> <num_epochs> \
                       <num_nodes> <num_trials>"
                  << std::endl;
        return 1;
    }

    // Initialize parameters
    std::string data_directory(argv[1]);
    std::string data(argv[2]);
    const double gamma = 0.0001;
    std::string algorithm(argv[3]);
    std::string ml_model_name(argv[4]);
    const double alpha = std::stod(argv[5]);
    double decay = std::stod(argv[6]);
    uint32_t aggregate_batch_size = atoi(argv[7]);
    const uint32_t num_epochs = atoi(argv[8]);
    const uint32_t node_rank = 0;
    const uint32_t num_nodes = atoi(argv[9]);
    const uint32_t num_trials = atoi(argv[10]);
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
      // Initialize ml_model, ml_sst, ml_stat, and worker for training
      ml_model::ml_model* ml_model;
      if (ml_model_name == "log_reg") {
	std::cout << "ml_model_name=" << ml_model_name << std::endl;
         ml_model = new ml_model::multinomial_log_reg(
     	     [&]() {return (utils::dataset)numpy::numpy_dataset(
		                       data_directory + "/" + data,
		                       num_nodes - 1, 0);},
                                       alpha, gamma, decay, batch_size);
      } else if (ml_model_name == "dnn") {
	const std::vector<uint32_t> layer_size_vec {784, 50, 10};
        const uint32_t num_layers = 3;
        ml_model = new ml_model::deep_neural_network(
	    layer_size_vec, num_layers,	     
            [&]() {return (utils::dataset)numpy::numpy_dataset(
		                       data_directory + "/" + data,
		                       num_nodes - 1, 0);},
                                       alpha, batch_size, false);
      } else {
	std::cerr << "Wrong ml_model_name input: " << ml_model_name << std::endl;
	exit(1);
      }
      
      sst::MLSST ml_sst(members, node_rank,
			ml_model->get_model_size(), num_nodes);
      ml_model->set_model_mem(
	      (double*)std::addressof(ml_sst.model_or_gradient[0][0]));
      if (ml_model_name == "log_reg") {
	utils::zero_arr(ml_model->get_model(), ml_model->get_model_size());
      } else if (ml_model_name == "dnn") {
	std::string model_full_path = data_directory + "/" + data + "/model_784-50-10.npy";
	ml_model->init_model(ml_model->get_model(), model_full_path);
      } else {
      	std::cerr << "Wrong ml_model_name input: " << ml_model_name << std::endl;
	exit(1);
      }
      
      for (uint row = 1; row < ml_sst.get_num_rows(); ++row) {
	ml_model->push_back_to_grad_vec(
	      (double*)std::addressof(ml_sst.model_or_gradient[row][0]));
      }
      utils::ml_stat_t ml_stat(trial_num, num_nodes, num_epochs,
			       alpha, decay, batch_size, node_rank,
			       ml_sst, ml_model);
      server::server* srv;
      if(algorithm == "sync") {
	srv = new server::sync_server(ml_model, ml_sst, ml_stat);
      } else if(algorithm == "async") {
	srv = new server::async_server(ml_model, ml_sst, ml_stat);
      } else if(algorithm == "fully_async") {
	srv = new server::fully_async_server(ml_model, ml_sst, ml_stat);
      } else {
	std::cerr << "Wrong algorithm input: " << algorithm << std::endl;
	exit(1);
      }

      // Train
      srv->train(num_epochs);
      
      std::cout << "trial_num " << trial_num << " done." << std::endl;
      std::cout << "Collecting results..." << std::endl;
      for(size_t epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
	ml_stat.collect_results(epoch_num, ml_model, ml_model_name);
      }
      ml_stat.print_results();
      ml_stat.fout_log_per_epoch();
      ml_stat.fout_analysis_per_epoch();
      ml_stat.fout_gradients_per_epoch();
      if (algorithm == "fully_async") {
	ml_stat.fout_op_time_log(true, true);
      } else {
	ml_stat.fout_op_time_log(true, false);
      }
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

server::server::server(ml_model::ml_model* ml_model,
				   sst::MLSST& ml_sst, utils::ml_stat_t& ml_stat)
  : ml_model(ml_model), ml_sst(ml_sst), ml_stat(ml_stat) {
}

void server::server::train(const size_t num_epochs) {
  // virtual function
}

server::sync_server::sync_server(ml_model::ml_model* ml_model,
				   sst::MLSST& ml_sst, utils::ml_stat_t& ml_stat)
  : server(ml_model, ml_sst, ml_stat) {
}

server::sync_server::~sync_server() {
  std::cout << "sync_server destructor does nothing." << std::endl;
}

void server::sync_server::train(const size_t num_epochs) {
  ml_sst.sync_with_members(); // barrier pair with server #1
  ml_stat.timer.set_start_time();
  const uint32_t num_nodes = ml_sst.get_num_rows();
  const size_t num_batches = ml_model->get_num_batches();
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
	    ml_model->update_model(row);
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
    ml_stat.set_epoch_parameters(epoch_num + 1, ml_model->get_model(),
				 ml_sst);
    ml_sst.sync_with_members(); // barrier pair with worker #3
  }
  ml_sst.sync_with_members(); // barrier pair with worker #4
}

server::async_server::async_server(ml_model::ml_model* ml_model,
				   sst::MLSST& ml_sst, utils::ml_stat_t& ml_stat)
  : server(ml_model, ml_sst, ml_stat) {
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
      const size_t num_batches = ml_model->get_num_batches();
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
	    ml_model->update_model(row);
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
    ml_stat.set_epoch_parameters(epoch_num + 1, ml_model->get_model(),
				 ml_sst);
    ml_sst.sync_with_members(); // barrier pair with worker #3
  }
  training = false;
  model_update_broadcast_thread.join();
  ml_sst.sync_with_members(); // barrier pair with worker #4
}

server::fully_async_server::fully_async_server(ml_model::ml_model* ml_model,
				   sst::MLSST& ml_sst, utils::ml_stat_t& ml_stat)
  : server(ml_model, ml_sst, ml_stat) {
}

server::fully_async_server::~fully_async_server() {
  std::cout << "fully_async_server destructor does nothing." << std::endl;
}

void server::fully_async_server::train(const size_t num_epochs) {
  std::atomic<bool> training = true;
  std::atomic<uint64_t> last_model_round = 0;
  auto model_update_loop =
    [this, num_epochs, &training, &last_model_round]() mutable {
      pthread_setname_np(pthread_self(), ("update"));
      const uint32_t num_nodes = ml_sst.get_num_rows();
      const size_t num_batches = ml_model->get_num_batches();
      std::vector<uint64_t> last_round(num_nodes, 0);
      while(training) {
	for (uint row = 1; row < num_nodes; ++row) {
	  // If new gradients arrived, update model
	  if(last_round[row] < num_epochs * num_batches &&
	     ml_sst.round[row] > last_round[row]) {
	    // Counts # of lost gradients.
	    if(ml_sst.round[row] - last_round[row] > 1) {
	      ml_stat.num_lost_gradients_per_node[row] +=
		ml_sst.round[row] - last_round[row] - 1;
	    }
	    last_round[row] = ml_sst.round[row];
	    ml_stat.timer.set_compute_start();
	    ml_model->update_model(row);
	    ml_stat.timer.set_compute_end(FRONT_END_THREAD);
	    ml_sst.round[0]++;
	  }
	}
      }
    };

  auto model_broadcast_loop =
    [this, &training, &last_model_round]() mutable {
      pthread_setname_np(pthread_self(), ("broadcast"));
      while(training) {
	if(ml_sst.round[0] > last_model_round) {
	  ml_stat.timer.set_push_start();
	  ml_sst.put_with_completion();
	  ml_stat.timer.set_push_end(BACK_END_THREAD);
	  ml_stat.num_broadcasts++;
	  // TODO: count the model mixing ratio here.
	  last_model_round = ml_sst.round[0];
	}
      }
    };
  
  std::thread model_update_thread = std::thread(model_update_loop);
  std::thread model_broadcast_thread = std::thread(model_broadcast_loop);
  ml_sst.sync_with_members(); // barrier pair with worker #1
  ml_stat.timer.set_start_time();
  for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
    // Between the barrier #1 above and #2 below, workers train.
    ml_stat.timer.set_train_start();
    ml_sst.sync_with_members(); // barrier pair with worker #2
    ml_stat.timer.set_train_end();
    ml_stat.set_epoch_parameters(epoch_num + 1, ml_model->get_model(),
				 ml_sst);
    ml_sst.sync_with_members(); // barrier pair with worker #3
  }
  training = false;
  model_update_thread.join();
  model_broadcast_thread.join();
  ml_sst.sync_with_members(); // barrier pair with worker #4
}
