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

#ifndef NODE_TYPE
#define SERVER false
#define WORKER true
#endif

#ifndef LOG_TYPE
#define SERVER_LOG true
#define WORKER_LOG false

#define FULLY_ASYNC_LOG true
#define NOT_FULLY_ASYNC_LOG false
#endif

#ifndef SGD_TYPE
#define SYNC 0
#define ASYNC 1
#define FULLY_ASYNC 2
#endif

int main(int argc, char* argv[]) {
    if(argc < 11) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_directory> <syn/mnist/rff> \
                       <sync/async/fully_async> \
                       <log_reg/dnn> \
                       <alpha> <decay> \
                       <aggregate_batch_size> <num_epochs> \
                       <num_nodes> <num_trials>"
                  << std::endl;
        return 1;
    }

    // Initialize parameters
    std::string data_directory(argv[1]);
    std::string data(argv[2]);
    const double gamma = 0.0001;
    std::string sgd_type(argv[3]);
    std::string ml_model_name(argv[4]);
    const double alpha = std::stod(argv[5]);
    double decay = std::stod(argv[6]);
    uint32_t aggregate_batch_size = atoi(argv[7]);
    const uint32_t num_epochs = atoi(argv[8]);
    const uint32_t node_rank = 0;
    const uint32_t num_nodes = atoi(argv[9]);
    const uint32_t num_trials = atoi(argv[10]);
    const size_t batch_size = aggregate_batch_size / (num_nodes - 1);
    bool has_buffers;
    if(sgd_type == "fully_async") {
      has_buffers = true;
    } else {
      has_buffers = false;
    }
    std::string init_model_file = data_directory + "/" + data
	                            + "/model_784-128-10.npy";

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
	     alpha, gamma, decay, batch_size,
	     std::string(""), has_buffers, SERVER);
      } else if (ml_model_name == "dnn") {
	const std::vector<uint32_t> layer_size_vec {784, 128, 10};
        const uint32_t num_layers = 3;
        ml_model = new ml_model::deep_neural_network(
	    layer_size_vec, num_layers,	     
            [&]() {return (utils::dataset)numpy::numpy_dataset(
		                       data_directory + "/" + data,
		                       num_nodes - 1, 0);},
	    alpha, batch_size, 
	    init_model_file,
	    has_buffers, SERVER);
      } else {
	std::cerr << "Wrong ml_model_name input: "
		  << ml_model_name << std::endl;
	exit(1);
      }

      std::cout << "members= " << std::endl;
      for (auto member : members) {
	std::cout << member << " ";
      }
      std::cout << std::endl;
      
      sst::MLSST* ml_sst;
      if(sgd_type == "sync" || sgd_type == "async") {
	ml_sst = new sst::MLSST_NOBUF(members, node_rank,
				      ml_model->get_model_size(),
				      num_nodes, ml_model);
      } else if(sgd_type == "fully_async") {
	ml_sst = new sst::MLSST_BUF(members, node_rank,
				    ml_model->get_model_size(),
				    num_nodes, ml_model);
      } else {
      	std::cerr << "Wrong sgd_type input: "
		  << sgd_type << std::endl;
	exit(1);
      }
      std::cout << "server #1" << std::endl;
      ml_sst->connect_ml_model_to_ml_sst();
      std::cout << "server #2" << std::endl;

      utils::ml_stat_t ml_stat(trial_num, num_nodes, num_epochs,
			       alpha, decay, batch_size, node_rank,
			       ml_sst, ml_model);
      std::cout << "server #3" << std::endl;
      server::server* srv;
      if(sgd_type == "sync") {
	srv = new server::sync_server(ml_model, ml_sst, ml_stat);
      } else if(sgd_type == "async") {
	srv = new server::async_server(ml_model, ml_sst, ml_stat);
      } else if(sgd_type == "fully_async") {
	srv = new server::fully_async_server(ml_model, ml_sst, ml_stat);
      } else {
	std::cerr << "Wrong sgd_type input: "
		  << sgd_type << std::endl;
	exit(1);
      }
      std::cout << "server #4" << std::endl;

      // Train
      srv->train(num_epochs);
      std::cout << "server #5" << std::endl;
      
      std::cout << "trial_num " << trial_num << " done." << std::endl;
      std::cout << "Collecting results..." << std::endl;
      for(size_t epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
	ml_stat.collect_results(epoch_num, ml_model, ml_model_name);
      }
      ml_stat.print_results();
      ml_stat.fout_log_per_epoch();
      ml_stat.fout_analysis_per_epoch();
      ml_stat.fout_gradients_per_epoch();
      if (sgd_type == "fully_async") {
	ml_stat.fout_op_time_log(SERVER_LOG, FULLY_ASYNC_LOG);
      } else {
	ml_stat.fout_op_time_log(SERVER_LOG, NOT_FULLY_ASYNC_LOG);
      }
      ml_stats.push_back(ml_stat);
      std::cout << "Collecting results done." << std::endl;
      ml_sst->sync_barrier_with_server_and_workers(); // barrier #5

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
	ml_sst->sync_barrier_with_server_and_workers(); // barrier #6
      }
    }
}

server::server::server(ml_model::ml_model* ml_model,
		       sst::MLSST* ml_sst,
		       utils::ml_stat_t& ml_stat)
  : ml_model(ml_model),
    ml_sst(ml_sst),
    ml_stat(ml_stat) {
}

// virtual function
void server::server::train(const size_t num_epochs) {
}

server::sync_server::sync_server(ml_model::ml_model* ml_model,
				 sst::MLSST* ml_sst,
				 utils::ml_stat_t& ml_stat)
  : server(ml_model, ml_sst, ml_stat) {
}

server::sync_server::~sync_server() {
  std::cout << "sync_server destructor does nothing." << std::endl;
}

void server::sync_server::train(const size_t num_epochs) {
  ml_sst->sync_barrier_with_server_and_workers(); // barrier #1
  ml_stat.timer.set_start_time();
  const uint32_t num_nodes = ml_sst->get_num_nodes();
  const size_t num_batches = ml_model->get_num_batches();
  const uint ml_sst_col = 1;
  std::vector<bool> done(num_nodes, false);

  for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
    ml_stat.timer.set_train_start();
    for(size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      std::fill(done.begin(), done.end(), false);
      uint32_t num_done = 0;
      while(num_done < num_nodes - 1) {
	for(uint32_t node_id = 1; node_id < num_nodes; ++node_id) {
	  if(!done[node_id] &&
	     ml_sst->has_new_gradient_received(node_id, SYNC)) {
	    ml_stat.timer.set_compute_start();
	    ml_model->update_model(node_id, ml_sst_col);
	    ml_stat.timer.set_compute_end();
	    done[node_id] = true;
	    num_done++;
	  }
	}
      }
      ml_sst->increment_model_version_num();
      ml_stat.timer.set_push_start();
      ml_sst->broadcast_new_model_and_version_num();
      ml_stat.timer.set_push_end();
    }
    ml_stat.timer.set_train_end();
    ml_sst->sync_barrier_with_server_and_workers(); // barrier #2
    ml_stat.set_epoch_parameters(epoch_num + 1,
				 ml_model->get_model(),
				 ml_sst);
    ml_sst->sync_barrier_with_server_and_workers(); // barrier #3
  }
  ml_sst->sync_barrier_with_server_and_workers(); // barrier #4
}

server::async_server::async_server(ml_model::ml_model* ml_model,
				   sst::MLSST* ml_sst,
				   utils::ml_stat_t& ml_stat)
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
      const uint32_t num_nodes = ml_sst->get_num_nodes();
      const size_t num_batches = ml_model->get_num_batches();
      const uint ml_sst_col = 1;
      std::vector<uint32_t> model_receivers;
      while(training) {
	for (uint node_id = 1; node_id < num_nodes; ++node_id) {
	  if(ml_sst->has_new_gradient_received(node_id, ASYNC)) {
	    ml_stat.log_num_lost_gradients(node_id, ASYNC, ml_sst);
	    ml_sst->mark_new_gradient_consumed(node_id, ASYNC);
	    ml_stat.timer.set_compute_start();
	    ml_model->update_model(node_id, ml_sst_col);
	    ml_stat.timer.set_compute_end();
	    model_receivers.push_back(node_id);
	    ml_sst->increment_model_version_num();
	  }
	}
	
	if(!model_receivers.empty()) {
	  ml_stat.timer.set_push_start();
	  ml_sst->push_new_model_and_version_num(model_receivers);
	  ml_stat.timer.set_push_end();
	  // will be interpreted as # pushes not broadcasts
	  ml_stat.num_broadcasts++; 
	  model_receivers.clear();
	}
      }
    };
  std::thread model_update_broadcast_thread
    = std::thread(model_update_broadcast_loop);
  ml_sst->sync_barrier_with_server_and_workers(); //#1 training start
  ml_stat.timer.set_start_time();
  for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
    ml_stat.timer.set_train_start();
    // Workers are training during this time frame here.
    //#2 training end per epoch
    ml_sst->sync_barrier_with_server_and_workers(); 
    ml_stat.timer.set_train_end();
    ml_stat.set_epoch_parameters(epoch_num + 1,
				 ml_model->get_model(),
				 ml_sst);
    //#3 statistics per epoch
    ml_sst->sync_barrier_with_server_and_workers(); 
  }
  training = false;
  model_update_broadcast_thread.join();
  ml_sst->sync_barrier_with_server_and_workers(); // #4 trainig end
}

server::fully_async_server::fully_async_server(
				   ml_model::ml_model* ml_model,
				   sst::MLSST* ml_sst,
				   utils::ml_stat_t& ml_stat)
  : server(ml_model, ml_sst, ml_stat) {
}

server::fully_async_server::~fully_async_server() {
  std::cout << "fully_async_server destructor does nothing." << std::endl;
}

void server::fully_async_server::train(const size_t num_epochs) {
  // std::atomic<bool> training = true;
  // std::atomic<uint64_t> last_model_round = 0;
  // auto model_update_loop =
  //   [this, num_epochs, &training, &last_model_round]() mutable {
  //     pthread_setname_np(pthread_self(), ("update"));
  //     const uint32_t num_nodes = ml_sst->get_num_nodes();
  //     const size_t num_batches = ml_model->get_num_batches();
  //     std::vector<uint64_t> last_round(num_nodes, 0);
  //     while(training) {
  // 	for (uint row = 1; row < num_nodes; ++row) {
  // 	  // If new gradients arrived, update model
  // 	  if(last_round[row] < num_epochs * num_batches &&
  // 	     ml_sst.round[row] > last_round[row]) {
  // 	    // std::cout << "updated: row=" << row << "ml_sst.round=" << ml_sst.round[row] << std::endl;
  // 	    // Counts # of lost gradients.
  // 	    if(ml_sst.round[row] - last_round[row] > 1) {
  // 	      ml_stat.num_lost_gradients_per_node[row] +=
  // 		ml_sst.round[row] - last_round[row] - 1;
  // 	    }
	    
  // 	    last_round[row] = ml_sst.round[row];
  // 	    ml_stat.timer.set_compute_start();
  // 	    ml_model->update_model(row);
  // 	    ml_stat.timer.set_compute_end(COMPUTE_THREAD);
  // 	    ml_sst.round[0]++;
  // 	    // std::cout << "ml_sst.round[0]= " << ml_sst.round[0] << std::endl;
  // 	  }
  // 	}
  //     }
  //   };

  // auto model_broadcast_loop =
  //   [this, &training, &last_model_round]() mutable {
  //     pthread_setname_np(pthread_self(), ("broadcast"));
  //     while(training) {
  // 	if(ml_sst.round[0] > last_model_round) {
  // 	  ml_stat.timer.set_push_start();
  // 	  last_model_round = ml_sst.round[0];
  // 	  ml_sst.put_with_completion();
  // 	  ml_stat.timer.set_push_end(NETWORK_THREAD);
  // 	  ml_stat.num_broadcasts++;
  // 	  // TODO: measure model inconsistency ratio per broadcast here.
  // 	}
  //     }
  //   };
  
  // std::thread model_update_thread = std::thread(model_update_loop);
  // std::thread model_broadcast_thread = std::thread(model_broadcast_loop);
  // ml_sst->sync_barrier_with_server_and_workers(); // barrier #1 with workers for the training start
  // ml_stat.timer.set_start_time();
  // for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
  //   ml_stat.timer.set_train_start();
  //   // Workers are training during this time frame.
  //   ml_sst->sync_barrier_with_server_and_workers(); // barrier #2 with workers for the training end per epoch
  //   ml_stat.timer.set_train_end();
  //   ml_stat.set_epoch_parameters(epoch_num + 1, ml_model->get_model(),
  // 				 ml_sst);
  //   ml_sst->sync_barrier_with_server_and_workers(); // barrier #3 with workers for statistics per epoch
  // }
  // training = false;
  // model_update_thread.join();
  // model_broadcast_thread.join();
  // ml_sst->sync_barrier_with_server_and_workers(); // barrier #4 with workers for the trainig end
}
