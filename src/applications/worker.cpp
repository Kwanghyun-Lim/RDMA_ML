#include <bits/stdc++.h>
#include <cblas.h>
#include <chrono>
#include <cmath>
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
    if(argc < 12) {
        std::cerr << "Usage: " << argv[0]
		  << " <data_directory> <syn/mnist/rff> \
                       <sync/async/fully_async> \
                       <log_reg/dnn> \
                       <alpha> <decay> \
                       <aggregate_batch_size> <num_epochs> \
                       <node_rank> <num_nodes> <num_trials>"
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
    uint32_t aggregate_batch_size = std::stod(argv[7]);
    const uint32_t num_epochs = atoi(argv[8]);
    const uint32_t node_rank = atoi(argv[9]);
    const uint32_t num_nodes = atoi(argv[10]);
    const uint32_t num_trials = atoi(argv[11]);
    const size_t batch_size = aggregate_batch_size / (num_nodes - 1);
    bool has_buffers;
    if(sgd_type == "fully_async") {
      has_buffers = true;
    } else {
      has_buffers = false;
    }
    std::string init_model_file = data_directory + "/" + data
	                            + "/model_784-128-10.npy";
    // openblas_set_num_threads(1);

    std::cout << "ml_model_name=" << ml_model_name << std::endl;
    std::cout << "num_trials=" << num_trials << std::endl;

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
    ip_addrs[0] = ip_addrs_static.at(0);
    ip_addrs[node_rank] = ip_addrs_static.at(node_rank);
    sst::verbs_initialize(ip_addrs, node_rank);
    
    // multiple trials for statistics
    for(uint32_t trial_num = 0; trial_num < num_trials; ++trial_num) {
      std::cout << "trial_num " << trial_num << std::endl;
      
      // Initialize ml_model, ml_sst, ml_stat, and worker for training
      ml_model::ml_model* ml_model;
      if (ml_model_name == "log_reg") {
	std::cout << "ml_model_name=" << ml_model_name << std::endl;
         ml_model = new ml_model::multinomial_log_reg(
     	     [&]() {return (utils::dataset)numpy::numpy_dataset(
		                       data_directory + "/" + data,
		                       num_nodes - 1, node_rank - 1);},
	     alpha, gamma, decay, batch_size,
	     std::string(""), has_buffers, WORKER);
      } else if (ml_model_name == "dnn") {
	const std::vector<uint32_t> layer_size_vec {784, 128, 10};
        const uint32_t num_layers = 3;
        ml_model = new ml_model::deep_neural_network(
	    layer_size_vec, num_layers,	     
            [&]() {return (utils::dataset)numpy::numpy_dataset(
		                       data_directory + "/" + data,
		                       num_nodes - 1, node_rank - 1);},
	    alpha, batch_size,
	    init_model_file,
	    has_buffers, WORKER);
      } else {
	std::cerr << "Wrong ml_model_name input: "
		  << ml_model_name << std::endl;
	exit(1);
      }
      
      sst::MLSST* ml_sst;
      if(sgd_type == "sync" || sgd_type == "async") {
	ml_sst = new sst::MLSST_NOBUF(std::vector<uint32_t>{0, node_rank},
				      node_rank,
				      ml_model->get_model_size(),
				      num_nodes, ml_model);
      } else if(sgd_type == "fully_async") {
	ml_sst = new sst::MLSST_BUF(std::vector<uint32_t>{0, node_rank},
				    node_rank,
				    ml_model->get_model_size(),
				    num_nodes, ml_model);
      } else {
      	std::cerr << "Wrong sgd_type input: "
		  << sgd_type << std::endl;
	exit(1);
      }
      std::cout << "worker #1" << std::endl;
      ml_sst->connect_ml_model_to_ml_sst();
      std::cout << "worker #2" << std::endl;
      
      utils::ml_stat_t ml_stat(trial_num, num_nodes, num_epochs,
			       alpha, decay, batch_size, node_rank,
			       ml_sst, ml_model);
      std::cout << "worker #3" << std::endl;
      worker::worker* wrk;
      if(sgd_type == "sync") {
	wrk = new worker::sync_worker(ml_model, ml_sst,
				      ml_stat, node_rank);
      } else if(sgd_type == "async") {
	wrk = new worker::async_worker(ml_model, ml_sst,
				       ml_stat, node_rank);
      } else if(sgd_type == "fully_async") {
	wrk = new worker::fully_async_worker(ml_model, ml_sst,
					     ml_stat, node_rank);
      } else {
	std::cerr << "Wrong sgd_type input: "
		  << ml_model_name << std::endl;
	exit(1);
      }
      std::cout << "worker #4" << std::endl;

      // Train
      wrk->train(num_epochs);
      std::cout << "worker #5" << std::endl;
      
      std::cout << "trial_num " << trial_num << " done." << std::endl;
      std::cout << "Collecting results..." << std::endl;
      if (sgd_type == "fully_async") {
	ml_stat.fout_op_time_log(WORKER_LOG, FULLY_ASYNC_LOG);
      } else {
	ml_stat.fout_op_time_log(WORKER_LOG, NOT_FULLY_ASYNC_LOG);
      }
      std::cout << "Collecting results done." << std::endl;
      ml_sst->sync_barrier_with_server_and_workers(); // barrier #5
      if (trial_num == num_trials - 1) {
	std::cout << "All trainings and loggings done." << std::endl;
	ml_sst->sync_barrier_with_server_and_workers(); // barrier #6
      }
    }
}

worker::worker::worker(ml_model::ml_model* ml_model,
		 sst::MLSST* ml_sst,
		 utils::ml_stat_t& ml_stat,
		 const uint32_t node_rank)
  : ml_model(ml_model),
    ml_sst(ml_sst),
    ml_stat(ml_stat),
    node_rank(node_rank) {
}

void worker::worker::train(const size_t num_epochs) {
  // virtual function
}

worker::sync_worker::sync_worker(ml_model::ml_model* ml_model,
				   sst::MLSST* ml_sst,
				   utils::ml_stat_t& ml_stat,
				   const uint32_t node_rank)
  : worker(ml_model, ml_sst, ml_stat, node_rank) {
}

worker::sync_worker::~sync_worker() {
  std::cout << "sync_worker deconstructor does nothing." << std::endl;
}

void worker::sync_worker::train(const size_t num_epochs) {
  ml_sst->sync_barrier_with_server_and_workers(); // barrier #1
  ml_stat.timer.set_start_time();
  std::vector<uint32_t> server{0};

  const size_t num_batches = ml_model->get_num_batches();
  for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
    for(size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      ml_stat.timer.set_wait_start();
      ml_sst->wait_for_new_model(SYNC);
      ml_stat.timer.set_wait_end();
      
      ml_stat.timer.set_compute_start();
      ml_model->compute_gradient(batch_num);
      ml_sst->increment_gradient_version_num();
      ml_stat.timer.set_compute_end();
      
      ml_stat.timer.set_push_start();
      ml_sst->push_new_gradient_and_version_num();
      ml_stat.timer.set_push_end();
    }
    ml_sst->sync_barrier_with_server_and_workers(); // barrier #2
    // Between those two,
    // server stores intermidiate models and parameters for statistics
    ml_sst->sync_barrier_with_server_and_workers(); // barrier #3
  }
  ml_sst->sync_barrier_with_server_and_workers(); // barrier #4
}

worker::async_worker::async_worker(ml_model::ml_model* ml_model,
				   sst::MLSST* ml_sst,
				   utils::ml_stat_t& ml_stat,
				   const uint32_t node_rank)
  : worker(ml_model, ml_sst, ml_stat, node_rank) {
}

worker::async_worker::~async_worker() {
  std::cout << "async_worker deconstructor does nothing." << std::endl;
}

void worker::async_worker::train(const size_t num_epochs) {
  ml_sst->sync_barrier_with_server_and_workers(); // barrier #1
  ml_stat.timer.set_start_time();
  std::vector<uint32_t> server{0};
  
  const size_t num_batches = ml_model->get_num_batches();
  for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
    for(size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      ml_stat.timer.set_wait_start();
      ml_sst->wait_for_new_model(ASYNC);
      ml_stat.timer.set_wait_end();

      ml_stat.timer.set_compute_start();
      ml_model->compute_gradient(batch_num);
      ml_sst->increment_gradient_version_num();
      ml_stat.timer.set_compute_end();

      ml_stat.timer.set_push_start();
      ml_sst->push_new_gradient_and_version_num();
      ml_stat.timer.set_push_end();
    }
    ml_sst->sync_barrier_with_server_and_workers(); // barrier #2
    // During this time frame,
    // server stores intermidiate model for statistics
    ml_sst->sync_barrier_with_server_and_workers(); // barrier #3 
  }
  ml_sst->sync_barrier_with_server_and_workers(); // barrier #4 
}

worker::fully_async_worker::fully_async_worker(ml_model::ml_model* ml_model,
				   sst::MLSST* ml_sst,
				   utils::ml_stat_t& ml_stat,
				   const uint32_t node_rank)
  : worker(ml_model, ml_sst, ml_stat, node_rank) {
}

worker::fully_async_worker::~fully_async_worker() {
  std::cout << "fully_async_worker deconstructor does nothing." << std::endl;
}

void worker::fully_async_worker::train(const size_t num_epochs) {
  // std::atomic<bool> training = true;
  // std::atomic<uint64_t> last_model_round = 0;
  // std::atomic<uint64_t> last_gradient_round = 0;
  // float staleness_threshold = 3.0;
  // float model_staleness = 0.0;
  // float normalized_model_round = 0.0;
  
  // auto gradient_push_loop =
  //   [this, &training, &last_gradient_round]() mutable {
  //     pthread_setname_np(pthread_self(), ("push"));
  //     while(training) {
  // 	if(ml_sst.round[1] > last_gradient_round) {
  // 	  last_gradient_round = ml_sst.round[1];
  // 	  ml_stat.timer.set_push_start();
  // 	  ml_sst.put_with_completion();
  // 	  ml_stat.timer.set_push_end(NETWORK_THREAD);
  // 	}
  //     }
  //   };
  
  // std::thread gradient_push_thread = std::thread(gradient_push_loop);
  
  // ml_sst->sync_barrier_with_server_and_workers(); // barrier #1 with server and other workers for the training start
  // ml_stat.timer.set_start_time();

  // const uint32_t num_nodes = ml_sst->get_num_nodes();
  // const size_t num_batches = ml_model->get_num_batches();
  // for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
  //     for(size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
  // 	  ml_stat.timer.set_compute_start();
  // 	  ml_model->compute_gradient(batch_num);
  // 	  last_model_round = ml_sst.round[0]; 
  // 	  ml_sst.round[1]++;
  // 	  ml_stat.timer.set_compute_end(COMPUTE_THREAD);
	  
  // 	  ml_stat.timer.set_wait_start(COMPUTE_THREAD);
  // 	  // TO FIX: seems still sync between server and worker
  // 	  // while(ml_sst.round[0] == last_model_round) {
  // 	  // }
  // 	  normalized_model_round = ml_sst.round[0] / (num_nodes - 1);
  // 	  model_staleness = (float)ml_sst.round[1] - 1.0 - normalized_model_round;
  // 	  // std::cout << "normalized_model_round= " << normalized_model_round
  // 	  //           << " ml_sst.round[1]= " << ml_sst.round[1] 
  // 	  //           << " model_staleness= " << model_staleness << std::endl;
  // 	  while(model_staleness > staleness_threshold) {
  // 	    normalized_model_round = ml_sst.round[0] / (num_nodes - 1);
  // 	    model_staleness = (float)ml_sst.round[1] - 1.0 - normalized_model_round;
  // 	  }
  // 	  ml_stat.timer.set_wait_end(COMPUTE_THREAD);
  //     }
  //     ml_sst->sync_barrier_with_server_and_workers(); // barrier #2 with server and other workers for the training end per epoch
  //     // During this time frame, server stores intermidiate model for statistics
  //     ml_sst->sync_barrier_with_server_and_workers(); // barrier #3 with server and other workers for statistics per epoch
  // }

  // training = false;
  // gradient_push_thread.join();
  // ml_sst->sync_barrier_with_server_and_workers(); // barrier #4 with server and other workers for the trainig end
}

