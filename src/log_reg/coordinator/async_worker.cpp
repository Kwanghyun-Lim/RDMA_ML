#include "async_worker.hpp"

#include <bits/stdc++.h>
#include <chrono>
#include <map>
#include <queue>
#include <sys/time.h>
#include <thread>

coordinator::async_worker::async_worker(log_reg::multinomial_log_reg& m_log_reg,
					sst::MLSST& ml_sst, const uint32_t node_rank)
  : m_log_reg(m_log_reg), ml_sst(ml_sst), node_rank(node_rank) {
}

void coordinator::async_worker::train(const size_t num_epochs) {
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

void coordinator::async_worker::train_SVRG(const size_t num_epochs) {
    const size_t num_batches = m_log_reg.get_num_batches();
    for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
        m_log_reg.copy_model(m_log_reg.get_model(),
  			   m_log_reg.get_anchor_model(),
			   m_log_reg.get_model_size());
        m_log_reg.compute_full_gradient(m_log_reg.get_anchor_model());

        for(size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
            m_log_reg.compute_gradient(batch_num,
				       m_log_reg.get_model());
	    m_log_reg.update_gradient(batch_num);
	    ml_sst.round[1]++;
	    ml_sst.put_with_completion();
        }
    }
    ml_sst.sync_with_members(); // for time_taken
}
