#pragma once

#include <atomic>
#include <map>
#include <memory>

#include "coordinator/ml_sst.hpp"
#include "coordinator/tcp.hpp"
#include "utils/ml_stat.hpp"
#include "log_reg.hpp"

namespace worker {
class async_worker {
public:
    async_worker(log_reg::multinomial_log_reg& m_log_reg,
		 sst::MLSST& ml_sst,
		 utils::ml_stat_t& ml_stat,
		 const uint32_t node_rank);

  void train(const size_t num_epochs);
  
private:
  log_reg::multinomial_log_reg& m_log_reg;
  sst::MLSST& ml_sst;
  utils::ml_stat_t& ml_stat;
  const uint32_t node_rank;
};
}  // namespace coordinator
