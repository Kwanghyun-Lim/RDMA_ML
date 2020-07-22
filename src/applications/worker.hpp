#pragma once

#include <atomic>
#include <map>
#include <memory>

#include "coordinator/ml_sst.hpp"
#include "coordinator/tcp.hpp"
#include "log_reg.hpp"

namespace worker {
class async_worker {
public:
    async_worker(log_reg::multinomial_log_reg& m_log_reg,
		 sst::MLSST& ml_sst, const uint32_t node_rank);

  void train(const size_t num_epochs);
  
private:
  log_reg::multinomial_log_reg& m_log_reg;
  sst::MLSST& ml_sst;
  const uint32_t node_rank;
};
}  // namespace coordinator
