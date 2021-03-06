#pragma once

#include <atomic>
#include <map>
#include <memory>

#include "coordinator/ml_sst.hpp"
#include "coordinator/tcp.hpp"
#include "utils/ml_stat.hpp"
#include "log_reg.hpp"
#include "dnn.hpp"

namespace worker {
class worker {
public:
  worker(ml_model::ml_model* ml_model,
		 sst::MLSST* ml_sst,
		 utils::ml_stat_t& ml_stat,
		 const uint32_t node_rank);

  virtual void train(const size_t num_epochs);
  
protected:
  ml_model::ml_model* ml_model;
  sst::MLSST* ml_sst;
  utils::ml_stat_t& ml_stat;
  const uint32_t node_rank;
};

class sync_worker: public worker {
public:
  sync_worker(ml_model::ml_model* ml_model,
		 sst::MLSST* ml_sst,
		 utils::ml_stat_t& ml_stat,
		 const uint32_t node_rank);

  void train(const size_t num_epochs);
  ~sync_worker();
};
  
class async_worker: public worker {
public:
  async_worker(ml_model::ml_model* ml_model,
		 sst::MLSST* ml_sst,
		 utils::ml_stat_t& ml_stat,
		 const uint32_t node_rank);

  void train(const size_t num_epochs);
  ~async_worker();
};

class fully_async_worker: public worker {
public:
  fully_async_worker(ml_model::ml_model* ml_model,
		 sst::MLSST* ml_sst,
		 utils::ml_stat_t& ml_stat,
		 const uint32_t node_rank);

  void train(const size_t num_epochs);
  ~fully_async_worker();
};
}  // namespace coordinator
