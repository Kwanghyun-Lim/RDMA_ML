#pragma once

#include <atomic>
#include <map>
#include <memory>

#include "coordinator/ml_sst.hpp"
#include "coordinator/tcp.hpp"
#include "utils/ml_stat.hpp"
#include "log_reg.hpp"

namespace server {
class server {
public:
  server(ml_model::ml_model* ml_model,
	 sst::MLSST& ml_sst, utils::ml_stat_t& ml_stat);

  virtual void train(const size_t num_epochs);
  
protected:
  ml_model::ml_model* ml_model;
  sst::MLSST& ml_sst;
  utils::ml_stat_t& ml_stat;
};
  
class sync_server: public server {
public:
  sync_server(ml_model::ml_model* ml_model,
	      sst::MLSST& ml_sst, utils::ml_stat_t& ml_stat);

  void train(const size_t num_epochs);
  ~sync_server();
};
  
class async_server: public server {
public:
  async_server(ml_model::ml_model* ml_model,
	       sst::MLSST& ml_sst, utils::ml_stat_t& ml_stat);

  void train(const size_t num_epochs);
  ~async_server();
};

class fully_async_server: public server {
public:
  fully_async_server(ml_model::ml_model* ml_model,
	       sst::MLSST& ml_sst, utils::ml_stat_t& ml_stat);

  void train(const size_t num_epochs);
  ~fully_async_server();
};
}  // namespace server
