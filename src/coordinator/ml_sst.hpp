#pragma once

#include <vector>

#include "ml_model.hpp"
#include "sst.h"

namespace sst {

#ifndef SGD_TYPE
#define SYNC 0
#define ASYNC 1
#define FULLY_ASYNC 2
#endif
  
class MLSST {
public:
  MLSST(const bool has_buf,
	const int my_rank,
	const int num_nodes,
	ml_model::ml_model* ml_model);

  /*** main methods ***/
  virtual bool has_new_gradient_received(int node_id, int sgd_type) = 0;
  virtual void mark_new_gradient_consumed(int node_id, int sgd_type) = 0;
  virtual void wait_for_new_model(int sgd_type) = 0;
  virtual void broadcast_new_model_and_version_num() = 0;
  virtual void push_new_model_and_version_num(
			      std::vector<uint32_t>& model_receivers) = 0;
  virtual void push_new_gradient_and_version_num() = 0;
  virtual void sync_barrier_with_server_and_workers() = 0;
  
  /*** setters ***/
  virtual void connect_ml_model_to_ml_sst() = 0;
  virtual void increment_model_version_num() = 0;
  virtual void increment_gradient_version_num() = 0;
  
  /*** getters ***/
  virtual uint64_t get_latest_model_version_num() const = 0;
  virtual uint64_t get_latest_gradient_version_num() const = 0;
  virtual uint64_t get_latest_gradient_version_num(int node_id) const = 0;
  virtual void print() = 0;
  const bool get_has_buf() const;
  const int get_my_rank() const;
  const int get_num_nodes() const;
  const ml_model::ml_model* get_ml_model_ptr() const;

  /*** getters for statistics ***/
  virtual bool has_gradient_loss(int node_id) const = 0;
  virtual uint64_t get_num_lost_gradient(int node_id) const = 0;
  
protected:  
  const bool has_buf;
  const int my_rank;
  const uint32_t num_nodes;
  ml_model::ml_model* ml_model;
};

class MLSST_NOBUF : public SST<MLSST_NOBUF>, public MLSST {
public:
  MLSST_NOBUF(const std::vector<uint32_t>& members,
	      const uint32_t my_rank,
	      const size_t num_params,
	      const uint32_t num_nodes,
	      ml_model::ml_model* ml_model);

  /*** main methods ***/
  bool has_new_gradient_received(int node_id, int sgd_type);
  void mark_new_gradient_consumed(int node_id, int sgd_type);
  void wait_for_new_model(int sgd_type);
  void broadcast_new_model_and_version_num();
  void push_new_model_and_version_num(
				  std::vector<uint32_t>& model_receivers);
  void push_new_gradient_and_version_num();
  void sync_barrier_with_server_and_workers();
  
  /*** setters ***/
  void connect_ml_model_to_ml_sst();
  void increment_model_version_num();
  void increment_gradient_version_num();
  
  /*** getters ***/
  uint64_t get_latest_model_version_num() const;
  uint64_t get_latest_gradient_version_num() const;
  uint64_t get_latest_gradient_version_num(int node_id) const;
  void print();

  /*** getters for statistics ***/
  bool has_gradient_loss(int node_id) const;
  uint64_t get_num_lost_gradient(int node_id) const;

  
private:
  void initialize();
  
  SSTFieldVector<double> model_or_gradient;
  SSTField<uint64_t> round;
  SSTFieldVector<uint64_t> last_round;
};

class MLSST_BUF : public SST<MLSST_BUF>, public MLSST {
public:
  MLSST_BUF(const std::vector<uint32_t>& members,
	    const uint32_t my_rank,
	    const size_t num_params,
	    const uint32_t num_nodes,
	    ml_model::ml_model* ml_model);

  /*** main methods ***/
  bool has_new_gradient_received(int node_id, int sgd_type);
  void mark_new_gradient_consumed(int node_id, int sgd_type);
  void wait_for_new_model(int sgd_type);
  void broadcast_new_model_and_version_num();
  void push_new_model_and_version_num(
				  std::vector<uint32_t>& model_receivers);
  void push_new_gradient_and_version_num();
  void sync_barrier_with_server_and_workers();
  
  /*** setters ***/
  void connect_ml_model_to_ml_sst();
  void increment_model_version_num();
  void increment_gradient_version_num();
  
  /*** getters ***/
  uint64_t get_latest_model_version_num() const;
  uint64_t get_latest_gradient_version_num() const;
  uint64_t get_latest_gradient_version_num(int node_id) const;
  void print();

  /*** getters for statistics ***/
  bool has_gradient_loss(int node_id) const;
  uint64_t get_num_lost_gradient(int node_id) const;

private:
  void initialize();
  
  SSTFieldVector<double> model_or_gradient1;
  SSTField<uint64_t> round1;
  SSTFieldVector<uint64_t> last_round1;
  
  SSTFieldVector<double> model_or_gradient2;
  SSTField<uint64_t> round2;
  SSTFieldVector<uint64_t> last_round2;

  SSTFieldVector<double> model_or_gradient3;
  SSTField<uint64_t> round3;
  SSTFieldVector<uint64_t> last_round3;
};
}  // namespace sst
