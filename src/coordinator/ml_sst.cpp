#include <algorithm>
#include <assert.h>
#include <iostream>

#include "ml_sst.hpp"
#include "utils/utils.hpp"

sst::MLSST::MLSST(const bool has_buf,
		  const int my_node_id,
		  const int num_nodes,
		  ml_model::ml_model* ml_model)
  : has_buf(has_buf),
    my_node_id(my_node_id),
    num_nodes(num_nodes),
    ml_model(ml_model) {
}

/*** getters ***/
const bool sst::MLSST::get_has_buf() const {
  return has_buf;
};

const int sst::MLSST::get_my_node_id() const {
  return my_node_id;
};

const int sst::MLSST::get_num_nodes() const {
  return num_nodes;
}

const ml_model::ml_model* sst::MLSST::get_ml_model_ptr() const {
  return ml_model;
};

sst::MLSST_NOBUF::MLSST_NOBUF(const std::vector<uint32_t>& members,
			      const uint32_t my_node_id,
			      const size_t num_params,
			      const uint32_t num_nodes,
			      ml_model::ml_model* ml_model)
  : MLSST(false, my_node_id, num_nodes, ml_model),
    sst::SST<MLSST_NOBUF>(this, SSTParams{members, my_node_id}),
    model_or_gradient(num_params),
    last_round(num_nodes) {
    SSTInit(model_or_gradient, round, last_round);
    initialize();
}

bool sst::MLSST_NOBUF::has_new_gradient_received(int node_id,
						 int sgd_type) {
  if(sgd_type == SYNC) {
    return round[node_id] > round[0];
  } else {
    return round[node_id] > last_round[0][node_id];
  }
}

void sst::MLSST_NOBUF::mark_new_gradient_consumed(int node_id,
						  int sgd_type) {
  if(sgd_type == ASYNC) {
    last_round[0][node_id] = round[node_id];
  }
}

void sst::MLSST_NOBUF::wait_for_new_model(int sgd_type) {
  if(sgd_type == SYNC) {
    while(round[0] < round[1]) { 
    }
  } else {
    const int my_node_id = get_my_node_id();
    while (last_round[0][my_node_id] != round[1]) {
    }
  }
}

void sst::MLSST_NOBUF::broadcast_new_model_and_version_num() {
  assert (!ml_model->get_is_worker());
  put_with_completion();
}

void sst::MLSST_NOBUF::push_new_model_and_version_num(
				 std::vector<uint32_t>& model_receivers) {
  assert (!ml_model->get_is_worker());
  put_with_completion(model_receivers);
}

void sst::MLSST_NOBUF::push_new_gradient_and_version_num() {
  assert (ml_model->get_is_worker());
  put_with_completion();
}

void sst::MLSST_NOBUF::sync_barrier_with_server_and_workers() {
  sync_with_members();
}

void sst::MLSST_NOBUF::connect_ml_model_to_ml_sst() {
  
  // Set model memory of ml_model object to ml_sst
  ml_model->set_model_mem(
		 (double*)std::addressof(model_or_gradient[0][0]));
  if(ml_model->get_model_name() == "log_reg") {
    utils::zero_arr(ml_model->get_model(), ml_model->get_model_size());
  } else if(ml_model->get_model_name() == "dnn") {
    ml_model->init_model(ml_model->get_model(),
			 ml_model->get_init_model_file());
  } else {
    std::cerr << "ERROR: ml_model_name in ml_model is initialized wrong"
	      << std::endl;
    exit(1);
  }
  
  // Set gradient memory of ml_model object to ml_sst
  if(ml_model->get_is_worker()) {
    ml_model->set_gradient_mem(
	    (double*)std::addressof(model_or_gradient[1][0]));
  } else {
    const uint32_t num_nodes = get_num_rows();
    for (uint row = 1; row < num_nodes; ++row) {
      std::vector<double*> grad_row;
      grad_row.push_back(
	      (double*)std::addressof(model_or_gradient[row][0]));
      ml_model->push_back_to_grad_matrix(grad_row);
    }
  }
}

void sst::MLSST_NOBUF::increment_model_version_num() {
  round[0]++;
}

void sst::MLSST_NOBUF::increment_gradient_version_num() {
  round[1]++;
}

uint64_t sst::MLSST_NOBUF::get_latest_model_version_num() const {
  return round[0];
}

// For workers
uint64_t sst::MLSST_NOBUF::get_latest_gradient_version_num() const {
  assert (ml_model->get_is_worker());
  return round[1];
}

// For servers
uint64_t sst::MLSST_NOBUF::get_latest_gradient_version_num(int node_id) const {
  assert (!ml_model->get_is_worker());
  return round[node_id];
}


void sst::MLSST_NOBUF::print() {
    uint32_t num_nodes = get_num_rows();

    std::cout << "round ";
    for(uint i = 0; i < num_nodes; ++i) {
      std::cout << round[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "last_round";
    std::cout << std::endl;
    for(uint i = 0; i < num_nodes; ++i) {
      std::cout << "[" << i << "] ";
      for(uint j = 0; j < last_round.size(); ++j) {
	std::cout << last_round[i][j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
}

/*** getters for statistics ***/
bool sst::MLSST_NOBUF::has_gradient_loss(int node_id) const {
  return (round[node_id] - last_round[0][node_id]) > 1;
}

uint64_t sst::MLSST_NOBUF::get_num_lost_gradient(int node_id) const {
  return round[node_id] - last_round[0][node_id] - 1;
}

void sst::MLSST_NOBUF::initialize() {
    uint32_t num_nodes = get_num_nodes();
    for(uint i = 0; i < num_nodes; ++i) {
        utils::zero_arr((double*)std::addressof(
                                model_or_gradient[i][0]),
                        model_or_gradient.size());
    }
    for(uint i = 0; i < num_nodes; ++i) {
        round[i] = 0;
    }
    
    for(uint i = 0; i < num_nodes; ++i) {
      for(uint j = 0; j < last_round.size(); ++j) {
        last_round[i][j] = 0;
      }
    }
    put_with_completion();
    sync_with_members();
}

sst::MLSST_BUF::MLSST_BUF(const std::vector<uint32_t>& members,
			  const uint32_t my_node_id,
			  const size_t num_params,
			  const uint32_t num_nodes,
			  ml_model::ml_model* ml_model)
  : MLSST(true, my_node_id, num_nodes, ml_model),
    sst::SST<MLSST_BUF>(this, SSTParams{members, my_node_id}),
    model_or_gradient1(num_params),
    last_round1(num_nodes),
    model_or_gradient2(num_params),
    last_round2(num_nodes),
    model_or_gradient3(num_params),
    last_round3(num_nodes) {
    SSTInit(model_or_gradient1, round1, last_round1,
	    model_or_gradient2, round2, last_round2,
	    model_or_gradient3, round3, last_round3);
    initialize();
}

/*** main methods ***/
// TODO
bool sst::MLSST_BUF::has_new_gradient_received(int node_id,
					       int sgd_type) {
  return true;
}
void sst::MLSST_BUF::mark_new_gradient_consumed(int node_id,
						int sgd_type) {}
void sst::MLSST_BUF::wait_for_new_model(int sgd_type) {}
void sst::MLSST_BUF::broadcast_new_model_and_version_num() {}
void sst::MLSST_BUF::push_new_model_and_version_num(
			      std::vector<uint32_t>& model_receivers) {}
void sst::MLSST_BUF::push_new_gradient_and_version_num() {}
void sst::MLSST_BUF::sync_barrier_with_server_and_workers() {
  sync_with_members();
}

/*** setters ***/
void sst::MLSST_BUF::connect_ml_model_to_ml_sst() {
  
  // Set model memory of ml_model object to ml_sst
  ml_model->set_model_mem(
		 (double*)std::addressof(model_or_gradient1[0][0]));
  if(ml_model->get_model_name() == "log_reg") {
    utils::zero_arr(ml_model->get_model(), ml_model->get_model_size());
  } else if(ml_model->get_model_name() == "dnn") {
    ml_model->init_model(ml_model->get_model(),
			 ml_model->get_init_model_file());
  } else {
    std::cerr << "ERROR: ml_model-> ml_model_name is initialized wrong"
	      << std::endl;
    exit(1);
  }
  
  // Set gradient memory of ml_model object to ml_sst
  if(ml_model->get_is_worker()) {
      std::vector<double*> grad_row;
      grad_row.push_back(
	      (double*)std::addressof(model_or_gradient1[1][0]));
      grad_row.push_back(
	      (double*)std::addressof(model_or_gradient2[1][0]));
      grad_row.push_back(
	      (double*)std::addressof(model_or_gradient3[1][0]));
      ml_model->push_back_to_grad_matrix(grad_row);
      ml_model->set_gradient_mem(
		     (double*)std::addressof(model_or_gradient1[1][0]));
  } else {
    const uint32_t num_nodes = get_num_rows();
    for (uint row = 1; row < num_nodes; ++row) {
      std::vector<double*> grad_row;
      grad_row.push_back(
	      (double*)std::addressof(model_or_gradient1[row][0]));
      grad_row.push_back(
	      (double*)std::addressof(model_or_gradient2[row][0]));
      grad_row.push_back(
	      (double*)std::addressof(model_or_gradient3[row][0]));
      ml_model->push_back_to_grad_matrix(grad_row);
    }
  }
}

// TODO
void sst::MLSST_BUF::increment_model_version_num() {}
void sst::MLSST_BUF::increment_gradient_version_num() {}

/*** getters ***/
uint64_t sst::MLSST_BUF::get_latest_model_version_num() const {
  return round1[0];
}

// For workers
uint64_t sst::MLSST_BUF::get_latest_gradient_version_num() const {
  assert (ml_model->get_is_worker());
  return std::max(std::max(round1[1], round2[1]), round3[1]);
}

// For servers
uint64_t sst::MLSST_BUF::get_latest_gradient_version_num(int node_id) const {
  assert (!ml_model->get_is_worker());
  return std::max(std::max(round1[node_id], round2[node_id]),
		  round3[node_id]);
}

void sst::MLSST_BUF::print() {
    uint32_t num_nodes = get_num_rows();

    std::cout << "round1 ";
    for(uint i = 0; i < num_nodes; ++i) {
      std::cout << round1[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "last_round1";
    std::cout << std::endl;
    for(uint i = 0; i < num_nodes; ++i) {
      std::cout << "[" << i << "] ";
      for(uint j = 0; j < last_round1.size(); ++j) {
	std::cout << last_round1[i][j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "round2 ";
    for(uint i = 0; i < num_nodes; ++i) {
      std::cout << round2[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "last_round2";
    std::cout << std::endl;
    for(uint i = 0; i < num_nodes; ++i) {
      std::cout << "[" << i << "] ";
      for(uint j = 0; j < last_round2.size(); ++j) {
	std::cout << last_round2[i][j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "round3 ";
    for(uint i = 0; i < num_nodes; ++i) {
      std::cout << round3[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "last_round3";
    std::cout << std::endl;
    for(uint i = 0; i < num_nodes; ++i) {
      std::cout << "[" << i << "] ";
      for(uint j = 0; j < last_round3.size(); ++j) {
	std::cout << last_round3[i][j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
}

/*** getters for statistics ***/
bool sst::MLSST_BUF::has_gradient_loss(int node_id) const {
  std::cerr << "not supported yet" << std::endl;
  exit(1);
  return true;
}

uint64_t sst::MLSST_BUF::get_num_lost_gradient(int node_id) const {
  std::cerr << "not supported yet" << std::endl;
  exit(1);
  return 1;
}

void sst::MLSST_BUF::initialize() {
    uint32_t num_nodes = get_num_rows();
    for(uint i = 0; i < num_nodes; ++i) {
        utils::zero_arr((double*)std::addressof(
                                model_or_gradient1[i][0]),
                        model_or_gradient1.size());
        utils::zero_arr((double*)std::addressof(
                                model_or_gradient2[i][0]),
                        model_or_gradient2.size());
        utils::zero_arr((double*)std::addressof(
                                model_or_gradient3[i][0]),
                        model_or_gradient3.size());
    }
    
    for(uint i = 0; i < num_nodes; ++i) {
        round1[i] = 0;
        round2[i] = 0;
        round3[i] = 0;
    }
    
    for(uint i = 0; i < num_nodes; ++i) {
      for(uint j = 0; j < last_round1.size(); ++j) {
        last_round1[i][j] = 0;
        last_round2[i][j] = 0;
        last_round3[i][j] = 0;
      }
    }
    
    put_with_completion();  // end of initialization
    sync_with_members();
}
