#include <iostream>

#include "ml_model.hpp"

// virtual functions
size_t ml_model::ml_model::get_model_size() const {
  std::cerr << "BUG: virtual function get_model_size() is called." << std::endl;
  return -1;
}

void ml_model::ml_model::set_model_mem(double* model) {
  std::cerr << "BUG: virtual function set_model_mem() is called." << std::endl;
}

void ml_model::ml_model::push_back_to_grad_vec(double* gradient) {
  std::cerr << "BUG: virtual function push_back_to_grad_vec() is called." << std::endl;
}
size_t ml_model::ml_model::get_num_batches() const {
  std::cerr << "BUG: virtual function get_num_batches() is called." << std::endl;
  return -1;
}
void ml_model::ml_model::update_model(uint ml_sst_row) {
  std::cerr << "BUG: virtual function update_model() is called." << std::endl;
}

double* ml_model::ml_model::get_model() const {
  std::cerr << "BUG: get_model() is called." << std::endl;
  return NULL;
}

void ml_model::ml_model::compute_gradient(const size_t batch_num, double* given_model) {
  std::cerr << "BUG: compute_gradient() is called." << std::endl;
}

void ml_model::ml_model::train(const size_t num_epochs) {
  std::cerr << "BUG: train() is called." << std::endl;
}

double ml_model::ml_model::training_error() {
  std::cerr << "BUG: training_error() is called." << std::endl;
  return -1.0;
}

double ml_model::ml_model::training_loss() {
  std::cerr << "BUG: training_loss() is called." << std::endl;
  return -1.0;
}

double ml_model::ml_model::get_loss_opt() const {
  std::cerr << "BUG: get_loss_opt() is called." << std::endl;
  return -1.0;
}

double ml_model::ml_model::test_error() {
  std::cerr << "BUG: test_error() is called." << std::endl;
  return -1.0;
}

double ml_model::ml_model::gradient_norm() {
  std::cerr << "BUG: gradient_norm() is called." << std::endl;
  return -1.0;
}

double ml_model::ml_model::distance_to_optimum() {
  std::cerr << "BUG: distance_t_optimum() is called." << std::endl;
  return -1.0;
}

