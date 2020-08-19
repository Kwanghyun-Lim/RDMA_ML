#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>

#include "utils/utils.hpp"
#include "utils/cnpy.hpp"

namespace DNN {
class relu {
public:
  relu(size_t x_len, const size_t batch_size);
  ~relu() { std::cout << "Relu destructor" << std::endl; };
  // X is a pointer to the matrix of (batch_size, features) in the previous affine object
  double* forward(double* X); 
  double* backward(double* dL_dY);

  // x_len only counts the number of features of one sample. It doesn't count batch_size.
  size_t x_len;
  double* Y; // Y = X*W + B 
  double* dL_dX; // dL/dX (batch_size, x_len)
  const size_t batch_size;
};

class affine {
public:
  affine(size_t W_row_len, size_t W_col_len, const size_t batch_size);
  double* forward(double* X);
  // for the first layer
  double* forward(double* X, const size_t batch_num, const size_t total_num_data); 
  double* backward(double* dL_dY);
  // for the first layer
  double* backward(double* dL_dY, const size_t batch_num, const size_t total_num_data); 

  double* X; // It points to the precious layer's Y
  
  double* W; // It points to some offset within the ml_sst row
  double* b; // It points to some offset within the ml_sst row
  double* Y; // Y = X*W + B // (batch_size, W_row_len) * (W_row_len, W_col_len) + (batch_size, W_col_len)

  double* dL_dW; // It points to some offset within the ml_sst row
  double* dL_db; // It points to some offset within the ml_sst row
  double* dL_dX; // dL/dX

  size_t W_row_len;
  size_t W_col_len;
  size_t b_len;
  const size_t batch_size;
};

class softmax {
public:
  softmax(size_t y_len, const size_t batch_size);
  double* forward(double* X);
  double* backward(double* Y_labels, const size_t batch_num, const size_t total_num_data);

  double* Y; // Y is a prediction matrix of (batch_size, num_classes)
  double* dL_dX;
  
  size_t y_len; // doesn't count batch_size
  const size_t batch_size;
};
  
class deep_neural_network {
public:
    deep_neural_network(const std::vector<uint32_t>& layer_size_vec, const uint32_t num_layers,
		    const utils::reader_t& dataset_loader, const double alpha,
		    const size_t batch_size, const bool is_worker);

    void train(const size_t num_epochs);

    double training_error();
    double training_loss();
    double test_error();

    void compute_gradient(const size_t batch_num);
    void update_model();

    void set_model_mem(double* model);
    void initialize_model_mem_with_zero();
    void set_gradient_mem(double* gradient);

    size_t get_model_size() const;

    size_t get_num_batches(const utils::images_t& images) const;
    size_t get_num_images() const;
    void init_model(double* model, std::string full_path);

private:
    double compute_error(const utils::images_t& images, const utils::labels_t& labels);
    double compute_loss(const utils::images_t& images, const utils::labels_t& labels);

    const bool is_worker;
    utils::dataset dataset;
    size_t model_size = 0;
    double* model = NULL;
    double* gradient;
    const double alpha;
    const size_t batch_size;
    const std::vector<uint32_t>& layer_size_vec;
    const uint32_t num_layers;

    std::vector<affine*> affine_layers;
    std::vector<relu*> relu_layers;
    softmax* last_layer;
};
}  // namespace DNN
