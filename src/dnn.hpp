#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>

#include "utils/utils.hpp"
#include "utils/cnpy.hpp"
#include "ml_model.hpp"
#include "layers_impl.hpp"

namespace ml_model {
class deep_neural_network : public ml_model {
public:
  deep_neural_network(const std::vector<uint32_t> layer_size_vec,
		      const uint32_t num_layers,
		      const utils::reader_t& dataset_loader,
		      const double alpha,
		      const size_t batch_size,
		      const std::string init_model_file,
		      const bool has_buffers,
		      const bool is_worker);

  /*** main methods ***/
  double compute_error(const utils::images_t& images,
		       const utils::labels_t& labels);
  double compute_loss(const utils::images_t& images,
		      const utils::labels_t& labels);
  void update_model();
  void update_model(uint ml_sst_row, uint ml_sst_col);
  void compute_gradient(size_t batch_num);
  void compute_gradient(size_t batch_num,
			double* src_model_ptr,
			double* dst_grad_ptr);

  /*** setters ***/
  void set_model_mem(double* model);
  void set_gradient_mem(double* gradient);

  /*** getters for statistics ***/
  double get_loss_opt() const;
  double gradient_norm();
  double distance_to_optimum();
  double* get_full_gradient() const;

private:
  const std::vector<uint32_t> layer_size_vec;
  const uint32_t num_layers;

  std::vector<affine*> affine_layers;
  std::vector<relu*> relu_layers;
  softmax* last_layer;
};
}  // namespace ml_model
