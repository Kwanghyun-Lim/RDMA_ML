#pragma once

#include <memory>
#include <string>

#include "utils/utils.hpp"
#include "utils/cnpy.hpp"
#include "ml_model.hpp"

namespace ml_model {
class multinomial_log_reg : public ml_model {
public:
  multinomial_log_reg(const utils::reader_t& dataset_loader,
		      const double alpha,
		      const double gamma,
		      double decay,
		      const size_t batch_size,
		      const std::string init_model_file,
		      const bool has_buffers,
		      const bool is_worker);
  ~multinomial_log_reg();

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
  /*** helpers ***/
  // gradient_norm() helper function
  void compute_full_gradient(double* given_model);
  // update_model() helper funcion
  double decay_alpha();
  
  double* full_gradient; // for gradient norm statistics
  std::unique_ptr<double[]> full_predicted_labels; // for full_gradient
  // temporary space for predicted labels
  std::unique_ptr<double[]> predicted_labels;

  const double gamma;
  double decay;
  uint64_t num_model_updates;
};
}  // namespace ml_model
