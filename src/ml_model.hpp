#pragma once
#include <cstdlib>
#include <string>
#include <vector>

#include "utils/utils.hpp"

namespace ml_model {
// interface for ML models
class ml_model {
public:
  ml_model(const utils::reader_t& dataset_loader,
	   const std::string ml_model_name,
	   const std::string init_model_file,
	   const bool has_buffers,
	   const bool is_worker,
	   size_t model_size,
	   double alpha,
	   const size_t batch_size,
	   const size_t aggregate_batch_size);
  
  /*** main methods ***/
  void train(const size_t num_epochs);
  double training_error();
  double training_loss();
  double test_error();
  virtual void update_model();
  virtual void update_model(uint ml_sst_row, uint ml_sst_col);
  virtual void compute_gradient(const size_t batch_num);
  virtual void compute_gradient(size_t batch_num,
				double* src_model_ptr,
				double* dst_grad_ptr);
  virtual double compute_error(const utils::images_t& images,
			       const utils::labels_t& labels);
  virtual double compute_loss(const utils::images_t& images,
			      const utils::labels_t& labels);

  /*** setters ***/
  void push_back_to_grad_matrix(std::vector<double*> grad_row);
  void initialize_model_mem_with_zero();
  void init_model(double* model, std::string full_path);
  void copy_model(double* src, double* dst, size_t len);
  virtual void set_model_mem(double* model);
  virtual void set_gradient_mem(double* gradient);
  void save_npy_model() const;

  /*** getters ***/
  size_t get_num_part_images() const;
  size_t get_num_total_images() const;
  std::string get_init_model_file() const;
  std::string get_model_name() const;
  bool get_has_buffers() const;
  bool get_is_worker() const;
  size_t get_num_batches() const;
  size_t get_num_batches(const utils::images_t& images) const;
  size_t get_model_size() const;
  double* get_model() const;
  double* get_gradient() const;
  double* get_gradient(uint ml_sst_row, uint ml_sst_col) const;

  /*** getters for statistics ***/
  virtual double get_loss_opt() const = 0;
  virtual double gradient_norm() = 0;
  virtual double distance_to_optimum() = 0;
  virtual double* get_full_gradient() const = 0;

protected:
  utils::dataset dataset;
  
  const std::string ml_model_name;
  const std::string init_model_file; // full path
  const bool has_buffers;
  const bool is_worker;
  
  size_t model_size;
  double* model;
  double* gradient;
  std::vector<std::vector<double*>> grad_matrix;

  double alpha;
  const size_t batch_size;
  const size_t aggregate_batch_size;
};
} // namespace ml_model
