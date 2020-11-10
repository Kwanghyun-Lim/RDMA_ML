#include <iostream>
#include <string>

#include "ml_model.hpp"
#include "utils/numpy_reader.hpp"

ml_model::ml_model::ml_model(const utils::reader_t& dataset_loader,
			     const std::string ml_model_name,
			     const std::string init_model_file,
			     const bool has_buffers,
			     const bool is_worker,
			     size_t model_size,
			     double alpha,
	                     const size_t batch_size,
			     const size_t aggregate_batch_size)
  : dataset(dataset_loader()),
    ml_model_name(ml_model_name),
    init_model_file(init_model_file),
    has_buffers(has_buffers),
    is_worker(is_worker),
    model_size(model_size),
    grad_matrix(),
    alpha(alpha),
    batch_size(batch_size),
    aggregate_batch_size(aggregate_batch_size) {
}

/*** main methods ***/
void ml_model::ml_model::train(const size_t num_epochs) {
  const size_t num_batches = get_num_batches(dataset.training_images);
  for (size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
    for (size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      compute_gradient(batch_num);
      update_model();
    }
  }
}

double ml_model::ml_model::training_error() {
    return compute_error(dataset.training_images,
			 dataset.training_labels);
}

double ml_model::ml_model::training_loss() {
    return compute_loss(dataset.training_images,
			dataset.training_labels);
}

double ml_model::ml_model::test_error() {
    return compute_error(dataset.test_images,
			 dataset.test_labels);
}

// virtual function
void ml_model::ml_model::update_model() {
  std::cerr << "BUG: virtual function update_model() is called."
	    << std::endl;
}

// virtual function
void ml_model::ml_model::update_model(uint ml_sst_row, uint ml_sst_col) {
  std::cerr << "BUG: virtual function update_model() is called."
	    << std::endl;
}

// virtual function
void ml_model::ml_model::compute_gradient(size_t batch_num) {
  std::cerr << "BUG: virtual compute_gradient() is called." << std::endl;
}

// virtual function
void ml_model::ml_model::compute_gradient(size_t batch_num,
					  double* src_model_ptr,
					  double* dst_grad_ptr) {
  std::cerr << "BUG: virtual compute_gradient() is called." << std::endl;
}

// virtual function
double ml_model::ml_model::compute_error(const utils::images_t& images,
					 const utils::labels_t& labels) {
  std::cerr << "BUG: virtual compute_error() is called." << std::endl;
  return -1.0;
}

// virtual function
double ml_model::ml_model::compute_loss(const utils::images_t& images,
					const utils::labels_t& labels) {
  std::cerr << "BUG: virtual compute_loss() is called." << std::endl;
  return -1.0;
}

/*** setters ***/
void ml_model::ml_model::push_back_to_grad_matrix(
					   std::vector<double*> grad_row) {
  grad_matrix.push_back(grad_row);
}

void ml_model::ml_model::initialize_model_mem_with_zero() {
    utils::zero_arr(model, model_size);
}

void ml_model::ml_model::init_model(double* model, std::string full_path) {
  cnpy::NpyArray npy = cnpy::npy_load(full_path);
  assert (npy.shape[1] == get_model_size() && npy.shape[0] == 1);
  std::cout << "input W shape= (" << npy.shape[0] <<
    ", " << npy.shape[1] << ")" << std::endl;
  double* npy_arr = npy.data<double>();
  for (size_t i = 0; i < npy.shape[0] * npy.shape[1]; i++) {
    model[i] = npy_arr[i];
  }
}

void ml_model::ml_model::copy_model(double* src,
				    double* dst,
				    size_t len) {
  for (size_t i = 0; i < len; ++i) {
    dst[i] = src[i];
  }
}

// virtual function
void ml_model::ml_model::set_model_mem(double* model) {
  std::cerr << "BUG: virtual function set_model_mem() is called."
	    << std::endl;
}

// virtual function
void ml_model::ml_model::set_gradient_mem(double* gradient) {
  std::cerr << "BUG: virtual function set_gradient_mem() is called."
	    << std::endl;
}

void ml_model::ml_model::save_npy_model() const {
  std::string worker_dir = std::to_string(dataset.num_parts) + "workers";
  std::cout << "Saving a new sgd_w_opt.npy..." << std::endl;
  cnpy::npy_save(dataset.data_path + "/" + worker_dir + "/sgd_w_opt.npy",
		 model, {dataset.training_labels.num_classes,
			 dataset.training_images.num_pixels}, "w");
}

/*** getters ***/
size_t ml_model::ml_model::get_num_part_images() const {
    return dataset.training_images.num_part_images;
}

size_t ml_model::ml_model::get_num_total_images() const {
    return dataset.training_images.num_total_images;
}

std::string ml_model::ml_model::get_init_model_file() const {
  return init_model_file;
}

std::string ml_model::ml_model::get_model_name() const {
  return ml_model_name;
}

bool ml_model::ml_model::get_has_buffers() const {
  return this->has_buffers;
}

bool ml_model::ml_model::get_is_worker() const {
  return this->is_worker;
}

size_t ml_model::ml_model::get_num_batches() const {
  return get_num_batches(dataset.training_images);
}

size_t ml_model::ml_model::get_num_batches(const utils::images_t& images) const {
    return images.num_part_images / batch_size;
}

size_t ml_model::ml_model::get_model_size() const {
  return model_size;
}

double* ml_model::ml_model::get_model() const {
  return model;
}

double* ml_model::ml_model::get_gradient() const {
  return gradient;
}

double* ml_model::ml_model::get_gradient(uint ml_sst_row, uint ml_sst_col) const {
  return grad_matrix[ml_sst_row-1][ml_sst_col-1];
}
