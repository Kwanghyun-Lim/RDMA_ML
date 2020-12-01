#include "log_reg.hpp"

#include <assert.h>
#include <cblas.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

ml_model::multinomial_log_reg::multinomial_log_reg(
        const utils::reader_t& dataset_loader,
        const double alpha,
	const double gamma,
	double decay,
	const size_t batch_size,
	const std::string init_model_file,
	const bool has_buffers,
	const bool is_worker)
  : ml_model(dataset_loader,
	     std::string("log_reg"),
	     init_model_file,
	     has_buffers,
	     is_worker,
	     (alpha / dataset.num_parts),
	     batch_size,
	     (batch_size * dataset.num_parts)),
    full_gradient(new double[model_size]),
    full_predicted_labels(std::make_unique<double[]>(
                          dataset.training_labels.num_classes *
			  dataset.training_images.num_total_images)),
    predicted_labels(std::make_unique<double[]>(
		       dataset.training_labels.num_classes * batch_size)),
    
    gamma(gamma),
    decay(decay),
    num_model_updates(0) {
  model_size = (dataset.training_labels.num_classes
		      * dataset.training_images.num_pixels);
}

ml_model::multinomial_log_reg::~multinomial_log_reg() {
  std::cout << "m_log_reg destructor called." << std::endl;
}

/*** main methods ***/
double ml_model::multinomial_log_reg::compute_error(
					  const utils::images_t& images,
					  const utils::labels_t& labels) {
    predicted_labels = std::make_unique<double[]>(
            labels.num_classes * images.num_total_images);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                labels.num_classes, images.num_total_images,
		images.num_pixels,
		1, model, images.num_pixels,
		images.arr.get(), images.num_total_images,
		0, predicted_labels.get(), images.num_total_images);
    utils::softmax(predicted_labels.get(),
                   labels.num_classes, images.num_total_images);
    size_t num_incorrect = 0;
    for(size_t i = 0; i < images.num_total_images; ++i) {
        if (cblas_idamax(labels.num_classes,
			 predicted_labels.get() + i,
			 images.num_total_images)
	    != cblas_idamax(labels.num_classes,
			    labels.arr.get() + i,
			    images.num_total_images)) {
	    num_incorrect++;
	}
    }
    return (double)num_incorrect/images.num_total_images;
}

double ml_model::multinomial_log_reg::compute_loss(
					  const utils::images_t& images,
					  const utils::labels_t& labels) {
    predicted_labels = std::make_unique<double[]>(
            labels.num_classes * images.num_total_images);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                labels.num_classes, images.num_total_images,
		images.num_pixels,
		1, model, images.num_pixels,
		images.arr.get(), images.num_total_images,
		0, predicted_labels.get(), images.num_total_images);
    utils::softmax(predicted_labels.get(),
                   labels.num_classes, images.num_total_images);

    double loss = 0.0;
    for (size_t i = 0; i < labels.num_classes * images.num_total_images; ++i) {
      if (*(labels.arr.get() + i) == 1) {
	loss += -std::log(*(predicted_labels.get() + i));
      }
    }
    loss /= images.num_total_images;

    for (size_t i = 0; i < labels.num_classes * images.num_pixels; ++i) {
      loss += (gamma / 2) * std::pow(model[i], 2);
    }
    return loss;
}

void ml_model::multinomial_log_reg::update_model() {
    double decayed_alpha = decay_alpha();
    cblas_daxpy(model_size, -decayed_alpha, gradient, 1, model, 1);
    num_model_updates++;
}

void ml_model::multinomial_log_reg::update_model(uint ml_sst_row,
						 uint ml_sst_col) {
    double decayed_alpha = decay_alpha();
    cblas_daxpy(model_size, -decayed_alpha,
		grad_matrix[ml_sst_row-1][ml_sst_col-1], 1, model, 1);
    num_model_updates++;
}

// For single node and distributed nodes without buffers
void ml_model::multinomial_log_reg::compute_gradient(size_t batch_num) {
  compute_gradient(batch_num, model, gradient);
}

// For distributed nodes with buffers
void ml_model::multinomial_log_reg::compute_gradient(
					      size_t batch_num,
					      double* src_model_ptr,
					      double* dst_grad_ptr) {
    const utils::images_t& images = dataset.training_images;
    const utils::labels_t& labels = dataset.training_labels;
    // convert from local batch_num to global batch_num
    batch_num += dataset.part_num * get_num_batches();
    
    utils::submatrix_multiply(CblasNoTrans, CblasNoTrans,
			      src_model_ptr, images.arr.get(),
			      predicted_labels.get(),
			      0, 0, 0, batch_num * batch_size,
			      labels.num_classes, batch_size,
			      images.num_pixels,
			      images.num_pixels, images.num_total_images,
			      1, 0);
    utils::softmax(predicted_labels.get(),
                   labels.num_classes, batch_size);
    cblas_dgeadd(CblasRowMajor, labels.num_classes, batch_size,
		 -1, labels.arr.get() + batch_num * batch_size,
		 images.num_total_images,
		 1, predicted_labels.get(), batch_size);
    utils::submatrix_multiply(CblasNoTrans, CblasTrans,
                              predicted_labels.get(), images.arr.get(),
			      dst_grad_ptr,
                              0, 0, 0, batch_num * batch_size,
                              labels.num_classes, images.num_pixels,
			      batch_size, 
			      batch_size, images.num_total_images,
			      1/(double)batch_size, 0);
    cblas_daxpy(model_size, gamma, src_model_ptr, 1, dst_grad_ptr, 1);
}


/*** setters ***/
void ml_model::multinomial_log_reg::set_model_mem(double* model) {
    this->model = model;
}

void ml_model::multinomial_log_reg::set_gradient_mem(double* gradient) {
    this->gradient = gradient;
}

/*** getters for statistics ***/
double ml_model::multinomial_log_reg::get_loss_opt() const {
  std::ifstream loss_opt_file;
  std::string worker_dir = std::to_string(dataset.num_parts) + "workers";
  loss_opt_file.open(dataset.data_path + "/svrg_loss_opt.txt");
  std::string loss_opt_str;
  loss_opt_file >> loss_opt_str;
  double loss_opt = std::stod(loss_opt_str);
  return loss_opt;
}

double ml_model::multinomial_log_reg::gradient_norm() {
  compute_full_gradient(model);
  double norm = 0.0;
  for (size_t i = 0; i < get_model_size(); ++i) {
    norm += std::pow(full_gradient[i], 2);
  }
  norm = sqrt(norm);
  return norm;
}

double ml_model::multinomial_log_reg::distance_to_optimum() {
  cnpy::NpyArray arr; 
  arr = cnpy::npy_load(dataset.data_path + "/svrg_w_opt.npy");
  double* model_opt = arr.data<double>();
  double norm = 0.0;
  for (size_t i = 0; i < get_model_size(); ++i) {
    norm += std::pow(model[i] - model_opt[i], 2);
  }
  norm = sqrt(norm);
  return norm;
}

double* ml_model::multinomial_log_reg::get_full_gradient() const {
  return full_gradient;
}

/*** helpers ***/
void ml_model::multinomial_log_reg::compute_full_gradient(
					       double* model_ptr) {
    const utils::images_t& images = dataset.training_images;
    const utils::labels_t& labels = dataset.training_labels;
    
    utils::submatrix_multiply(CblasNoTrans, CblasNoTrans,
			      model_ptr, images.arr.get(),
			      full_predicted_labels.get(),
			      0, 0, 0, 0,
			      labels.num_classes,
			      images.num_total_images, images.num_pixels,
			      images.num_pixels, images.num_total_images,
			      1, 0);
    utils::softmax(full_predicted_labels.get(),
                   labels.num_classes, images.num_total_images);
    cblas_dgeadd(CblasRowMajor, labels.num_classes,
		 images.num_total_images,
		 -1, labels.arr.get(), images.num_total_images,
		 1, full_predicted_labels.get(), images.num_total_images);
    utils::submatrix_multiply(CblasNoTrans, CblasTrans,
                              full_predicted_labels.get(),
			      images.arr.get(), full_gradient,
                              0, 0, 0, 0,
                              labels.num_classes, images.num_pixels,
			      images.num_total_images, 
			      images.num_total_images,
			      images.num_total_images,
			      1/(double)images.num_total_images, 0);
    cblas_daxpy(model_size, gamma, model_ptr, 1, full_gradient, 1);
}

double ml_model::multinomial_log_reg::decay_alpha() {
  double epoch = floor(num_model_updates /
		       (get_num_batches() * dataset.num_parts));
  return alpha * pow(decay, epoch);
}
