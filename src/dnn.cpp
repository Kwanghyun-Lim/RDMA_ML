#include <assert.h>
#include <cblas.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>

#include "dnn.hpp"

ml_model::deep_neural_network::deep_neural_network(
	const std::vector<uint32_t> layer_size_vec,
	const uint32_t num_layers,
        const utils::reader_t& dataset_loader,
        const double alpha,
        const size_t batch_size,
	const std::string init_model_file,
	const bool has_buffers,
	const bool is_worker)
  : ml_model(dataset_loader,
	     std::string("dnn"),
	     init_model_file,
	     has_buffers,
	     is_worker,
	     0, // model size in ml_model will be updated in constructor.
	     (alpha / dataset.num_parts),
	     batch_size,
	     (batch_size * dataset.num_parts)),
    layer_size_vec(layer_size_vec),
    num_layers(num_layers) {
  for (int i = 0; i < num_layers - 1; ++i) {
    model_size += layer_size_vec[i] * layer_size_vec[i+1] +
      layer_size_vec[i+1]; 
  }
  
  for (int i = 0; i < num_layers - 1; ++i) {
    affine_layers.push_back(new affine(layer_size_vec[i],
				       layer_size_vec[i+1], batch_size));
  }

  // exclude the last layer since we use softmax as the last layer.
  for (int i = 0; i < num_layers - 2; ++i) {
    relu_layers.push_back(new relu(layer_size_vec[i+1], batch_size));
  }

  last_layer = new softmax(layer_size_vec[num_layers-1], batch_size);
}

/*** main methods ***/
double ml_model::deep_neural_network::compute_error(
				  const utils::images_t& images,
				  const utils::labels_t& labels) {
    size_t num_incorrect = 0;
    const size_t num_batches = get_num_batches(images);
    for (size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      // feed forward
      double* Y = affine_layers[0]->forward(images.arr.get(),
					    batch_num,
					    images.num_total_images);
      for (size_t i = 1; i < num_layers - 1; ++i) {
	Y = relu_layers[i-1]->forward(Y);
	Y = affine_layers[i]->forward(Y);
      }
      Y = last_layer->forward(Y);
      utils::matrix_t _Y(Y, batch_size, last_layer->y_len);
      for (size_t i = 0; i < batch_size; ++i) {
	if (cblas_idamax(labels.num_classes, _Y.arr + i*_Y.num_cols, 1) 
	    != cblas_idamax(labels.num_classes,
			    labels.arr.get() + i
			    + (batch_num * batch_size),
			    images.num_total_images)) {
	  num_incorrect++;
	}
      }
    }
  
    return (double)num_incorrect/images.num_total_images;
}

double ml_model::deep_neural_network::compute_loss(
				     const utils::images_t& images,
				     const utils::labels_t& labels) {
    double loss = 0.0;
    double* a;
    double* b;
    const size_t num_batches = get_num_batches(images);
    for (size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      // feed forward
      double* Y = affine_layers[0]->forward(images.arr.get(),
					    batch_num,
					    images.num_total_images);
      for (size_t i = 1; i < num_layers - 1; ++i) {
	Y = relu_layers[i-1]->forward(Y);
	Y = affine_layers[i]->forward(Y);
      }
      Y = last_layer->forward(Y);

      for (size_t i = 0; i < batch_size; ++i) {
	for (size_t j = 0; j < labels.num_classes; ++j) {
	  a = labels.arr.get() + (batch_num * batch_size + i) +
	    (images.num_total_images * j);
	  b = Y + (labels.num_classes * i) + j;
	  if (*a == 1) {
	    loss += -std::log(*b + 1e-7);
	  }
	}
      }
   }
   loss /= images.num_total_images;
   return loss;
}

// For single node
void ml_model::deep_neural_network::update_model() {
  utils::matrix_t _gradient(gradient, 1, model_size);
  utils::matrix_t _model(model, 1, model_size);
  cblas_daxpy(model_size, -alpha, gradient, 1, model, 1);
}

// For distributed nodes
void ml_model::deep_neural_network::update_model(uint ml_sst_row,
						 uint ml_sst_col) {
  utils::matrix_t _gradient(grad_matrix[ml_sst_row-1][ml_sst_col-1],
			    1, model_size);
  utils::matrix_t _model(model, 1, model_size);
  cblas_daxpy(model_size, -alpha,
	      grad_matrix[ml_sst_row-1][ml_sst_col-1], 1, model, 1);
}

// For single node and distributed nodes without buffers
void ml_model::deep_neural_network::compute_gradient(size_t batch_num) {
  compute_gradient(batch_num, model, gradient);
}

// For distributed nodes with buffers
void ml_model::deep_neural_network::compute_gradient(
					    size_t batch_num,
					    double* src_model_ptr,
					    double* dst_grad_ptr) {
    const utils::images_t& images = dataset.training_images;
    const utils::labels_t& labels = dataset.training_labels;
    
    // convert from local batch_num to global batch_num
    batch_num += dataset.part_num * get_num_batches(); 

    // set_model_mem() can be used when we introduce buffers for models,
    // which can be our future work.
    // set_model_mem(src_model_ptr);
    set_gradient_mem(dst_grad_ptr);
    
    // feed forward
    double* Y =	affine_layers[0]->forward(images.arr.get(),
					  batch_num,
					  images.num_total_images);
    utils::matrix_t affine_Y(Y, affine_layers[0]->batch_size,
			     affine_layers[0]->W_col_len);
    
    for (size_t i = 1; i < num_layers - 1; ++i) {
      Y = relu_layers[i-1]->forward(Y);
      utils::matrix_t relu_Y(Y, relu_layers[i-1]->batch_size,
			     relu_layers[i-1]->x_len);
      
      Y = affine_layers[i]->forward(Y);
      utils::matrix_t affine_Y2(Y, affine_layers[i]->batch_size,
				affine_layers[i]->W_col_len);
    }
    
    Y = last_layer->forward(Y);
    utils::matrix_t softmax_Y(Y, last_layer->batch_size,
			      last_layer->y_len);

    // back propagation
    double* dL_dX = last_layer->backward(labels.arr.get(),
					 batch_num,
					 images.num_total_images);
    utils::matrix_t softmax_dL_dX(dL_dX, last_layer->batch_size,
				  last_layer->y_len);
    
    for (size_t i = num_layers - 2; i > 0; --i) {
	dL_dX = affine_layers[i]->backward(dL_dX);
	utils::matrix_t affine_dL_dX(dL_dX, affine_layers[i]->batch_size,
				     affine_layers[i]->W_row_len);
	
	dL_dX = relu_layers[i-1]->backward(dL_dX);
	utils::matrix_t relu_dL_dX(dL_dX, relu_layers[i-1]->batch_size,
				   relu_layers[i-1]->x_len);
    }
    
    dL_dX = affine_layers[0]->backward(dL_dX, batch_num,
				       images.num_total_images);
    utils::matrix_t affine_dL_dX(dL_dX, affine_layers[0]->batch_size,
				 affine_layers[0]->W_row_len);
}

/*** setters ***/
void ml_model::deep_neural_network::set_model_mem(double* model) {
  this->model = model;
  double* model_seek = model;
  for (int i = 0; i < num_layers - 1; ++i) {
    size_t W_b_size
      = affine_layers[i]->W_row_len * affine_layers[i]->W_col_len +
      affine_layers[i]->b_len;

    affine_layers[i]->W = model_seek;
    affine_layers[i]->b = model_seek + W_b_size - affine_layers[i]->b_len;
    model_seek += W_b_size;
  }
}

void ml_model::deep_neural_network::set_gradient_mem(double* gradient) {
    this->gradient = gradient;
    double* gradient_seek = gradient;
    if(is_worker) {
      for (int i = 0; i < num_layers - 1; ++i) {
	size_t W_b_size
	  = affine_layers[i]->W_row_len * affine_layers[i]->W_col_len +
	    affine_layers[i]->b_len;

	affine_layers[i]->dL_dW = gradient_seek;
	affine_layers[i]->dL_db = gradient_seek + W_b_size -
	  affine_layers[i]->b_len;
	gradient_seek += W_b_size;
      }
    }
}

/*** getters for statistics ***/
double ml_model::deep_neural_network::get_loss_opt() const {
  std::cerr << "ERROR: dnn could not do get_loss_opt()"
            << std::endl;
  exit(1);
  return -1.0;
}

double ml_model::deep_neural_network::gradient_norm() {
  std::cerr << "ERROR: dnn doesn't support gradient_norm() yet."
            << std::endl;
  exit(1);
  return -1.0;
}

double ml_model::deep_neural_network::distance_to_optimum() {
  std::cerr << "ERROR: dnn could not do get_loss_opt()"
            << std::endl;
  exit(1);
  return -1.0;
}

double* ml_model::deep_neural_network::get_full_gradient() const {
  std::cerr << "ERROR: dnn doesn't support get_full_gradient() yet."
            << std::endl;
  exit(1);
  return NULL;
}
