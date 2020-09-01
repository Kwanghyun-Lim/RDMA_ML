#include "dnn.hpp"

#include <assert.h>
#include <cblas.h>
#include <cmath>
#include <cstdio>
#include <assert.h>
#include <iostream>

ml_model::relu::relu(size_t x_len, const size_t batch_size) : x_len(x_len), batch_size(batch_size) {
  Y = new double[batch_size * x_len];
  dL_dX = new double[batch_size * x_len];
}

double* ml_model::relu::forward(double* X) {
  for (size_t i = 0; i < x_len * batch_size; ++i) {
    if (X[i] <= 0) {
      Y[i] = 0;
    } else {
      Y[i] = X[i];
    }
  }
  return Y;
}

double* ml_model::relu::backward(double* dL_dY) {
  for (size_t i = 0; i < x_len * batch_size; ++i) {
    if (Y[i] <= 0) {
      dL_dX[i] = 0;
    } else {
      dL_dX[i] = dL_dY[i]; // 1 * dL_dY where 1 is dY_dX
    }
  }
  return dL_dX; // size is (batch_size, x_len)
}

ml_model::affine::affine(size_t W_row_len, size_t W_col_len, const size_t batch_size)
  : W_row_len(W_row_len), W_col_len(W_col_len), b_len(W_col_len), batch_size(batch_size) {
  Y = new double[batch_size * W_col_len];
  dL_dX = new double[batch_size * W_row_len];
}

double* ml_model::affine::forward(double* X) {
  this->X = X;
  utils::matrix_t _X(X, batch_size, W_row_len);
  utils::matrix_t _W(W, W_row_len, W_col_len);
  utils::matrix_t _Y(Y, batch_size, W_col_len);
  utils::mat_mul(_X, NoTrans,_W, NoTrans, _Y);
  
  // Y += B
  utils::matrix_t _b(b, 1, b_len);
  for (size_t i = 0; i < batch_size; ++i) {
    submat_add_assign(1.0, _Y, i, 0, 1, _Y.num_cols,
  		      1.0, _b, 0, 0, _b.num_rows, _b.num_cols);
  }
  return Y;
}

// The parameter, X, is the whole input X matrix (e.g., the whole images matrix).
double* ml_model::affine::forward(double* X, const size_t batch_num, const size_t total_num_data) {
  this->X = X;
  utils::matrix_t _X(X, W_row_len, total_num_data);
  utils::matrix_t _W(W, W_row_len, W_col_len);
  utils::matrix_t _Y(Y, batch_size, W_col_len);
  utils::submat_mul(_X, 0, batch_num * batch_size, _X.num_rows, batch_size, Trans,
  		    _W, 0, 0, _W.num_rows, _W.num_cols, NoTrans, _Y);
  // Y += B
  utils::matrix_t _b(b, 1, b_len);
  for (size_t i = 0; i < batch_size; ++i) {
    submat_add_assign(1.0, _Y, i, 0, 1, _Y.num_cols,
  		      1.0, _b, 0, 0, _b.num_rows, _b.num_cols);
  }
  return Y;
}

double* ml_model::affine::backward(double* dL_dY) {
  utils::matrix_t _X(X, batch_size, W_row_len);
  utils::matrix_t _dL_dY(dL_dY, batch_size, W_col_len);
  utils::matrix_t _dL_dW(dL_dW, W_row_len, W_col_len);
  utils::mat_mul(_X, Trans, _dL_dY, NoTrans, _dL_dW);
  
  utils::matrix_t _dL_db(dL_db, 1, b_len);
  utils::zero_arr(dL_db, b_len);
  for (size_t i = 0; i < batch_size; ++i) {
    submat_add_assign(1.0, _dL_db, 0, 0, _dL_db.num_rows, _dL_db.num_cols,
		      1.0, _dL_dY, i, 0, 1, _dL_dY.num_cols);
  }
  
  utils::matrix_t _W(W, W_row_len, W_col_len);
  utils::matrix_t _dL_dX(dL_dX, batch_size, W_row_len);
  utils::mat_mul(_dL_dY, NoTrans, _W, Trans, _dL_dX);
   return dL_dX;
}

// This function is for the first hidden layer. The X here is the whole training samples.
//  We are reading only the subset of them as much as batch size.
double* ml_model::affine::backward(double* dL_dY, const size_t batch_num, const size_t total_num_data) {
  utils::matrix_t _X(X, W_row_len, total_num_data);
  utils::matrix_t _dL_dY(dL_dY, batch_size, W_col_len);
  utils::matrix_t _dL_dW(dL_dW, W_row_len, W_col_len);
  utils::submat_mul(_X, 0, batch_num * batch_size, _X.num_rows, batch_size, NoTrans, _dL_dY, 0, 0, _dL_dY.num_rows, _dL_dY.num_cols, NoTrans, _dL_dW);
  
  utils::matrix_t _dL_db(dL_db, 1, b_len);
  utils::zero_arr(dL_db, b_len);
  for (size_t i = 0; i < batch_size; ++i) {
    submat_add_assign(1.0, _dL_db, 0, 0, _dL_db.num_rows, _dL_db.num_cols,
		      1.0, _dL_dY, i, 0, 1, _dL_dY.num_cols);
  }

  utils::matrix_t _W(W, W_row_len, W_col_len);
  utils::matrix_t _dL_dX(dL_dX, batch_size, W_row_len);
  utils::mat_mul(_dL_dY, NoTrans, _W, Trans, _dL_dX);

  return dL_dX;
}

ml_model::softmax::softmax(size_t y_len, const size_t batch_size) : y_len(y_len), batch_size(batch_size) {
  Y = new double[batch_size * y_len];
  dL_dX = new double[batch_size * y_len];
}

double* ml_model::softmax::forward(double* X) {
  utils::softmax(X, Y, batch_size, y_len);
  return Y;
}

double* ml_model::softmax::backward(double* Y_labels, const size_t batch_num, const size_t total_num_data) {
  utils::matrix_t _Y_labels(Y_labels, y_len, total_num_data);
  utils::matrix_t _dL_dX(dL_dX, batch_size, y_len);
  utils::submat_trans(_Y_labels, 0, batch_num * batch_size, y_len, batch_size, _dL_dX);
  
  utils::matrix_t _Y(Y, batch_size, y_len);
  utils::mat_add_assign(-1/(float)batch_size, _dL_dX, 1/(float)batch_size, _Y);
  return dL_dX;
}

ml_model::deep_neural_network::deep_neural_network(
	const std::vector<uint32_t> layer_size_vec,
	const uint32_t num_layers,
        const utils::reader_t& dataset_loader,
        const double alpha,
        const size_t batch_size,
	const bool is_worker)
        : layer_size_vec(layer_size_vec),
	  num_layers(num_layers),
          dataset(dataset_loader()),
          alpha(alpha),
          batch_size(batch_size),
          is_worker(is_worker) {
  for (int i = 0; i < num_layers - 1; ++i) {
    model_size += layer_size_vec[i] * layer_size_vec[i+1] + layer_size_vec[i+1]; 
  }
  
  for (int i = 0; i < num_layers - 1; ++i) {
    affine_layers.push_back(new affine(layer_size_vec[i], layer_size_vec[i+1], batch_size));
  }

  // exclude the last layer since we use softmax as the last layer.
  for (int i = 0; i < num_layers - 2; ++i) {
    relu_layers.push_back(new relu(layer_size_vec[i+1], batch_size));
  }

  last_layer = new softmax(layer_size_vec[num_layers-1], batch_size);
}

void ml_model::deep_neural_network::train(const size_t num_epochs) {
  const size_t num_batches = get_num_batches(dataset.training_images);
  for (size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
    for (size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      compute_gradient(batch_num);
      update_model();
    }
  }
}

double ml_model::deep_neural_network::training_error() {
    return compute_error(dataset.training_images, dataset.training_labels);
}

double ml_model::deep_neural_network::training_loss() {
    return compute_loss(dataset.training_images, dataset.training_labels);
}

double ml_model::deep_neural_network::test_error() {
    return compute_error(dataset.test_images, dataset.test_labels);
}

void ml_model::deep_neural_network::compute_gradient(const size_t batch_num) {
    const utils::images_t& images = dataset.training_images;
    const utils::labels_t& labels = dataset.training_labels;

    // feed forward
    double* Y =	affine_layers[0]->forward(images.arr.get(), batch_num, images.num_total_images);
    utils::matrix_t affine_Y(Y, affine_layers[0]->batch_size, affine_layers[0]->W_col_len);
    
    for (size_t i = 1; i < num_layers - 1; ++i) {
      Y = relu_layers[i-1]->forward(Y);
      utils::matrix_t relu_Y(Y, relu_layers[i-1]->batch_size, relu_layers[i-1]->x_len);
      
      Y = affine_layers[i]->forward(Y);
      utils::matrix_t affine_Y2(Y, affine_layers[i]->batch_size, affine_layers[i]->W_col_len);
    }
    
    Y = last_layer->forward(Y);
    utils::matrix_t softmax_Y(Y, last_layer->batch_size, last_layer->y_len);

    // back propagation
    double* dL_dX = last_layer->backward(labels.arr.get(), batch_num, images.num_total_images);
    utils::matrix_t softmax_dL_dX(dL_dX, last_layer->batch_size, last_layer->y_len);
    
    for (size_t i = num_layers - 2; i > 0; --i) {
	dL_dX = affine_layers[i]->backward(dL_dX);
	utils::matrix_t affine_dL_dX(dL_dX, affine_layers[i]->batch_size, affine_layers[i]->W_row_len);
	
	dL_dX = relu_layers[i-1]->backward(dL_dX);
	utils::matrix_t relu_dL_dX(dL_dX, relu_layers[i-1]->batch_size, relu_layers[i-1]->x_len);
    }
    
    dL_dX = affine_layers[0]->backward(dL_dX, batch_num, images.num_total_images);
    utils::matrix_t affine_dL_dX(dL_dX, affine_layers[0]->batch_size, affine_layers[0]->W_row_len);
    
}

void ml_model::deep_neural_network::update_model() {
  utils::matrix_t _gradient(gradients[0], 1, model_size);
  utils::matrix_t _model(model, 1, model_size);
  cblas_daxpy(model_size, -alpha, gradients[0], 1, model, 1);
}

void ml_model::deep_neural_network::update_model(uint ml_sst_row) {
  utils::matrix_t _gradient(gradients[ml_sst_row-1], 1, model_size);
  utils::matrix_t _model(model, 1, model_size);
  cblas_daxpy(model_size, -alpha, gradients[ml_sst_row-1], 1, model, 1);
}

void ml_model::deep_neural_network::set_model_mem(double* model) {
  this->model = model;
  double* model_seek = model;
  for (int i = 0; i < num_layers - 1; ++i) {
    size_t W_b_size
      = affine_layers[i]->W_row_len * affine_layers[i]->W_col_len + affine_layers[i]->b_len;

    affine_layers[i]->W = model_seek;
    affine_layers[i]->b = model_seek + W_b_size - affine_layers[i]->b_len;
    model_seek += W_b_size;
  }
}

void ml_model::deep_neural_network::initialize_model_mem_with_zero() {
    utils::zero_arr(this->model, model_size);
}

void ml_model::deep_neural_network::set_gradient_mem(double* gradient) {
    assert (model != NULL);
    this->gradient = gradient;
    double* gradient_seek = gradient;
    if (is_worker) {
      for (int i = 0; i < num_layers - 1; ++i) {
	size_t W_b_size
	  = affine_layers[i]->W_row_len * affine_layers[i]->W_col_len + affine_layers[i]->b_len;

	affine_layers[i]->dL_dW = gradient_seek;
	affine_layers[i]->dL_db = gradient_seek + W_b_size - affine_layers[i]->b_len;
	gradient_seek += W_b_size;
      }
    }
}

void ml_model::deep_neural_network::push_back_to_grad_vec(double* gradient) {
  assert (model != NULL);
  gradients.push_back(gradient);
  double* gradient_seek = gradient;
  if (is_worker) {
    for (int i = 0; i < num_layers - 1; ++i) {
      size_t W_b_size
	= affine_layers[i]->W_row_len * affine_layers[i]->W_col_len + affine_layers[i]->b_len;

      affine_layers[i]->dL_dW = gradient_seek;
      affine_layers[i]->dL_db = gradient_seek + W_b_size - affine_layers[i]->b_len;
      gradient_seek += W_b_size;
    }
  }
}

double* ml_model::deep_neural_network::get_model() const {
  return model;
}

size_t ml_model::deep_neural_network::get_model_size() const {
    return model_size;
}

size_t ml_model::deep_neural_network::get_num_batches() const {
    return dataset.training_images.num_part_images / batch_size;
}

size_t ml_model::deep_neural_network::get_num_batches(const utils::images_t& images) const {
    return images.num_total_images / batch_size;
}

size_t ml_model::deep_neural_network::get_num_images() const {
    return dataset.training_images.num_total_images;
}

double ml_model::deep_neural_network::compute_error(const utils::images_t& images, const utils::labels_t& labels) {
    size_t num_incorrect = 0;
    const size_t num_batches = get_num_batches(images);
    for (size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      // feed forward
      double* Y = affine_layers[0]->forward(images.arr.get(), batch_num, images.num_total_images);
      for (size_t i = 1; i < num_layers - 1; ++i) {
	Y = relu_layers[i-1]->forward(Y);
	Y = affine_layers[i]->forward(Y);
      }
      Y = last_layer->forward(Y);
      utils::matrix_t _Y(Y, batch_size, last_layer->y_len);
      for (size_t i = 0; i < batch_size; ++i) {
	if (cblas_idamax(labels.num_classes, _Y.arr + i*_Y.num_cols, 1) 
	    != cblas_idamax(labels.num_classes, labels.arr.get() + i + (batch_num * batch_size), images.num_total_images)) {
	  num_incorrect++;
	}
      }
    }
  
    return (double)num_incorrect/images.num_total_images;
}

double ml_model::deep_neural_network::compute_loss(const utils::images_t& images, const utils::labels_t& labels) {
    double loss = 0.0;
    double* a;
    double* b;
    const size_t num_batches = get_num_batches(images);
    for (size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      // feed forward
      double* Y = affine_layers[0]->forward(images.arr.get(), batch_num, images.num_total_images);
      for (size_t i = 1; i < num_layers - 1; ++i) {
	Y = relu_layers[i-1]->forward(Y);
	Y = affine_layers[i]->forward(Y);
      }
      Y = last_layer->forward(Y);

      for (size_t i = 0; i < batch_size; ++i) {
	for (size_t j = 0; j < labels.num_classes; ++j) {
	  a = labels.arr.get() + (batch_num * batch_size + i) + (images.num_total_images * j);
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

void ml_model::deep_neural_network::init_model(double* model, std::string full_path) {
  cnpy::NpyArray npy = cnpy::npy_load(full_path);
  assert (npy.shape[1] == get_model_size() && npy.shape[0] == 1);
  std::cout << "input W shape= (" << npy.shape[0] << ", " << npy.shape[1] << ")" << std::endl;
  double* npy_arr = npy.data<double>();
  for (size_t i = 0; i < npy.shape[0] * npy.shape[1]; i++) {
    model[i] = npy_arr[i];
  }
}
