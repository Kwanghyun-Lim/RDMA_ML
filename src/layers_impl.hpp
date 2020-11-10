#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>

#include "utils/utils.hpp"
#include "utils/cnpy.hpp"

namespace ml_model {
class relu {
public:
  relu(size_t x_len, const size_t batch_size);
  ~relu() { std::cout << "Relu destructor" << std::endl; };
  // X is a pointer to the matrix of (batch_size, features)
  // in the previous affine object
  double* forward(double* X);
  double* backward(double* dL_dY);

  // x_len only counts the number of features of one sample.
  // It doesn't count batch_size.
  size_t x_len;
  double* Y; // Y = X*W + B 
  double* dL_dX; // dL/dX (batch_size, x_len)
  const size_t batch_size;
};
  
relu::relu(size_t x_len, const size_t batch_size)
  : x_len(x_len), batch_size(batch_size) {
  Y = new double[batch_size * x_len];
  dL_dX = new double[batch_size * x_len];
}

double* relu::forward(double* X) {
  for (size_t i = 0; i < x_len * batch_size; ++i) {
    if (X[i] <= 0) {
      Y[i] = 0;
    } else {
      Y[i] = X[i];
    }
  }
  return Y;
}
  
double* relu::backward(double* dL_dY) {
  for (size_t i = 0; i < x_len * batch_size; ++i) {
    if (Y[i] <= 0) {
      dL_dX[i] = 0;
    } else {
      dL_dX[i] = dL_dY[i]; // 1 * dL_dY where 1 is dY_dX
    }
  }
  return dL_dX; // size is (batch_size, x_len)
}
  
class affine {
public:
  affine(size_t W_row_len, size_t W_col_len, const size_t batch_size);
  double* forward(double* X);
  // for the first layer
  double* forward(double* X,
		  const size_t batch_num,
		  const size_t total_num_data); 
  double* backward(double* dL_dY);
  // for the first layer
  double* backward(double* dL_dY,
		   const size_t batch_num,
		   const size_t total_num_data); 

  double* X; // It points to the precious layer's Y
  
  double* W; // It points to some offset within the ml_sst row
  double* b; // It points to some offset within the ml_sst row
  // (batch_size, W_row_len) * (W_row_len, W_col_len)
  // + (batch_size, W_col_len)
  double* Y; // Y = X*W + B 

  double* dL_dW; // It points to some offset within the ml_sst row
  double* dL_db; // It points to some offset within the ml_sst row
  double* dL_dX; // dL/dX

  size_t W_row_len;
  size_t W_col_len;
  size_t b_len;
  const size_t batch_size;
};

affine::affine(size_t W_row_len,
			 size_t W_col_len,
			 const size_t batch_size)
  : W_row_len(W_row_len), W_col_len(W_col_len),
    b_len(W_col_len), batch_size(batch_size) {
  Y = new double[batch_size * W_col_len];
  dL_dX = new double[batch_size * W_row_len];
}

double* affine::forward(double* X) {
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

// The parameter, X, is the whole input X matrix
// (e.g., the whole images matrix).
double* affine::forward(double* X,
				  const size_t batch_num,
				  const size_t total_num_data) {
  this->X = X;
  utils::matrix_t _X(X, W_row_len, total_num_data);
  utils::matrix_t _W(W, W_row_len, W_col_len);
  utils::matrix_t _Y(Y, batch_size, W_col_len);
  utils::submat_mul(_X, 0, batch_num * batch_size,
		    _X.num_rows, batch_size, Trans,
  		    _W, 0, 0, _W.num_rows, _W.num_cols, NoTrans, _Y);
  // Y += B
  utils::matrix_t _b(b, 1, b_len);
  for (size_t i = 0; i < batch_size; ++i) {
    submat_add_assign(1.0, _Y, i, 0, 1, _Y.num_cols,
  		      1.0, _b, 0, 0, _b.num_rows, _b.num_cols);
  }
  return Y;
}

double* affine::backward(double* dL_dY) {
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

// This function is for the first hidden layer.
// The X here is the whole training samples.
//  We are reading only the subset of them as much as batch size.
double* affine::backward(double* dL_dY,
				   const size_t batch_num,
				   const size_t total_num_data) {
  utils::matrix_t _X(X, W_row_len, total_num_data);
  utils::matrix_t _dL_dY(dL_dY, batch_size, W_col_len);
  utils::matrix_t _dL_dW(dL_dW, W_row_len, W_col_len);
  utils::submat_mul(_X, 0, batch_num * batch_size, _X.num_rows,
		    batch_size, NoTrans, _dL_dY, 0, 0,
		    _dL_dY.num_rows, _dL_dY.num_cols, NoTrans, _dL_dW);
  
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

class softmax {
public:
  softmax(size_t y_len, const size_t batch_size);
  double* forward(double* X);
  double* backward(double* Y_labels,
		   const size_t batch_num,
		   const size_t total_num_data);

  double* Y; // Y is a prediction matrix of (batch_size, num_classes)
  double* dL_dX;
  
  size_t y_len; // doesn't count batch_size
  const size_t batch_size;
};

softmax::softmax(size_t y_len, const size_t batch_size)
  : y_len(y_len), batch_size(batch_size) {
  Y = new double[batch_size * y_len];
  dL_dX = new double[batch_size * y_len];
}

double* softmax::forward(double* X) {
  utils::softmax(X, Y, batch_size, y_len);
  return Y;
}

double* softmax::backward(double* Y_labels,
				    const size_t batch_num,
				    const size_t total_num_data) {
  utils::matrix_t _Y_labels(Y_labels, y_len, total_num_data);
  utils::matrix_t _dL_dX(dL_dX, batch_size, y_len);
  utils::submat_trans(_Y_labels, 0, batch_num * batch_size,
		      y_len, batch_size, _dL_dX);
  
  utils::matrix_t _Y(Y, batch_size, y_len);
  utils::mat_add_assign(-1/(float)batch_size, _dL_dX,
			1/(float)batch_size, _Y);
  return dL_dX;
}
} // namespace ml_model
