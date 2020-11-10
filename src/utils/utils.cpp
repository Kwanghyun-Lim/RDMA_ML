#include <cblas.h>
#include <cmath>
#include <iostream>
#include <random>

#include "utils.hpp"
#define _USE_MATH_DEFINES  // this is for PI in cmath

void utils::zero_arr(double* arr, const size_t size) {
    for(size_t i = 0; i < size; ++i, ++arr) {
        *arr = 0;
    }
}

void utils::softmax(double* x, const size_t m, const size_t n) {
    for(size_t i = 0; i < n; ++i) {
        double* y = x + i;
        double sum_exp = 0;

        double max;
        for(size_t j = 0; j < m; ++j, y += n) {
            if(j == 0) {
                max = *y;
            } else {
                if(*y > max) {
                    max = *y;
                }
            }
        }
	
        y = x + i;
        for(size_t j = 0; j < m; ++j, y += n) {
            *y -= max;
        }

        y = x + i;
        for(size_t j = 0; j < m; ++j, y += n) {
            *y = std::exp(*y);
            sum_exp += *y;
        }

        y = x + i;
        for(size_t j = 0; j < m; ++j, y += n) {
            *y /= sum_exp;
        }
    }
}

// numerically stable softmax
// It aggregates over axis=1!
void utils::softmax(double* x, double* y, size_t m, size_t n) {
  for(size_t i = 0; i < m; ++i) {
    double* it1 = x + i*n;
    double* it2 = y + i*n;
    double sum_exp = 0;

    // find max to shift x for better numerically stability
    double max;
    for(size_t j = 0; j < n; ++j, it1 += 1) {
      if(j == 0) {
	max = *it1;
      } else {
	if(*it1 > max) {
	  max = *it1;
	}
      }
    }
    it1 = x + i*n;
	
    for(size_t j = 0; j < n; ++j, it1 += 1, it2 += 1) {
      *it2 = std::exp(*it1 - max);
      sum_exp += *it2;
    }
    it2 = y + i*n;
	
    for(size_t j = 0; j < n; ++j, it2 += 1) {
      *it2 /= sum_exp;
    }
  }
}

void utils::submatrix_multiply(CBLAS_TRANSPOSE TransA,
                               CBLAS_TRANSPOSE TransB,
                               double* A, double* B, double* C,
                               int ai, int aj, int bi, int bj,
                               int m, int n, int k, int lda, int ldb,
                               double alpha, double beta) {
    cblas_dgemm(CblasRowMajor, TransA, TransB,
                m, n, k, alpha,
                A + (ai * lda) + aj, lda,
                B + (bi * ldb) + bj, ldb,
                beta, C, n);
}

std::string utils::fullpath(const std::string& dir, const std::string& filename) {
    return dir + "/" + filename;
}

utils::dataset::dataset(const std::string data_path, const uint32_t num_parts,
                        const uint32_t part_num) : data_path(data_path), num_parts(num_parts), part_num(part_num) {
}

utils::matrix_t::matrix_t(double* arr, size_t num_rows, size_t num_cols) : num_rows(num_rows), num_cols(num_cols), arr(arr) {
}

utils::matrix_t::matrix_t(size_t num_rows, size_t num_cols) : num_rows(num_rows), num_cols(num_cols), arr(new double[num_rows * num_cols]) {
  for(uint i = 0; i < num_rows; i++) {
      for(uint j = 0; j < num_cols; j++) {
	arr[i*num_cols + j] = 0;
      }
  }
}

void utils::matrix_t::print() {
  for(uint i = 0; i < num_rows; i++) {
      for(uint j = 0; j < num_cols; j++) {
  	std::cout << arr[i*num_cols + j] << " ";
      }
      std::cout << std::endl;
  }
}

void utils::mat_mul(matrix_t& A, CBLAS_TRANSPOSE TransA, matrix_t& B, CBLAS_TRANSPOSE TransB, matrix_t& C) {
  size_t A_num_rows, A_num_cols, B_num_rows, B_num_cols;
  if (TransA == NoTrans) {
    A_num_rows = A.num_rows;
    A_num_cols = A.num_cols;
  } else {
    A_num_rows = A.num_cols;
    A_num_cols = A.num_rows;
  }
  
  if (TransB == NoTrans) {
    B_num_rows = B.num_rows;
    B_num_cols = B.num_cols;
  } else {
    B_num_rows = B.num_cols;
    B_num_cols = B.num_rows;
  }
  
  assert (A_num_cols == B_num_rows);
  assert (A_num_rows == C.num_rows && B_num_cols == C.num_cols);
  
  utils::submatrix_multiply(TransA, TransB, A.arr, B.arr, C.arr, 0, 0, 0, 0, A_num_rows, B_num_cols, A_num_cols, A.num_cols, B.num_cols, 1, 0);
}

void utils::submat_mul(matrix_t& A, size_t ai, size_t aj, size_t submatA_num_rows, size_t submatA_num_cols, CBLAS_TRANSPOSE TransA,
		       matrix_t& B, size_t bi, size_t bj, size_t submatB_num_rows, size_t submatB_num_cols, CBLAS_TRANSPOSE TransB, matrix_t& C) {
  
  size_t A_num_rows, A_num_cols, B_num_rows, B_num_cols;
  if (TransA == NoTrans) {
    A_num_rows = submatA_num_rows;
    A_num_cols = submatA_num_cols;
  } else {
    A_num_rows = submatA_num_cols;
    A_num_cols = submatA_num_rows;
  }
  
  if (TransB == NoTrans) {
    B_num_rows = submatB_num_rows;
    B_num_cols = submatB_num_cols;
  } else {
    B_num_rows = submatB_num_cols;
    B_num_cols = submatB_num_rows;
  }
  
  assert (A_num_cols == B_num_rows);
  assert (A_num_rows == C.num_rows && B_num_cols == C.num_cols);
  
  utils::submatrix_multiply(TransA, TransB, A.arr, B.arr, C.arr, ai, aj, bi, bj, A_num_rows, B_num_cols, A_num_cols, A.num_cols, B.num_cols, 1, 0);
}


// C := alpha * A + beta * C (This doesn't support transpose.)
void utils::submat_add_assign(double beta, matrix_t& C, size_t ci, size_t cj, size_t submatC_num_rows, size_t submatC_num_cols,
		       double alpha, matrix_t& A, size_t ai, size_t aj, size_t submatA_num_rows, size_t submatA_num_cols) {
  
  assert (submatC_num_rows == submatA_num_rows && submatC_num_cols == submatA_num_cols);
  
  cblas_dgeadd(CblasRowMajor, submatC_num_rows, submatC_num_cols,
	       alpha, A.arr + (ai*submatA_num_cols) + aj, A.num_cols,
	       beta, C.arr + (ci*submatC_num_cols) + cj, C.num_cols);
}

void utils::mat_add_assign(double beta, matrix_t& C, double alpha, matrix_t& A) {
  utils::submat_add_assign(beta, C, 0, 0, C.num_rows, C.num_cols, alpha, A, 0, 0, A.num_rows, A.num_cols);
}

// Store A transpose to A_T
void utils::mat_trans(matrix_t& A, matrix_t& A_T) {
  assert (A.num_rows == A_T.num_cols && A.num_cols == A_T.num_rows);
  for(size_t i = 0; i < A.num_rows; ++i) {
    for(size_t j = 0; j < A.num_cols; ++j) {
      A_T.arr[j*A_T.num_cols + i] = A.arr[i*A.num_cols + j];
    }
  }
}
void utils::submat_trans(matrix_t& A, size_t ai, size_t aj, size_t submatA_num_rows, size_t submatA_num_cols, matrix_t& A_T) {
  assert (submatA_num_rows == A_T.num_cols && submatA_num_cols == A_T.num_rows);
  assert (ai + submatA_num_rows <= A.num_rows && aj + submatA_num_cols <= A.num_cols);
  for(size_t i = 0; i < A_T.num_rows; ++i) {
    for(size_t j = 0; j < A_T.num_cols; ++j) {
      A_T.arr[i*A_T.num_cols + j] = A.arr[(j+ai)*A.num_cols + (i+aj)];
    }
  }
}

double utils::mat_norm(matrix_t& A) {
  double norm = 0.0;
  for (size_t i = 0; i < A.num_rows * A.num_cols; ++i) {
    norm += A.arr[i]*A.arr[i];
  }
  
  return std::sqrt(norm);
}

