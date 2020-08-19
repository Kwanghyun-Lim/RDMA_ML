#pragma once

#include <assert.h>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <functional>
#include <memory>
#include <string>

#define NoTrans CblasNoTrans
#define Trans CblasTrans

namespace utils {

template <typename T>
struct atomwrapper
{
  std::atomic<T> _a;
  atomwrapper() :_a() {}
  atomwrapper(const std::atomic<T> &a) : _a(a.load()) {}
  atomwrapper(const atomwrapper &other) :_a(other._a.load()) {}
  atomwrapper &operator=(const atomwrapper &other) {
    _a.store(other._a.load());
  }
  atomwrapper &operator=(T val) {
    _a.store(val);
  }
  T get_val() {
    return _a.load();
  }
};

class matrix_t {
public:
  double* arr;
  size_t num_rows;
  size_t num_cols;
  matrix_t(double* arr, size_t num_rows, size_t num_cols);
  matrix_t(size_t num_rows, size_t num_cols);
  void print();
};
  
class images_t {
public:
    std::unique_ptr<double[]> arr;
    size_t num_part_images;
    size_t num_total_images;
    size_t num_pixels;
    images_t() = default;
    images_t(images_t&&) = default;
    images_t(const images_t&) = delete;
    images_t& operator=(images_t&& images) = default;
    images_t& operator=(const images_t& images) = delete;
};

class labels_t {
public:
    std::unique_ptr<double[]> arr;
    size_t num_part_labels;
    size_t num_total_labels;
    size_t num_classes;
    labels_t() = default;
    labels_t(labels_t&&) = default;
    labels_t(const labels_t&) = delete;
    labels_t& operator=(labels_t&& labels) = default;
    labels_t& operator=(const labels_t& labels) = delete;
};

class dataset {
public:
    dataset(const std::string data_path, const uint32_t num_parts, const uint32_t part_num);
  
    images_t training_images;
    images_t test_images;
    labels_t training_labels;
    labels_t test_labels;

    const std::string data_path;
    const uint32_t num_parts;
    const uint32_t part_num;
};

typedef std::function<utils::dataset()> reader_t;

void zero_arr(double* arr, const size_t size);
void softmax(double* x, const size_t m, const size_t n);
// numerically stable softmax
void softmax(double* x, double* y, size_t m, size_t n);
// assumes that C has the right dimension and n as lda
void submatrix_multiply(CBLAS_TRANSPOSE TransA,
                        CBLAS_TRANSPOSE TransB,
                        double* A, double* B, double* C,
                        int ai, int aj, int bi, int bj,
                        int m, int n, int k, int lda, int ldb,
			double alpha, double beta);

std::string fullpath(const std::string& dir, const std::string& filename);

void mat_mul(matrix_t& A, CBLAS_TRANSPOSE TransA, matrix_t& B, CBLAS_TRANSPOSE TransB, matrix_t& C);
void submat_mul(
   matrix_t& A, size_t ai, size_t aj, size_t submatA_num_rows, size_t submatA_num_cols, CBLAS_TRANSPOSE TransA,
   matrix_t& B, size_t bi, size_t bj, size_t submatB_num_rows, size_t submatB_num_cols, CBLAS_TRANSPOSE TransB, matrix_t& C);
void mat_add_assign(double beta, matrix_t& C, double alpha, matrix_t& A);
void submat_add_assign(double beta, matrix_t& C, size_t ci, size_t cj, size_t submatC_num_rows, size_t submatC_num_cols,
		       double alpha, matrix_t& A, size_t ai, size_t aj, size_t submatA_num_rows, size_t submatA_num_cols);
void mat_trans(matrix_t& A, matrix_t& A_T);
void submat_trans(matrix_t& A, size_t ai, size_t aj, size_t submatA_num_rows, size_t submatA_num_cols, matrix_t& A_T);
double mat_norm(matrix_t& A);
}  // namespace utils
