#pragma once
#include <cstdlib>

namespace ml_model {
// interface for ML models
class ml_model { 
public:
  virtual size_t get_model_size() const;
  virtual void set_model_mem(double* model);
  virtual void push_back_to_grad_vec(double* gradient);
  virtual size_t get_num_batches() const;
  virtual void update_model(uint ml_sst_row);
  virtual double* get_model() const;
  virtual void compute_gradient(const size_t batch_num, double* given_model);

  virtual void train(const size_t num_epochs);
  virtual double training_error();
  virtual double training_loss();
  virtual double get_loss_opt() const;
  virtual double test_error();
  virtual double gradient_norm();
  virtual double distance_to_optimum();
};
} // namespace ml_model
