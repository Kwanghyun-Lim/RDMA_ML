#pragma once

#include <memory>
#include <string>

#include "utils/utils.hpp"
#include "utils/cnpy.hpp"

namespace log_reg {
class multinomial_log_reg {
public:
    multinomial_log_reg(const utils::reader_t& dataset_loader,
			const double alpha, const double gamma, double decay, const size_t batch_size);

    void train(const size_t num_epochs);
 
    double training_error();
    double training_loss();
    double get_loss_opt() const;
    double test_error();
    double gradient_norm();
    double distance_to_optimum();
  
    void compute_gradient(const size_t batch_num, double* given_model);
    void compute_full_gradient(double* given_model);
    void update_model(uint ml_sst_row);
    void update_model();
    double decay_alpha();
    void update_gradient(const size_t batch_num);
    void copy_model(double* src, double* dst, size_t len);

    void set_model_mem(double* model);
    void initialize_model_mem_with_zero();
    void push_back_to_grad_vec(double* gradient);

    double* get_model() const;
    double* get_gradient(uint ml_sst_row) const;
    double* get_anchor_model() const;
    double* get_full_gradient() const;
    double* get_sample_gradient() const;
  
    size_t get_model_size() const;

    size_t get_num_batches() const;
    size_t get_num_part_images() const;
    size_t get_num_total_images() const;

    void save_npy_model() const;
private:
    double compute_error(const utils::images_t& images, const utils::labels_t& labels);
    double compute_loss(const utils::images_t& images, const utils::labels_t& labels);
  
    utils::dataset dataset;
    const size_t model_size;
    double* model;
    std::vector<double*> gradients;
    double* full_gradient; // for gradient norm statistics
    std::unique_ptr<double[]> full_predicted_labels; // for full_gradient
  
    double alpha;
    const double gamma;
    double decay;
    const size_t batch_size;
    const size_t aggregate_batch_size;
    uint64_t num_model_updates;

    // temporary space for predicted labels
    std::unique_ptr<double[]> predicted_labels;
};
}  // namespace log_reg
