#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

#include "utils/numpy_reader.hpp"
#include "log_reg.hpp"
#include "dnn.hpp"

int main(int argc, char* argv[]) {
    if(argc < 7) {
      std::cerr << "Usage: " << argv[0]
		<< " <data_directory> <sync/mnist/rff> \
<log_reg/dnn> <alpha> <decay> <batch_size> <num_epochs>" << std::endl;
        return 1;
    }

    // Initialize parameters
    std::string data_directory(argv[1]);
    std::string data(argv[2]);
    std::string ml_model_name(argv[3]);
    const double alpha = std::stod(argv[4]);
    const double gamma = 0.0001;
    double decay = std::stod(argv[5]);
    const uint32_t batch_size = atoi(argv[6]);
    const uint32_t num_epochs = atoi(argv[7]);
    
    ml_model::ml_model* ml_model;
    if (ml_model_name == "log_reg") {
        ml_model = new ml_model::multinomial_log_reg(
     	     [&]() {return (utils::dataset)numpy::numpy_dataset(
		                       data_directory + "/" + data,
		                       1, 0);},
                                       alpha, gamma, decay, batch_size);
    } else if (ml_model_name == "dnn") {
        const std::vector<uint32_t> layer_size_vec {784, 50, 10};
        const uint32_t num_layers = 3;
        ml_model = new ml_model::deep_neural_network(
	    layer_size_vec, num_layers,	     
            [&]() {return (utils::dataset)numpy::numpy_dataset(
		                       data_directory + "/" + data,
		                       1, 0);},
                                       alpha, batch_size, true);
    } else {
      	std::cerr << "Wrong ml_model_name input: " << ml_model_name << std::endl;
	exit(1);
    }

    double* model = new double[ml_model->get_model_size()]; // ml_sst first row
    if (ml_model_name == "log_reg") {
      utils::zero_arr(model, ml_model->get_model_size());
    } else if (ml_model_name == "dnn") {
      std::string model_full_path = data_directory + "/" + data + "/model_784-50-10.npy";
      ml_model->init_model(model, model_full_path);
    } else {
      	std::cerr << "Wrong ml_model_name input: " << ml_model_name << std::endl;
	exit(1);
    }
    
    double* gradient = new double[ml_model->get_model_size()]; // ml_sst my own row
    utils::zero_arr(gradient, ml_model->get_model_size());
    
    ml_model->set_model_mem(model);
    ml_model->push_back_to_grad_vec(gradient);
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_REALTIME, &start_time);
    ml_model->train(num_epochs);
    clock_gettime(CLOCK_REALTIME, &end_time);
    const double time_taken = (double)(end_time.tv_sec - start_time.tv_sec)
                         + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    const double training_error = ml_model->training_error();
    const double test_error = ml_model->test_error();
    const double loss = ml_model->training_loss();
    
    std::cout << "Number of epochs: " << num_epochs << std::endl;
    std::cout << "Time taken to train: "
    	      << std::fixed << std::setprecision(2)
              << time_taken << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(5)
              << "Training error: " << training_error << std::endl;
    std::cout << std::fixed << std::setprecision(5)
    	      << "Test error: " << test_error << std::endl;
    std::cout << std::fixed << std::setprecision(5)
    	      << "Loss: " << loss << std::endl;

    std::ofstream fout("dnn", std::ofstream::app);
    fout << num_epochs << " " << time_taken << " "
         << 100 * (1 - training_error) << " " 
    	 << 100 * (1 - test_error) << " "
    	 << loss << std::endl;
}
