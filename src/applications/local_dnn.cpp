#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

#include "utils/numpy_reader.hpp"
#include "dnn.hpp"

int main(int argc, char* argv[]) {
    if(argc < 6) {
      std::cerr << "Usage: " << argv[0]
		<< " <data_directory> <sync/mnist/rff> \
                     <alpha> <batch_size> <num_epochs>" << std::endl;
        return 1;
    }

    // Initialize parameters
    std::string data_directory(argv[1]);
    std::string data(argv[2]);
    const double alpha = std::stod(argv[3]);
    const uint32_t batch_size = atoi(argv[4]);
    const uint32_t num_epochs = atoi(argv[5]);
    const std::vector<uint32_t> layer_size_vec {784, 50, 10};
    const uint32_t num_layers = 3;
    
    DNN::deep_neural_network dnn(
	    layer_size_vec, num_layers,	     
            [&]() {
	      return (utils::dataset)numpy::numpy_dataset(
		                       data_directory + "/" + data,
		                       1, 0);
            },
            alpha, batch_size, true);

    
    double* model = new double[dnn.get_model_size()]; // ml_sst first row
    double* gradient = new double[dnn.get_model_size()]; // ml_sst my own row
    // utils::zero_arr(model, dnn.get_model_size());
    utils::zero_arr(gradient, dnn.get_model_size());
    std::string model_full_path
      = data_directory + "/" + data + "/model_784-50-10.npy";
    dnn.init_model(model, model_full_path);
    // for (size_t i = 0; i < dnn.get_model_size(); i++) {
    //   if(model[i] != 0)
    // 	std::cout << model[i] << " ";
    // }
    
    std::cout << "W0 =" << std::endl;
    // for (size_t i = 0; i < dnn.get_model_size(); i++) {
    //   if(model[i] != 0)
    // 	std::cout << model[i] << " ";
    // }
    
    std::cout << "[main] gradient " << gradient << std::endl;
    dnn.set_model_mem(model);
    dnn.set_gradient_mem(gradient);
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_REALTIME, &start_time);
    dnn.train(num_epochs);
    std::cout << "[main] train done " << std::endl;
    clock_gettime(CLOCK_REALTIME, &end_time);
    const double time_taken = (double)(end_time.tv_sec - start_time.tv_sec)
                         + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    const double training_error = dnn.training_error();
    const double test_error = dnn.test_error();
    const double loss = dnn.training_loss();
    
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
