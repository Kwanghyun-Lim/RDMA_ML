# RDMA_ML
This is a research project of Cornell University led by Kwanghyun Lim, and it is a work in progress.
If you have any question, feel free to contact me. (kwanghyun@cs.cornell.edu)

## How to get optimal loss and optimal model for convergence checking on logistic regression
We used SVRG to get optimal loss and optimal model under a single node with Python code.

## Grid search for async hyper-parameters
### MNIST
Model size: 64KB
Precision: double
Number of workers: 15
Number of epochs: 15
Batch size (total/per worker): 120/8 
Gamma: 1e-4

Grid 1:
alphas: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
decays: [0.9, 0.95, 0.99]

Grid 2:
alphas: [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1]
decays: [0.9, 0.95, 0.99]

result: lr=0.9, decay=0.9

### MNIST RFF
Model size: 800KB
Precision: double
Number of workers: 15
Number of epochs: 50
Batch size (total/per worker): 120/8 
Gamma: 1e-4

Grid 1: 
alphas: [10, 1, 0.1, 0.01, 0.001, 0.0001]
decays: [0.9, 0.95, 0.99, 0.999]

Grid 2: 
alphas: [10, 5, 1, 0.5]
decays: [0.9, 0.95, 0.99, 0.999]

Grid 3: 
alphas: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
decays: [0.9, 0.95, 0.99]

result: lr=9, decay=0.9

## How to run (Emacs Lisp)
### MNIST
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset mnist sync log_reg 0.9 0.9 120 15 16 5" (concat "./worker ~/RDMA_ML/dataset mnist sync log_reg 0.9 0.9 120 15 " (number-to-string (1- x)) " 16 5"))))
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset mnist async log_reg 0.9 0.9 120 15 16 5" (concat "./worker ~/RDMA_ML/dataset mnist async log_reg 0.9 0.9 120 15 " (number-to-string (1- x)) " 16 5"))))
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset mnist fully_async log_reg 0.9 0.9 120 15 16 5" (concat "./worker ~/RDMA_ML/dataset mnist fully_async log_reg 0.9 0.9 120 15 " (number-to-string (1- x)) " 16 5"))))
### MNIST RFF
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset rff sync log_reg 9 0.9 120 50 16 5" (concat "./worker ~/RDMA_ML/dataset rff sync log_reg 9 0.9 120 50 " (number-to-string (1- x)) " 16 5"))))
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset rff async log_reg 9 0.9 120 50 16 5" (concat "./worker ~/RDMA_ML/dataset rff async log_reg 9 0.9 120 50 " (number-to-string (1- x)) " 16 5"))))
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset rff fully_async log_reg 9 0.9 120 50 16 5" (concat "./worker ~/RDMA_ML/dataset rff fully_async log_reg 9 0.9 120 50 " (number-to-string (1- x)) " 16 5"))))

### MNIST dnn
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset mnist sync dnn 0.1 0.9 120 50 16 3" (concat "./worker ~/RDMA_ML/dataset mnist sync dnn 0.1 0.9 120 50 " (number-to-string (1- x)) " 16 3"))))
