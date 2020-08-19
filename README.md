# RDMA_ML
## How to get optimal loss and optimal model for convergence checking on logistic regression
We used SVRG to get optimal loss and optimal model under a single node with Python code.
The hyper-parameter setup with SVRG are below.
### MNIST
epochs= , batch size= , lr= , decay= , regularizer= ,

### MNIST RFF
epochs= , batch size= , lr= , decay= , regularizer= ,

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
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset mnist sync 0.9 0.9 120 15 16 5" (concat "./worker ~/RDMA_ML/dataset mnist sync 0.9 0.9 120 15 " (number-to-string (1- x)) " 16 5"))))
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset mnist async 0.9 0.9 120 15 16 5" (concat "./worker ~/RDMA_ML/dataset mnist async 0.9 0.9 120 15 " (number-to-string (1- x)) " 16 5"))))
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset mnist fully_async 0.9 0.9 120 15 16 5" (concat "./worker ~/RDMA_ML/dataset mnist fully_async 0.9 0.9 120 15 " (number-to-string (1- x)) " 16 5"))))
### MNIST RFF
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset rff sync 9 0.9 120 50 16 5" (concat "./worker ~/RDMA_ML/dataset rff sync 9 0.9 120 50 " (number-to-string (1- x)) " 16 5"))))
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset rff async 9 0.9 120 50 16 5" (concat "./worker ~/RDMA_ML/dataset rff async 9 0.9 120 50 " (number-to-string (1- x)) " 16 5"))))
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset rff fully_async 9 0.9 120 50 16 5" (concat "./worker ~/RDMA_ML/dataset rff fully_async 9 0.9 120 50 " (number-to-string (1- x)) " 16 5"))))
