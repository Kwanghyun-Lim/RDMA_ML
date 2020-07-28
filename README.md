# RDMA_ML
## How to run
* Emacs Lisp
(run (lambda (x) (if (= x 1) "./server ~/RDMA_ML/dataset async mnist 0.9 0.9 120 15 16 5 1" (concat "./worker ~/RDMA_ML/dataset async mnist 0.9 0.9 120 15 " (number-to-string (1- x)) " 16 5 1"))))
