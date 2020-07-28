#!/bin/bash

echo "traget directory is set as /home/rdma_for_ml/RDMA_ML/Release/applications/"
nodes=(128.84.139.11 128.84.139.12 128.84.139.14
       128.84.139.26 128.84.139.27 128.84.139.28 128.84.139.20
       128.84.139.21 128.84.139.23 128.84.139.17 128.84.139.18
       128.84.139.25 128.84.139.22 128.84.139.19 128.84.139.24)

scp rdma_for_ml@128.84.139.10:/home/rdma_for_ml/RDMA_ML/Release/applications/server.log . 
scp rdma_for_ml@128.84.139.10:/home/rdma_for_ml/RDMA_ML/Release/applications/RDMAwild* . 

for node in ${nodes[@]}; do
    scp rdma_for_ml@$node:/home/rdma_for_ml/RDMA_ML/Release/applications/worker*.log . 
done

