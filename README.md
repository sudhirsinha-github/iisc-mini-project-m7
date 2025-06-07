# iisc-mini-project-m7
iisc-mini-project-m7 IISC+AIMLOPS_C4+MLS_C4/block-v1:IISC+AIMLOPS_C4+MLS_C4+typ

**Mini-Project: MPI-Based Distributed Matrix Multiplication
**
**Project Title: Distributed Matrix Multiplication using MPI
**

Objective: To implement and evaluate the performance of matrix multiplication across multiple nodes using MPI.


An MPI-based distributed matrix multiplication program is presented below, along with instructions for setup, compilation, and execution. This project demonstrates how to parallelize matrix multiplication using the Message Passing Interface (MPI) to achieve significant performance gains over a serial implementation.

Tasks:
Environment Setup: Set up an MPI development environment.

Matrix Multiplication: Implement a standard matrix multiplication algorithm.

Distributed Implementation: Modify the algorithm for distributed computation using
MPI, focusing on data partitioning and inter-process communication.

Performance Metrics: Develop a system to measure execution time and scalability.

Scalability Testing: Test the algorithm on different numbers of nodes/processes.

Benchmarking: Benchmark against a serial implementation to evaluate performance
gains.

Deliverables: MPI-based distributed matrix multiplication code, performance metrics,
benchmarking report, and detailed documentation.



Setup for MAC
> brew install open-mpi


python3 -m venv mpi-gpu-env
source mpi-gpu-env/bin/activate


pip install mpi4py numpy
pip install torch torchvision torchaudio

pip install -r requirements.txt


## Documentation: MPI-based Distributed Matrix Multiplication

This document details the provided MPI-based distributed matrix multiplication code, focusing on its structure, implementation, and performance evaluation.

### 1. Introduction

This code implements a distributed matrix multiplication algorithm using MPI and PyTorch for GPU acceleration. It compares the performance of serial CPU, single-GPU, and distributed GPU implementations.

### 2. Deliverables

* **MPI-based Distributed Matrix Multiplication Code:** `distributed_matmul_gpu.py`
* **Performance Metrics:** Execution time for serial CPU, single-GPU, and distributed GPU. Speedup of GPU vs. CPU.
* **Benchmarking Report:** Printed to the console, includes execution times and speedup.

### 3. Design and Algorithm

* **Algorithm:** The code distributes the rows of the first matrix (A) across MPI processes and broadcasts the second matrix (B) to all processes. Each process performs a local matrix multiplication using the GPU and gathers the results on the master process.
* **Distributed Approach:** Rows of matrix A are scattered, matrix B is broadcasted, local matrix multiplication is performed on each process, and results are gathered.
* **Communication Patterns:** `MPI_Init`, `MPI_Comm_rank`, `MPI_Comm_size`, `MPI_Bcast`, `MPI_Scatter`, `MPI_Gather`, `MPI_Barrier`.
            Effective communication between processes is fundamental to achieving parallelism in distributed memory systems. The provided distributed_matmul_final.py script leverages several key MPI (Message Passing Interface) functions to coordinate data distribution, computation, and result collection. Below is a detailed explanation of each function used and its role in the distributed matrix multiplication process:

            MPI_Init() (Implicitly handled by mpi4py in main()'s entry):

            Purpose: This function (or its equivalent in mpi4py, which is automatically handled when you import mpi4py and create MPI.COMM_WORLD) initializes the MPI execution environment. It must be called before any other MPI function.

            Role in Project: Sets up the necessary infrastructure for processes to communicate and participate in the distributed computation. Without it, the MPI functions would not be available.

            comm.Get_rank():

            Purpose: Returns the rank (or ID) of the calling process within its communicator (comm). Ranks typically range from 0 to P−1, where P is the total number of processes. Process with rank 0 is traditionally designated as the "master" or "root" process, responsible for initial data distribution and final result collection.

            Role in Project: Allows each process to know its unique identity, enabling conditional execution (e.g., only rank 0 initializes matrices) and directing data flow (e.g., Scatter sends data from rank 0, Gather collects data to rank 0).

            comm.Get_size():

            Purpose: Returns the total number of processes in the specified communicator (comm).

            Role in Project: Used to determine how to divide the workload (e.g., N // size calculates how many rows each process will handle) and for overall coordination of the distributed algorithm.

            comm.Bcast(B, root=0):

            Purpose: Broadcasts data from one process (the root) to all other processes in the communicator. The root process sends the data, and all other processes receive it.

            Role in Project: In matrix multiplication C=A×B, the entire matrix B is needed by every process to compute its portion of C. The Bcast operation efficiently distributes matrix B from the master process (rank 0) to all other participating processes, ensuring they have the necessary data for their local multiplications.

            comm.Scatter(A, local_A, root=0):

            Purpose: Distributes elements from an array on the root process to all processes in the communicator. Each process receives a portion of the original array.

            Role in Project: Matrix A is distributed row-wise. The Scatter operation takes the full matrix A from the master process (rank 0) and distributes contiguous blocks of its rows (local_A) to each process. For example, if matrix A has N rows and there are size processes, each process will receive N / size rows of A.

            comm.Gather(local_C, C_dist, root=0):

            Purpose: Gathers results from all processes in the communicator to the root process. Each process sends its portion of data to the root, which then concatenates them into a single array.

            Role in Project: After each process computes its local_C (a portion of the resulting matrix C), the Gather operation collects these partial results from all processes and assembles them into the complete matrix C_dist on the master process (rank 0).

            comm.Barrier():

            Purpose: A synchronization point for all processes in the communicator. No process can proceed past the barrier until all processes in the communicator have reached it.

            Role in Project: Used before and after the distributed computation phase (dist_start_time and dist_duration measurements) to ensure accurate timing. By placing barriers, we ensure that all processes have completed the data distribution and computation before the timer is stopped, and that all processes are ready before it begins, providing a more reliable measure of the parallel execution time.

### 4. Implementation Details

* **Language:** Python
* **Libraries:** `numpy`, `torch`, `mpi4py`
* **Code Structure:**
    * `matrix_multiply_gpu(local_A_np, B_np, device)`: Performs matrix multiplication on the GPU using PyTorch.
    * `serial_matrix_multiply_cpu(A, B)`: Implements standard matrix multiplication on the CPU.
    * `main()`:  Orchestrates the distributed computation, benchmarking, and reporting.

* **Compilation and Execution Instructions:**
    * **Prerequisites:** MPI, PyTorch with GPU support.
    * **Execution:** `python -m mpi4py.run distributed_matmul_gpu.py <matrix_size>` (matrix\_size is optional).

### 5. Benchmarking and Performance Analysis

* **Test Environment:** Mac (GPU or CPU), MPI processes.
* **Benchmarking Methodology:** Measures execution time for serial CPU, single-GPU, and distributed GPU.
* **Metrics Collected:** Execution time, GPU vs. CPU speedup.
* **Results:** Printed to the console, including execution times and speedup.

### 6. Conclusion

The code demonstrates a distributed matrix multiplication implementation leveraging GPU acceleration. It provides a basic performance comparison between CPU, single-GPU, and distributed GPU executions.

![Snapshot Response ](https://github.com/sudhirsinha-github/iisc-mini-project-m7/blob/main/Screenshot%202025-06-07%20at%2011.53.02%E2%80%AFPM.png)


#####  Output from local system (MAC m4)

(mpi-gpu-env) in22909502@INMLMLJWKHC0 app % mpirun -np 1 python distributed_matmul_gpu.py 2048

Project: Distributed Matrix Multiplication (2048x2048)
Hardware: Mac (MPS) | MPI Processes: 1
------------------------------------------------------------

--- PERFORMANCE & BENCHMARKING REPORT ---
  - Serial CPU Time (1 Core):      0.011746 seconds
  - Non-Distributed GPU Time:      0.008588 seconds
  - Distributed GPU Time (1 procs):  0.122156 seconds
------------------------------------------------------------
✅ GPU vs. CPU Speedup: 1.37x
   (This shows the raw acceleration from using the GPU over a single CPU core)
   
✅ Result verification successful.
(mpi-gpu-env) in22909502@INMLMLJWKHC0 app % mpirun -np 2 python distributed_matmul_gpu.py 2048

Project: Distributed Matrix Multiplication (2048x2048)
Hardware: Mac (MPS) | MPI Processes: 2
------------------------------------------------------------

--- PERFORMANCE & BENCHMARKING REPORT ---
  - Serial CPU Time (1 Core):      0.009227 seconds
  - Non-Distributed GPU Time:      0.110056 seconds
  - Distributed GPU Time (2 procs):  0.050108 seconds
------------------------------------------------------------
✅ GPU vs. CPU Speedup: 0.08x
   (This shows the raw acceleration from using the GPU over a single CPU core)
✅ Result verification successful.
(mpi-gpu-env) in22909502@INMLMLJWKHC0 app % mpirun -np 4 python distributed_matmul_gpu.py 2048

Project: Distributed Matrix Multiplication (2048x2048)
Hardware: Mac (MPS) | MPI Processes: 4
------------------------------------------------------------

^C%                                                                                                                                                                   
(mpi-gpu-env) in22909502@INMLMLJWKHC0 app % mpirun -np 4 python distributed_matmul_gpu.py 2048

Project: Distributed Matrix Multiplication (2048x2048)
Hardware: Mac (MPS) | MPI Processes: 4
------------------------------------------------------------

--- PERFORMANCE & BENCHMARKING REPORT ---
  - Serial CPU Time (1 Core):      0.008609 seconds
  - Non-Distributed GPU Time:      0.184947 seconds
  - Distributed GPU Time (4 procs):  0.043474 seconds
------------------------------------------------------------

✅ GPU vs. CPU Speedup: 0.05x
   (This shows the raw acceleration from using the GPU over a single CPU core)

✅ Result verification successful.
(mpi-gpu-env) in22909502@INMLMLJWKHC0 app % 



--------
#### Unit test
> python -m unittest distributed_matmul_gpu_test.py
