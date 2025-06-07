# distributed_matmul_final.py
import numpy as np
import torch
from mpi4py import MPI
import time
import sys
import os

# --- IMPORTANT: Force NumPy to be single-threaded for a true serial benchmark ---
# This ensures our CPU benchmark is a fair, single-core comparison.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


def matrix_multiply_gpu(local_A_np, B_np, device):
    """Performs matrix multiplication on the GPU using PyTorch."""
    # Move data to the GPU
    local_A = torch.from_numpy(local_A_np).to(device)
    B = torch.from_numpy(B_np).to(device)
    
    # Perform the core computation
    local_C = torch.matmul(local_A, B)
    
    # Move result back to CPU for MPI communication
    return local_C.cpu().numpy()

def serial_matrix_multiply_cpu(A, B):
    """(Task 2) Implements a standard matrix multiplication algorithm on the CPU."""
    return np.dot(A, B)

def main():
    """Main function to run the distributed computation and benchmarking."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine matrix size from command-line arguments, default to 1024
    try:
        N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    except (ValueError, IndexError):
        N = 1024

    if N % size != 0:
        if rank == 0:
            print("Error: Matrix size N must be divisible by the number of MPI processes.", file=sys.stderr)
        MPI.Finalize()
        sys.exit(1)

    # --- GPU Device Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # The master process (rank 0) initializes data
    if rank == 0:
        print(f"Project: Distributed Matrix Multiplication ({N}x{N})")
        print(f"Hardware: Mac ({device.type.upper()}) | MPI Processes: {size}")
        print("-" * 60)
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
    else:
        A, B = None, np.empty((N, N), dtype=np.float32)

    # --- (Task 3) Distributed Implementation ---
    comm.Barrier() # Synchronize processes before starting timer
    dist_start_time = MPI.Wtime()
    
    # 1. Broadcast the entire matrix B to all processes
    comm.Bcast(B, root=0)
    
    # 2. Scatter rows of matrix A to all processes
    rows_per_process = N // size
    local_A = np.empty((rows_per_process, N), dtype=np.float32)
    comm.Scatter(A, local_A, root=0)
    
    # 3. Each process performs its multiplication on its local data
    local_C = matrix_multiply_gpu(local_A, B, device)
    
    # 4. Gather results back to the master process
    C_dist = None
    if rank == 0:
        C_dist = np.empty((N, N), dtype=np.float32)
    
    comm.Gather(local_C, C_dist, root=0)
    
    comm.Barrier() # Synchronize processes before stopping timer
    dist_duration = MPI.Wtime() - dist_start_time

    # --- (Tasks 4, 6) Performance Metrics & Benchmarking ---
    # The master process calculates final metrics and prints the report
    if rank == 0:
        print("\n--- PERFORMANCE & BENCHMARKING REPORT ---")

        # Benchmark 1: True Serial CPU
        cpu_start_time = time.perf_counter()
        serial_C_cpu = serial_matrix_multiply_cpu(A, B)
        cpu_duration = time.perf_counter() - cpu_start_time
        print(f"  - Serial CPU Time (1 Core):      {cpu_duration:.6f} seconds")

        # Benchmark 2: Single-Process GPU (No MPI)
        gpu_start_time = time.perf_counter()
        C_gpu_single = matrix_multiply_gpu(A, B, device)
        gpu_duration = time.perf_counter() - gpu_start_time
        print(f"  - Non-Distributed GPU Time:      {gpu_duration:.6f} seconds")

        # Benchmark 3: Distributed GPU (MPI)
        print(f"  - Distributed GPU Time ({size} procs):  {dist_duration:.6f} seconds")
        print("-" * 60)

        # Performance Gains Analysis
        gpu_vs_cpu_speedup = cpu_duration / gpu_duration
        print(f"✅ GPU vs. CPU Speedup: {gpu_vs_cpu_speedup:.2f}x")
        print("   (This shows the raw acceleration from using the GPU over a single CPU core)")
        
        # Verification of correctness
        assert np.allclose(serial_C_cpu, C_dist, atol=1e-4)
        print("✅ Result verification successful.")

if __name__ == "__main__":
    main()