import unittest
import numpy as np
import torch
import os
# Import the functions to be tested from your main script
# Assuming distributed_matmul_final.py is in the same directory or accessible via path
# If running this as a separate file, you might need to adjust the import
try:
    from distributed_matmul_final import matrix_multiply_gpu, serial_matrix_multiply_cpu
except ImportError:
    print("Could not import functions from distributed_matmul_final.py. "
          "Ensure the file is in the same directory or adjust the import path.")
    # Define dummy functions to prevent errors during syntax check if import fails
    def matrix_multiply_gpu(local_A_np, B_np, device):
        raise NotImplementedError("matrix_multiply_gpu not imported for testing")
    def serial_matrix_multiply_cpu(A, B):
        raise NotImplementedError("serial_matrix_multiply_cpu not imported for testing")


# It's good practice to ensure these are also single-threaded for consistent CPU testing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


class TestMatrixMultiplication(unittest.TestCase):
    """
    Unit tests for the matrix multiplication functions.
    """

    def setUp(self):
        """
        Set up common test data for all test methods.
        We'll use small, deterministic matrices for easy verification.
        """
        self.A_small = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self.B_small = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        # Expected result for A_small @ B_small
        # C[0,0] = (1*5) + (2*7) = 5 + 14 = 19
        # C[0,1] = (1*6) + (2*8) = 6 + 16 = 22
        # C[1,0] = (3*5) + (4*7) = 15 + 28 = 43
        # C[1,1] = (3*6) + (4*8) = 18 + 32 = 50
        self.expected_C_small = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)

        # Larger matrices for more robust testing, though still small enough for unit tests
        self.A_large = np.random.rand(64, 128).astype(np.float32)
        self.B_large = np.random.rand(128, 96).astype(np.float32)
        self.expected_C_large = np.dot(self.A_large, self.B_large)

        # Determine the device for GPU tests
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"\nRunning GPU tests on: {self.device.type.upper()}")


    def test_serial_matrix_multiply_cpu_small(self):
        """
        Test serial_matrix_multiply_cpu with small, known inputs.
        """
        print("Running test_serial_matrix_multiply_cpu_small...")
        result_C = serial_matrix_multiply_cpu(self.A_small, self.B_small)
        np.testing.assert_allclose(result_C, self.expected_C_small, rtol=1e-5, atol=1e-5)
        print("  test_serial_matrix_multiply_cpu_small PASSED.")

    def test_serial_matrix_multiply_cpu_large(self):
        """
        Test serial_matrix_multiply_cpu with larger, random inputs.
        """
        print("Running test_serial_matrix_multiply_cpu_large...")
        result_C = serial_matrix_multiply_cpu(self.A_large, self.B_large)
        np.testing.assert_allclose(result_C, self.expected_C_large, rtol=1e-5, atol=1e-5)
        print("  test_serial_matrix_multiply_cpu_large PASSED.")


    def test_matrix_multiply_gpu_small(self):
        """
        Test matrix_multiply_gpu with small, known inputs.
        Ensures the GPU function produces correct results,
        even if the actual device is CPU.
        """
        print(f"Running test_matrix_multiply_gpu_small on {self.device.type.upper()}...")
        result_C = matrix_multiply_gpu(self.A_small, self.B_small, self.device)
        np.testing.assert_allclose(result_C, self.expected_C_small, rtol=1e-5, atol=1e-5)
        print("  test_matrix_multiply_gpu_small PASSED.")


    def test_matrix_multiply_gpu_large(self):
        """
        Test matrix_multiply_gpu with larger, random inputs.
        Ensures the GPU function produces correct results,
        even if the actual device is CPU.
        """
        print(f"Running test_matrix_multiply_gpu_large on {self.device.type.upper()}...")
        result_C = matrix_multiply_gpu(self.A_large, self.B_large, self.device)
        np.testing.assert_allclose(result_C, self.expected_C_large, rtol=1e-5, atol=1e-5)
        print("  test_matrix_multiply_gpu_large PASSED.")


# This allows running the tests directly from the command line
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

