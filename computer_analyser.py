import torch
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import numpy as np


class ComputeAnalyser():
    def __init__(
        self,
        matrix_sizes: List[Tuple[int, int]] = [(128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)],
        bit_depths: List[int] = [2, 4, 8, 16],
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        results_file: str = "compute_benchmark_results.json"
    ):
        self.matrix_sizes = matrix_sizes
        self.bit_depths = bit_depths
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        self.results_file = Path(results_file)
        
        # Check CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, benchmarks will run on CPU")
        
        # Get GPU information
        self.gpu_info = self._get_gpu_info()
        
        # Enable TF32 for better performance on Ampere+ GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _get_gpu_info(self) -> Dict[str, str]:
        """Get GPU device information"""
        if torch.cuda.is_available():
            return {
                'name': torch.cuda.get_device_name(0),
                'capability': f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
                'total_memory': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
                'cuda_version': torch.version.cuda
            }
        return {'name': 'CPU', 'capability': 'N/A', 'total_memory': 'N/A', 'cuda_version': 'N/A'}
    
    def _get_dtype(self, bit_depth: int) -> torch.dtype:
        """Map bit depth to PyTorch dtype with tensor core optimization"""
        dtype_map = {
            2: torch.float8_e5m2,  # FP8 E5M2 (if available)
            4: torch.float16,       # FP16 (half precision)
            8: torch.bfloat16,      # BF16 (brain float)
            16: torch.float16,      # FP16
            32: torch.float32       # FP32
        }
        
        # Fallback for unsupported dtypes
        if bit_depth == 2:
            # Check if FP8 is available (requires PyTorch 2.1+ and compatible GPU)
            try:
                if hasattr(torch, 'float8_e5m2'):
                    return torch.float8_e5m2
            except:
                pass
            print(f"FP8 not available, using FP16 instead for {bit_depth}-bit")
            return torch.float16
        
        return dtype_map.get(bit_depth, torch.float32)
    
    def _benchmark_matmul(self, size: Tuple[int, int], dtype: torch.dtype) -> Dict[str, float]:
        """Benchmark matrix multiplication for given size and dtype"""
        M, N = size
        K = N  # Square matrices for simplicity
        
        # Create random matrices
        try:
            A = torch.randn(M, K, dtype=dtype, device=self.device)
            B = torch.randn(K, N, dtype=dtype, device=self.device)
        except Exception as e:
            print(f"Error creating tensors with dtype {dtype}: {e}")
            return {'time_ms': float('inf'), 'gflops': 0.0}
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = torch.matmul(A, B)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(self.num_iterations):
            C = torch.matmul(A, B)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        elapsed_time = (end_time - start_time) / self.num_iterations
        elapsed_ms = elapsed_time * 1000
        
        # FLOPS calculation: 2*M*N*K operations for matrix multiplication
        flops = 2 * M * N * K
        gflops = (flops / elapsed_time) / 1e9
        
        return {
            'time_ms': elapsed_ms,
            'gflops': gflops
        }
    
    def analyse(self, model: Optional[torch.nn.Module] = None) -> Dict:
        """
        Run compute analysis across matrix sizes and bit depths
        
        Args:
            model: Optional PyTorch model to analyze (future enhancement)
        
        Returns:
            Dictionary containing benchmark results
        """
        print(f"Running benchmarks on: {self.gpu_info['name']}")
        print(f"CUDA Capability: {self.gpu_info['capability']}")
        print("-" * 80)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'gpu_info': self.gpu_info,
            'benchmarks': {}
        }
        
        for size in self.matrix_sizes:
            size_key = f"{size[0]}x{size[1]}"
            results['benchmarks'][size_key] = {}
            
            print(f"\nMatrix Size: {size_key}")
            
            for bit_depth in self.bit_depths:
                dtype = self._get_dtype(bit_depth)
                dtype_name = str(dtype).split('.')[-1]
                
                try:
                    metrics = self._benchmark_matmul(size, dtype)
                    
                    results['benchmarks'][size_key][f"{bit_depth}bit"] = {
                        'dtype': dtype_name,
                        'time_ms': metrics['time_ms'],
                        'gflops': metrics['gflops']
                    }
                    
                    print(f"  {bit_depth}-bit ({dtype_name}): {metrics['time_ms']:.4f} ms, {metrics['gflops']:.2f} GFLOPS")
                    
                except Exception as e:
                    print(f"  {bit_depth}-bit: Error - {e}")
                    results['benchmarks'][size_key][f"{bit_depth}bit"] = {
                        'dtype': str(dtype),
                        'error': str(e)
                    }
        
        # Calculate cost ratios (relative to FP16)
        self._calculate_cost_ratios(results)        
        self._save_results(results)
        
        return results
    
    def _calculate_cost_ratios(self, results: Dict):
        """Calculate relative cost between bit depths"""
        print("\n" + "=" * 80)
        print("Cost Analysis (relative to 16-bit FP16):")
        print("=" * 80)
        
        for size_key, size_results in results['benchmarks'].items():
            if '16bit' not in size_results or 'time_ms' not in size_results['16bit']:
                continue
            
            baseline_time = size_results['16bit']['time_ms']
            
            print(f"\nMatrix Size: {size_key}")
            for bit_key, bit_results in size_results.items():
                if 'time_ms' in bit_results:
                    ratio = bit_results['time_ms'] / baseline_time
                    speedup = baseline_time / bit_results['time_ms']
                    bit_results['cost_ratio'] = ratio
                    bit_results['speedup'] = speedup
                    print(f"  {bit_key}: {speedup:.2f}x speedup (cost ratio: {ratio:.3f})")
    
    def _save_results(self, results: Dict):
        existing_results = []
        
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                try:
                    existing_results = json.load(f)
                    if not isinstance(existing_results, list):
                        existing_results = [existing_results]
                except json.JSONDecodeError:
                    existing_results = []
        
        existing_results.append(results)
        
        with open(self.results_file, 'w') as f:
            json.dump(existing_results, f, indent=2)
        
        print(f"\nResults saved to: {self.results_file}")

if __name__ == "__main__":
    analyzer = ComputeAnalyser(
        matrix_sizes=[(128, 128), (512, 512), (1024, 1024), (2048, 2048)],
        bit_depths=[4, 8, 16, 32],
        num_iterations=50,
        warmup_iterations=5
    )
    
    results = analyzer.analyse()