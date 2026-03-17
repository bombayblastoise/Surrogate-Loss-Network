#!/usr/bin/env python3
"""
Comprehensive GPU setup testing script for XGBoost and NVIDIA CUDA
"""
import sys
import numpy as np

print("=" * 80)
print("GPU SETUP DIAGNOSTIC TEST FOR XGBOOST")
print("=" * 80)

# Test 1: Check CUDA availability
print("\n[TEST 1] Checking CUDA/NVIDIA GPU Availability")
print("-" * 80)
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠ PyTorch CUDA not available")
except ImportError:
    print("⚠ PyTorch not installed (optional)")

# Test 2: Check XGBoost version and GPU support
print("\n[TEST 2] Checking XGBoost Installation and GPU Support")
print("-" * 80)
try:
    import xgboost as xgb
    print(f"✓ XGBoost version: {xgb.__version__}")
    
    # Check if GPU support is compiled in
    print("\n  Checking XGBoost build configuration...")
    try:
        # Try to get build info
        import xgboost
        build_info = xgboost.get_config()
        print(f"  XGBoost config: {build_info}")
    except:
        print("  Could not retrieve build config")
    
    # List available tree methods
    print("\n  Available tree methods:")
    try:
        test_params = {
            'n_estimators': 1,
            'tree_method': 'hist',  # Safe default
            'random_state': 42,
            'verbosity': 0,
        }
        
        # Try different tree methods to see which work
        tree_methods_to_test = ['auto', 'exact', 'approx', 'hist', 'gpu_hist', 'gpu_approx']
        
        X_test = np.random.rand(10, 5)
        y_test = np.random.randint(0, 2, 10)
        
        for method in tree_methods_to_test:
            try:
                params = test_params.copy()
                params['tree_method'] = method
                model = xgb.XGBClassifier(**params, objective='binary:logistic')
                model.fit(X_test, y_test, verbose=False)
                print(f"    ✓ '{method}' works")
            except Exception as e:
                error_msg = str(e)
                if "Invalid Input" in error_msg or "valid values are" in error_msg:
                    print(f"    ✗ '{method}' NOT AVAILABLE")
                else:
                    print(f"    ? '{method}' error: {str(e)[:60]}")
    except Exception as e:
        print(f"  Error testing tree methods: {e}")

except ImportError:
    print("✗ XGBoost not installed!")
    sys.exit(1)

# Test 3: Check if CUDA toolkit is installed
print("\n[TEST 3] Checking CUDA Toolkit Installation")
print("-" * 80)
import subprocess
import os

# Try to find CUDA
cuda_found = False
try:
    result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        nvcc_path = result.stdout.strip()
        print(f"✓ NVIDIA CUDA Compiler (nvcc) found at: {nvcc_path}")
        
        # Get CUDA version
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        cuda_version = result.stdout.strip().split('\n')[-1] if result.stdout else "Unknown"
        print(f"  CUDA Version: {cuda_version}")
        cuda_found = True
    else:
        print("✗ NVIDIA CUDA Compiler (nvcc) not found in PATH")
except Exception as e:
    print(f"✗ Error checking for nvcc: {e}")

# Check for nvidia-smi
print("\n  Checking NVIDIA GPU with nvidia-smi...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("✓ nvidia-smi found. GPU Info:")
        for line in result.stdout.split('\n')[:20]:  # First 20 lines
            if line.strip():
                print(f"  {line}")
    else:
        print("✗ nvidia-smi failed")
except Exception as e:
    print(f"✗ nvidia-smi error: {e}")

# Test 4: GPU-accelerated training test
print("\n[TEST 4] Attempting GPU-Accelerated XGBoost Training")
print("-" * 80)

X_train = np.random.rand(1000, 20)
y_train = np.random.randint(0, 2, 1000)
X_val = np.random.rand(200, 20)
y_val = np.random.randint(0, 2, 200)

# Try training with different configurations
gpu_configs = [
    {
        'name': 'CPU Hist (Safe Default)',
        'params': {
            'tree_method': 'hist',
            'n_jobs': 1,
        }
    },
    {
        'name': 'GPU Hist (NVIDIA GPU)',
        'params': {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
        }
    },
    {
        'name': 'GPU Approx (Alternative GPU)',
        'params': {
            'tree_method': 'gpu_approx',
            'gpu_id': 0,
        }
    },
]

for config in gpu_configs:
    print(f"\n  Testing: {config['name']}")
    try:
        params = {
            'n_estimators': 10,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'verbosity': 0,
        }
        params.update(config['params'])
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Time the prediction
        import time
        start = time.time()
        pred = model.predict(X_val)
        pred_time = time.time() - start
        
        print(f"    ✓ SUCCESS - Training completed in {pred_time*1000:.2f}ms")
        
    except Exception as e:
        error_str = str(e)
        if "Invalid Input" in error_str:
            print(f"    ✗ Tree method not available (not compiled in)")
        elif "gpu_id" in error_str:
            print(f"    ✗ GPU parameter error: {error_str[:60]}")
        else:
            print(f"    ✗ Error: {error_str[:60]}")

# Test 5: Summary and recommendations
print("\n" + "=" * 80)
print("DIAGNOSIS AND RECOMMENDATIONS")
print("=" * 80)

try:
    import xgboost as xgb
    
    # Try a simple CPU test
    try:
        model = xgb.XGBClassifier(tree_method='hist', n_estimators=1, random_state=42)
        X_test = np.random.rand(10, 5)
        y_test = np.random.randint(0, 2, 10)
        model.fit(X_test, y_test, verbose=False)
        print("\n✓ XGBoost CPU (hist) is working properly")
    except:
        print("\n✗ XGBoost CPU (hist) not working")
    
    # Check if GPU hist is available
    gpu_available = False
    try:
        model = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1, random_state=42, gpu_id=0)
        X_test = np.random.rand(10, 5)
        y_test = np.random.randint(0, 2, 10)
        model.fit(X_test, y_test, verbose=False)
        print("✓ XGBoost GPU (gpu_hist) is available and working!")
        gpu_available = True
    except Exception as e:
        if "Invalid Input" in str(e):
            print("✗ XGBoost GPU (gpu_hist) NOT COMPILED IN - XGBoost needs to be installed with GPU support")
            print("\n  To fix this, you need to reinstall XGBoost with GPU support:")
            print("    pip uninstall xgboost -y")
            print("    pip install xgboost-gpu")
            print("  OR")
            print("    conda install -c conda-forge py-xgboost-gpu")
        else:
            print(f"✗ XGBoost GPU error: {str(e)[:100]}")
    
    print("\n" + "-" * 80)
    if gpu_available:
        print("RECOMMENDATION: Use GPU acceleration (gpu_hist)")
        print("  - Set tree_method='gpu_hist'")
        print("  - Set gpu_id=0 (or appropriate GPU ID)")
    else:
        print("RECOMMENDATION: Use CPU acceleration with optimal parameters")
        print("  - Set tree_method='hist'")
        print("  - Set n_jobs=-1 for CPU parallelization")
        print("\nTo enable GPU support:")
        print("  1. Verify NVIDIA GPU is installed: nvidia-smi")
        print("  2. Verify CUDA toolkit is installed: nvcc --version")
        print("  3. Reinstall XGBoost with GPU support:")
        print("     pip install xgboost-gpu")
    
except Exception as e:
    print(f"\nError in diagnosis: {e}")

print("\n" + "=" * 80)
