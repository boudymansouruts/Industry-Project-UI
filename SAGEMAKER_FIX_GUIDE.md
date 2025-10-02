# SageMaker Deployment Guide

## 🚀 **Quick Fix for SageMaker Dependency Conflicts**

The dependency conflicts occur because SageMaker has pre-installed packages (AutoGluon, MLflow) that conflict with our requirements. Here's how to fix it:

### **Step 1: Use the SageMaker Installation Script**

Instead of `pip install -r requirements.txt`, use:

```bash
python install_sagemaker.py
```

This script will:
- ✅ Install packages compatible with AutoGluon
- ✅ Avoid version conflicts
- ✅ Test all critical imports
- ✅ Verify Whisper functionality

### **Step 2: Alternative Manual Installation**

If the script doesn't work, install packages individually:

```bash
# Install core packages with compatible versions
pip install 'torch>=2.2.0,<2.8.0'
pip install 'transformers>=4.30.0,<4.50.0'
pip install 'torchaudio>=2.2.0'
pip install 'librosa>=0.10.0'
pip install 'soundfile>=0.12.0'
pip install 'flask>=2.3.0'
pip install 'accelerate>=0.20.0'
pip install 'sentencepiece>=0.1.99'
pip install 'huggingface-hub>=0.15.0'
pip install 'datasets>=2.12.0,<4.2.0'
```

### **Step 3: Test Imports**

```python
# Test in Python console:
from transformers import WhisperProcessor, WhisperForConditionalGeneration
print("✅ Whisper imports successful!")
```

### **Step 4: Start Server**

```bash
python start_sagemaker_server.py
```

## 🔧 **Alternative Solutions**

### **Option A: Manual Import Fix**

If you prefer to use the original server, add this to the top of `risk_web_app.py`:

```python
# Add at the top of risk_web_app.py
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers>=4.30.0'])
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
```

### **Option B: Environment Setup**

```bash
# Set environment variables
export PYTHONPATH=$PWD
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/huggingface

# Create cache directories
mkdir -p /tmp/transformers_cache /tmp/huggingface

# Install dependencies
pip install transformers>=4.30.0 torch>=2.0.0

# Start server
python start_risk_server.py
```

## 📋 **SageMaker-Specific Requirements**

### **Required Packages:**
- `transformers>=4.30.0` (for WhisperProcessor)
- `torch>=2.0.0` (PyTorch backend)
- `torchaudio>=2.0.0` (audio processing)
- `librosa>=0.10.0` (audio analysis)
- `soundfile>=0.12.0` (audio I/O)
- `flask>=2.3.0` (web framework)

### **SageMaker Environment:**
- **Port**: 8080 (default)
- **Host**: 0.0.0.0 (all interfaces)
- **Cache**: `/tmp/` directories
- **Python Path**: Current directory

## 🎯 **Expected Results**

After running `python start_sagemaker_server.py`:

```
INFO:__main__:=== SageMaker Risk Audio Analysis Pipeline Startup ===
INFO:__main__:Checking dependencies...
INFO:__main__:✅ torch is available
INFO:__main__:✅ transformers is available
INFO:__main__:✅ librosa is available
INFO:__main__:Testing Whisper imports...
INFO:__main__:✅ WhisperProcessor and WhisperForConditionalGeneration imported successfully
INFO:__main__:Setting up environment...
INFO:__main__:✅ Environment setup complete
INFO:__main__:🚀 All checks passed, starting server...
INFO:__main__:Starting Flask server...
INFO:__main__:Starting server on 0.0.0.0:8080
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8080
 * Running on http://[::1]:8080
```

## 🚨 **Troubleshooting**

### **If still getting import errors:**

1. **Check Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

2. **Clear cache:**
   ```bash
   rm -rf /tmp/transformers_cache /tmp/huggingface
   ```

3. **Reinstall transformers:**
   ```bash
   pip uninstall transformers
   pip install transformers>=4.30.0
   ```

4. **Use conda environment:**
   ```bash
   conda create -n sagemaker python=3.9
   conda activate sagemaker
   pip install -r requirements.txt
   ```

## ✅ **Success Indicators**

- ✅ No import errors
- ✅ Server starts on port 8080
- ✅ WhisperProcessor loads successfully
- ✅ Web interface accessible
- ✅ Audio upload works
- ✅ Transcription processes

Your Risk Audio Analysis Pipeline should now work perfectly in SageMaker! 🎉
