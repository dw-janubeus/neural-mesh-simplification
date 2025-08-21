import torch
import psutil

try:
  import GPUtil
except:
  print("Unable to import GPUtil")

# GPU verification and memory management for Colab T4
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Clear cache and optimize for T4
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
print(f"\nSystem RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB")