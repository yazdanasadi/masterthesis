import torch

print("PyTorch version:", torch.__version__)
print("Compiled CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(0))