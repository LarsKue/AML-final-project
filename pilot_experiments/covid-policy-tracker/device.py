
import torch


# globally available device
device = torch.device("cpu")
if torch.cuda.is_available():
    print("Using CUDA.")
    device = torch.device("cuda")
