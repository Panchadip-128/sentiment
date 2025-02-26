import joblib
import torch

# Define a custom load function to force CPU loading
def custom_torch_load(f, *args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device("cpu")
    return torch_load_backup(f, *args, **kwargs)

# Monkey patch torch.load inside joblib.load
torch_load_backup = torch.load  # Backup the original function
torch.load = custom_torch_load  # Override with CPU-only loading

# Load the model
topic_model = joblib.load("bertopic_model_max_compressed.joblib")

# Restore the original torch.load function
torch.load = torch_load_backup

# Verify loading success
print("Model loaded successfully on CPU")
