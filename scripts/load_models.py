import joblib
import torch
import os

# Define a custom load function to force CPU loading
def custom_torch_load(f, *args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device("cpu")
    return torch_load_backup(f, *args, **kwargs)

# Monkey patch torch.load inside joblib.load
torch_load_backup = torch.load  # Backup the original function
torch.load = custom_torch_load  # Override with CPU-only loading

# Model file paths
distilbert_model_path = 'models/distilbert_model.joblib'
bert_topic_model_path = 'models/bertopic_model_max_compressed.joblib'
recommendation_model_path = 'models/recommendation_model.joblib'

# Load models
distilbert_model = joblib.load(distilbert_model_path)
bert_topic_model = joblib.load(bert_topic_model_path)
recommendation_model = joblib.load(recommendation_model_path)

# Restore the original torch.load function
torch.load = torch_load_backup

# Verify loading success
print("Models loaded successfully from the 'models' directory on CPU")
