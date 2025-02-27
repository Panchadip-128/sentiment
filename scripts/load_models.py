import joblib
import torch
import gdown
import os

# Define a custom load function to force CPU loading
def custom_torch_load(f, *args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device("cpu")
    return torch_load_backup(f, *args, **kwargs)

# Monkey patch torch.load inside joblib.load
torch_load_backup = torch.load  # Backup the original function
torch.load = custom_torch_load  # Override with CPU-only loading

# Function to download the model from Google Drive
def download_model(file_url, output_path):
    if not os.path.exists(output_path):
        gdown.download(file_url, output_path, quiet=False)

# URLs for the models on Google Drive (using file IDs)
distilbert_model_url = 'https://drive.google.com/uc?export=download&id=1WfjeGSQ7j4id1VSeGU8s2VzMBzNtImFT'
bert_topic_model_url = 'https://drive.google.com/uc?export=download&id=164n8QfrQF4RB2LlQzGe1BbaFmugbzBGR'
recommendation_model_url = 'https://drive.google.com/uc?export=download&id=17wFjVd9zTfHG33Eg7378Z6a1reohIkfE'

# Model file names
distilbert_model_name = 'distilbert_model.joblib'
bert_topic_model_name = 'bertopic_model_max_compressed.joblib'
recommendation_model_name = 'recommendation_model.joblib'

# Download the models if not already present
download_model(distilbert_model_url, distilbert_model_name)
download_model(bert_topic_model_url, bert_topic_model_name)
download_model(recommendation_model_url, recommendation_model_name)

# Load models
distilbert_model = joblib.load(distilbert_model_name)
bert_topic_model = joblib.load(bert_topic_model_name)
recommendation_model = joblib.load(recommendation_model_name)

# Restore the original torch.load function
torch.load = torch_load_backup

# Verify loading success
print("Models loaded successfully on CPU")
