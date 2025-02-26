import joblib
import gdown
import os

def load_model_from_drive(file_url, model_name):
    """Downloads the model from Google Drive and loads it."""
    # Specify where to save the model
    model_folder = 'models'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Download the model using gdown
    output_path = os.path.join(model_folder, model_name)
    gdown.download(file_url, output_path, quiet=False)

    # Load and return the model using joblib
    model = joblib.load(output_path)
    return model

# URLs for the models on Google Drive (replace with your actual links)
distil_bert_model_url = 'https://drive.google.com/file/d/1WfjeGSQ7j4id1VSeGU8s2VzMBzNtImFT/view?usp=sharing'
bert_topic_model_url = 'https://drive.google.com/file/d/164n8QfrQF4RB2LlQzGe1BbaFmugbzBGR/view?usp=sharing'
recommendation_model_url = 'https://drive.google.com/file/d/17wFjVd9zTfHG33Eg7378Z6a1reohIkfE/view?usp=sharing'

#model names
distilbert_model_name = 'distilbert_model.joblib'
bert_topic_model_name = 'bertopic_model.joblib'
recommendation_model_name = 'recommendation_model.joblib'

# Load all three models
distilbert_model = load_model_from_drive(distilbert_model_url, distilbert_model_name)
bert_topic_model = load_model_from_drive(bert_topic_model_url, bert_topic_model_name)
recommendation_model = load_model_from_drive(recommendation_model_url, recommendation_model_name)

# Now, you can use these models in your Streamlit app

