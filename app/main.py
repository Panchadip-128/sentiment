import streamlit as st
import pandas as pd
import json
import joblib
import gdown
import os

# Function to download and load model from Google Drive
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

# URLs for the models on Google Drive (using file IDs)
distilbert_model_url = 'https://drive.google.com/uc?export=download&id=1WfjeGSQ7j4id1VSeGU8s2VzMBzNtImFT'
bert_topic_model_url = 'https://drive.google.com/uc?export=download&id=164n8QfrQF4RB2LlQzGe1BbaFmugbzBGR'
recommendation_model_url = 'https://drive.google.com/uc?export=download&id=17wFjVd9zTfHG33Eg7378Z6a1reohIkfE'

# Model file names
distilbert_model_name = 'distilbert_model.joblib'
bert_topic_model_name = 'bertopic_model.joblib'
recommendation_model_name = 'recommendation_model.joblib'

# Load all three models
distilbert_model = load_model_from_drive(distilbert_model_url, distilbert_model_name)
bert_topic_model = load_model_from_drive(bert_topic_model_url, bert_topic_model_name)
recommendation_model = load_model_from_drive(recommendation_model_url, recommendation_model_name)

# Streamlit app layout
st.title("Intelligent Customer Feedback Analyzer")
st.write("Analyze customer feedback for sentiment, topics, and get personalized recommendations.")

# User input for customer feedback file
uploaded_file = st.file_uploader("Upload a Feedback File (CSV, JSON, TXT)", type=["csv", "json", "txt"])

# Function to extract feedback text from different file formats
def extract_feedback(file):
    if file.type == "text/csv":
        # If the file is CSV, read it and extract all text content (even if unlabelled)
        df = pd.read_csv(file)
        feedback_text = []
        for column in df.columns:
            feedback_text.extend(df[column].dropna().astype(str).tolist())  # Include all text in the CSV
        return feedback_text
    elif file.type == "application/json":
        # If the file is JSON, try to parse and extract the feedback text
        json_data = json.load(file)
        feedback_text = []
        # Adjust this depending on how the JSON is structured (e.g., each feedback is a list of feedback entries)
        if isinstance(json_data, list):
            feedback_text = [item.get('feedback', '') for item in json_data if 'feedback' in item]
        elif isinstance(json_data, dict):
            feedback_text = list(json_data.values())  # Include all values if feedback key doesn't exist
        return feedback_text
    elif file.type == "text/plain":
        # If the file is plain text, read it directly
        return [file.getvalue().decode("utf-8")]
    else:
        return ["Unsupported file type"]

# Display error or feedback extraction status
if uploaded_file:
    feedback_text_list = extract_feedback(uploaded_file)

    # If feedback is extracted, analyze it
    if feedback_text_list:
        for feedback_text in feedback_text_list:
            if st.button(f'Analyze Feedback: "{feedback_text[:30]}..."'):
                # Sentiment Analysis
                sentiment = distilbert_model.predict([feedback_text])
                sentiment_result = 'Positive' if sentiment == 1 else 'Negative'
                st.write(f"Sentiment: {sentiment_result}")

                # Topic Modeling
                topics = bert_topic_model.predict([feedback_text])
                st.write(f"Predicted Topic(s): {topics}")

                # Recommendation System
                recommendations = recommendation_model.predict([feedback_text])
                st.write(f"Recommended Actions: {recommendations}")
    else:
        st.error("Unable to extract feedback from the file.")
else:
    st.info("Please upload a feedback file to analyze.")
