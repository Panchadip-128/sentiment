import streamlit as st
import pandas as pd
from scripts.load_models import distilbert_model, bert_topic_model, recommendation_model
import gdown
# Streamlit app layout
st.title("Intelligent Customer Feedback Analyzer")
st.write("Analyze customer feedback for sentiment, topics, and get personalized recommendations.")

# User input for customer feedback file
uploaded_file = st.file_uploader("Upload a Feedback File (CSV, JSON, TXT)", type=["csv", "json", "txt"])

# Function to extract feedback text from different file formats
def extract_feedback(file):
    if file.type == "text/csv":
        df = pd.read_csv(file)
        feedback_text = []
        for column in df.columns:
            feedback_text.extend(df[column].dropna().astype(str).tolist())  # Include all text in the CSV
        return feedback_text
    elif file.type == "application/json":
        json_data = json.load(file)
        feedback_text = []
        if isinstance(json_data, list):
            feedback_text = [item.get('feedback', '') for item in json_data if 'feedback' in item]
        elif isinstance(json_data, dict):
            feedback_text = list(json_data.values())  # Include all values if feedback key doesn't exist
        return feedback_text
    elif file.type == "text/plain":
        return [file.getvalue().decode("utf-8")]
    else:
        return ["Unsupported file type"]

# Display error or feedback extraction status
if uploaded_file:
    feedback_text_list = extract_feedback(uploaded_file)

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

