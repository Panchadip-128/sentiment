import streamlit as st
import pandas as pd
import json
import torch
from transformers import DistilBertTokenizer
from scripts.load_models import distilbert_model, bert_topic_model
import os
import asyncio
import plotly.express as px
import plotly.graph_objects as go

# Handle asyncio event loop issue on Windows
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Streamlit app layout
st.title("Intelligent Customer Feedback Analyzer")
st.write("Analyze customer feedback for sentiment and topics.")

# File upload
uploaded_file = st.file_uploader("Upload a Feedback File (CSV, JSON, TXT)", type=["csv", "json", "txt"])

# Extract feedback text from different file formats
def extract_feedback(file):
    if file.type == "text/csv":
        df = pd.read_csv(file)
        feedback_text = df.iloc[:, 0].dropna().astype(str).tolist()  # Use first column as feedback
        return feedback_text
    elif file.type == "application/json":
        json_data = json.load(file)
        if isinstance(json_data, list):
            return [item.get("feedback", "") for item in json_data if "feedback" in item]
        elif isinstance(json_data, dict):
            return list(json_data.values())  # Use all values if feedback key doesn't exist
    elif file.type == "text/plain":
        return [file.getvalue().decode("utf-8")]
    return ["Unsupported file type"]

# Sentiment Analysis Function
def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    return f"The sentiment of the feedback is **{sentiment}**."

# Topic Modeling Function (Fix for BERTopic)
def get_topics(texts):
    topics, _ = bert_topic_model.transform(texts)  # Use `.transform()` instead of `.predict()`
    return topics if topics else ["No topic detected"]

# Graph Visualization Function
def visualize_topics(topics):
    topic_counts = pd.Series(topics).value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    fig = px.bar(topic_counts, x='Topic', y='Count', title='Topic Distribution', 
                 labels={'Topic': 'Topic', 'Count': 'Number of Feedbacks'},
                 color='Count', color_continuous_scale='Viridis')
    fig.update_layout(xaxis_title='Topic', yaxis_title='Number of Feedbacks', 
                      title_x=0.5, template='plotly_dark')
    st.plotly_chart(fig)

# Process uploaded file
if uploaded_file:
    feedback_text_list = extract_feedback(uploaded_file)

    if feedback_text_list:
        for i, feedback_text in enumerate(feedback_text_list):
            if st.button(f'Analyze: "{feedback_text[:30]}..."', key=f"analyze_{i}"):  # Add unique key
                st.subheader(f"Feedback Analysis {i+1}")
                
                # Sentiment Analysis
                sentiment_result = get_sentiment(feedback_text)
                st.write(sentiment_result)

                # Topic Modeling
                topic_result = get_topics([feedback_text])
                st.write(f"**Predicted Topic(s):** {topic_result}")
                
                # Visualize Topics
                visualize_topics(topic_result)
                
                st.markdown("---")  # Add separator between results
    else:
        st.error("Unable to extract feedback from the file.")
else:
    st.info("Please upload a feedback file to analyze.")

# User input for custom topic analysis
st.sidebar.header("Custom Topic Analysis")
custom_text = st.sidebar.text_area("Enter text for topic analysis:")
if st.sidebar.button("Analyze Custom Text"):
    if custom_text:
        # Sentiment Analysis for custom text
        custom_sentiment = get_sentiment(custom_text)
        st.sidebar.write(custom_sentiment)
        
        # Topic Modeling for custom text
        custom_topics = get_topics([custom_text])
        st.sidebar.write(f"**Predicted Topic(s):** {custom_topics}")
        visualize_topics(custom_topics)
    else:
        st.sidebar.error("Please enter text for analysis.")
