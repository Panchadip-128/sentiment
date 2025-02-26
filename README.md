# Intelligent Customer Feedback Analyzer
## **Project Overview**
### How It Will Help

This tool is a game-changer for businesses across industries that receive customer feedback in large volumes. By using advanced sentiment analysis, topic modeling, and personalized recommendation generation, it helps businesses quickly identify recurring issues, track sentiment trends, and understand customer needs at a deeper level. Whether it’s positive, negative, or neutral feedback, this analyzer’s sentiment analysis capabilities ensure that every review is assessed accurately and efficiently. 

Gone are the days of manually categorizing feedback or relying on outdated, inefficient tools. The **Intelligent Customer Feedback Analyzer** can process hundreds or even thousands of customer reviews in a matter of minutes, drastically reducing the time and effort required to gain meaningful insights. This means businesses can act quickly to address customer concerns, improve their offerings, and ultimately enhance the customer experience. The time saved on analysis can now be invested in implementing changes and strategies based on the data provided, making operations smoother and more responsive.

### Why It’s Revolutionary

This tool is not just another feedback analysis system; it's a **breakthrough solution** that combines cutting-edge machine learning techniques, including DistilBERT-based sentiment analysis and BERTopic-based topic modeling, to provide businesses with insights that are not just surface-level but rich and actionable. By employing advanced algorithms and transformer models, it uncovers deep patterns in customer feedback that would otherwise be impossible to detect manually. 

As businesses continue to scale, the feedback volume grows exponentially. However, the **Intelligent Customer Feedback Analyzer** is designed to scale with these growing demands, making it an essential long-term tool for companies of all sizes. The fact that it can process various input formats—CSV, JSON, TXT, and XLSX—means that it’s adaptable to any existing data structure businesses may have. Moreover, the analyzer not only processes feedback but also generates personalized recommendations for improvements based on the insights, empowering businesses to stay ahead of the competition by continually evolving their products and services.
### **Novelty and Uniqueness**

The **Intelligent Customer Feedback Analyzer** is a truly **novel** solution, distinct from anything currently available in the market. While many tools exist for customer feedback analysis, none combine **sentiment analysis**, **topic modeling**, and **recommendation systems** in such a **cohesive, automated** manner. This integration of advanced AI technologies, such as **DistilBERT** for sentiment analysis, **BERTopic** for topic modeling and recommendations, offers a level of **accuracy and depth** not seen in existing solutions.

What truly sets this project apart is its ability to process feedback in **multiple formats** (CSV, JSON, TXT, XLSX), which is rare in most feedback systems that are limited to specific data formats or require cumbersome preprocessing. This flexibility makes the tool easily adaptable for businesses of various sizes and industries, further enhancing its **accessibility and usability**.

Additionally, unlike existing tools that offer generic, surface-level insights, our system offers **context-aware, actionable insights**. It doesn't just analyze sentiment or categorize feedback; it delves deeper by uncovering **topics** and **patterns** in the data and providing tailored **recommendations** for improvement. This approach provides businesses with a much richer understanding of their customers' needs and pain points, enabling them to act on feedback in a **more impactful, strategic** way.

In essence, the **Intelligent Customer Feedback Analyzer** is not just an upgrade on existing tools—it's a **revolutionary** new approach that **combines multiple cutting-edge technologies** to provide **deeper insights** and **faster results**, while being **highly flexible** in terms of data input formats. It fills a crucial gap in the market by offering businesses the **ability to respond quickly and effectively** to customer feedback, driving **continuous improvement** and **business growth**.


### Future Integration Potential

As the tool matures, it offers tremendous potential for integration with other systems and platforms. Later stages of development could involve integrating the analyzer with CRM systems, customer support tools, and product management software. This seamless integration would allow businesses to feed real-time customer feedback into their existing workflows, enabling continuous monitoring of customer satisfaction, behavior, and sentiment. The tool could even be connected with marketing and social media platforms to track sentiment across various channels, providing a comprehensive view of the customer experience.

Additionally, with future machine learning model enhancements, the **Intelligent Customer Feedback Analyzer** can continuously evolve to provide even more sophisticated insights—allowing businesses to stay agile and proactive, rather than reactive. By integrating with automated customer service systems, it could even allow businesses to automatically respond to certain types of feedback, creating a more streamlined and efficient customer service process.

Ultimately, this tool is not just a solution to today’s customer feedback problems—it’s a forward-looking platform that grows with businesses, giving them the power to continuously adapt, improve, and succeed in a customer-centric world.


## **Features**
1. **Sentiment Analysis**:
   - **VADER Sentiment Analyzer**: A lexicon-based approach for analyzing sentiment.
   - **DistilBERT Sentiment Analysis**: A transformer-based approach for fine-grained sentiment classification using a fine-tuned DistilBERT model.
   - ### Flexible Data Input Handling
One of the standout features of the **Intelligent Customer Feedback Analyzer** is its ability to seamlessly accept and process customer feedback data in any format, ensuring maximum flexibility and accessibility for users. Whether the feedback comes in the form of a **CSV**, **JSON**, **TXT**, or **XLSX** file, our tool effortlessly ingests and processes it without any hassle. This means that businesses can easily integrate their existing data sources into the system, regardless of the format they use. By supporting a wide range of file types, the analyzer makes it incredibly easy for users to upload and work with data in the format they’re most comfortable with, removing barriers to adoption and simplifying the workflow. This flexibility allows businesses to quickly get started with analyzing feedback, saving time and resources while ensuring that they can leverage the tool's powerful features to unlock valuable insights from any source of data.
   
2. **Topic Modeling**:
   - Uses **BERTopic** for extracting meaningful topics from customer feedback, with features like:
     - **Vectorization** using **CountVectorizer** for feature extraction.
     - **Dimensionality Reduction** with **UMAP** to reduce the embeddings' dimensionality.
     - **Clustering** with **HDBSCAN** for grouping topics.
     - **Topic Representation** via **c-TF-IDF** for extracting keywords associated with each topic.
     - **Enhancement** of topics using transformer-based embeddings for more accurate and contextual clustering.
     - <img width="502" alt="image" src="https://github.com/user-attachments/assets/1f340190-4322-48e7-9a91-894b58ef759d" />
     <img width="488" alt="image" src="https://github.com/user-attachments/assets/5f069718-9121-437f-92a4-5b88c382f66f" />

3. **Recommendation System**:
   - Generates **similar reviews** based on topics using **Cosine Similarity** and **Sentence Embeddings**.
   - Customizable similarity balancing and topic thresholding to fine-tune results.
   - **Precomputed Embeddings** for faster inference and joblib saving for reusability.

## **Model Pipeline**
The analysis is broken down into the following key steps:

### 1. **Text Preprocessing & Sentiment Analysis**
   - **Data Loading**: Customer reviews are loaded and preprocessed for further analysis.
   - <img width="471" alt="image" src="https://github.com/user-attachments/assets/22dbb722-f1b3-4a44-9b98-eff5d5818d93" />
   - **Sentiment Analysis**: 
     - **VADER Sentiment Analyzer** is used for lexicon-based sentiment classification.
     - **DistilBERT** is fine-tuned on the Amazon Reviews dataset for higher accuracy and better sentiment classification.

#### **Advanced Preprocessing**
   - **Emoji Handling**: Emojis are preprocessed by converting them into their respective textual descriptions. This enhances the sentiment analysis by accounting for the emotional tone conveyed by emojis.
   - <img width="510" alt="image" src="https://github.com/user-attachments/assets/49251e47-53ff-47e8-986a-343efe099325" />

   - **Negation Handling**: Negation words (e.g., "not", "never") are specially handled to avoid misinterpretation of sentiment.
   - <img width="555" alt="image" src="https://github.com/user-attachments/assets/6f200c91-04e0-4d94-9795-81dc000e4567" />

   - **Bigrams**: Bigrams (two consecutive words) are extracted to capture commonly used phrases that might convey more meaning than single words.
   - <img width="698" alt="image" src="https://github.com/user-attachments/assets/4447b07f-06d9-460f-afe1-a6e38e162b7e" />

   - **Word Cloud Generation**: Visualizations of frequent words provide insight into common themes across feedback.
   - **Stopword Removal**: Common, non-informative words (like "the", "is", etc.) are removed to focus on important content.

### 2. **Topic Modeling with BERTopic**
   - **Feature Extraction**: Text is vectorized using **CountVectorizer** to convert reviews into numerical features.
   - <img width="332" alt="image" src="https://github.com/user-attachments/assets/edf0a2e1-2f6c-4a66-a26c-fc46e4a31600" />
   - **Dimensionality Reduction**: **UMAP** reduces the dimensionality of embeddings to make topic identification easier.
   - **Clustering**: **HDBSCAN** is used for clustering reviews based on topics.
   - **Topic Representation**: **c-TF-IDF** is used to extract important keywords for each topic.
   <img width="520" alt="image" src="https://github.com/user-attachments/assets/1ab5fb0c-1be5-4dc5-adb5-612457ad21ef" />
   - **Enhancing Topics**: Transformer-based embeddings enhance the clustering, providing more accurate topic representation.

### 3. **Recommendation System**
   - **Topic Matching**: Reviews are assigned to topics, and **Sentence Transformers** compute embeddings to find similarities between reviews.
   - <img width="589" alt="image" src="https://github.com/user-attachments/assets/d258f1bf-89a2-4b20-b133-a7578b340b82" />

   - **Cosine Similarity**: This metric is used to match reviews based on topic keywords.
   - <img width="339" alt="image" src="https://github.com/user-attachments/assets/437eb077-5ecc-413e-b6a1-637fbda435da" />
<img width="377" alt="image" src="https://github.com/user-attachments/assets/0c0ac431-33d1-4eb4-b0e4-11a48b7a6868" />

   - **Customization**: Users can tweak the alpha value for similarity balance and adjust the threshold for topic relevance.
   - **Joblib Saving**: All models are saved as joblib files for easy reuse without the need for retraining.

### 4. **Model Setup & Training**
   - **DistilBERT Fine-Tuning**: The pre-trained **distilbert-base-uncased** model is fine-tuned on the Amazon Reviews dataset.
   - **Metrics**: The model achieves a final **accuracy of 93.75%**, **Precision** of 93.78%, **Recall** of 93.75%, and **F1 Score** of 93.75%.
   - **Training Process**: Hugging Face's **Trainer API** is used to fine-tune the model, with mixed precision for faster training and efficient memory use.

### 5. **Model Evaluation**
   - Sentiment analysis is evaluated on a validation dataset using metrics like **Accuracy**, **Precision**, **Recall**, and **F1 Score**.
   - The model's performance is optimal with the **Jaccard score** of 88.13%, confirming its effectiveness in classifying sentiment accurately.

## **Deployment**
The model is deployed using **Streamlit** for real-time customer feedback analysis. However, the **joblib** files are too large to upload to GitHub, so I've provided **Google Drive links** to access these files directly. You can download the joblib files and use them in your environment.
## **Google Drive Links for Joblib Files**

The model's **joblib** files, which are too large to upload to GitHub, are available for download from the following **Google Drive** link. This contains all three joblib files required for the sentiment analysis and recommendation system pipeline.

[Download Joblib Files from Google Drive](https://drive.google.com/drive/folders/1x9II7BGCidEcaM1strWCy9KTcoRWhyz5?usp=sharing)

You can use these files in your environment by downloading them and loading them into your project. Make sure to place them in the appropriate directories and load them using `joblib.load()` when running the model.

### **Next Steps**:
1. **Streamlit Dashboard Enhancement**: The next exciting phase of the project is to enhance the Streamlit dashboard, transforming it into a **real-time, interactive interface**. This dashboard will feature multiple dynamic data visualizations, providing users with an intuitive and comprehensive view of customer feedback insights. With real-time updates, businesses will be able to track sentiment, discover emerging trends, and receive actionable insights at a glance. By resolving the requirements.txt and dependency issues, we will ensure smooth integration, enabling seamless user interaction and more robust data exploration capabilities. The final goal is to deliver a fully functional, user-friendly dashboard that empowers businesses to monitor and analyze customer feedback in a visually compelling and actionable manner, improving decision-making and operational efficiency.

<img width="592" alt="image" src="https://github.com/user-attachments/assets/cadd72a7-9faf-4a0b-8765-5388acecea3c" />


2. **Scalability**: Improve the processing time and scalability of the pipeline for large datasets by optimizing memory usage and inference speed.
3. **Model Enhancement**: Continue to refine the sentiment analysis model and BERTopic topic modeling for better accuracy and context.

## **Metrics and Evaluation**

### **Training Results of Sentiment Analysis Model(using DISTILBERT)**:
- **Training Loss**: 0.000700 (final epoch)
- **Validation Loss**: 0.537547 (final epoch)
- **Accuracy**: 93.75%
- **Precision**: 93.78%
- **Recall**: 93.75%
- **F1 Score**: 93.75%
- **IoU (Jaccard score)**: 88.13%

- <img width="575" alt="image" src="https://github.com/user-attachments/assets/d3893a0f-6fe2-4abe-bb2d-3bcf4ddb40e1" />


### **Sentiment Analysis Metrics**:
- **VADER Sentiment Accuracy**: Evaluated on test data, showcasing the lexicon-based accuracy in quick sentiment classification.
- **DistilBERT Fine-Tuned Accuracy**: 93.75% accuracy achieved, demonstrating the transformer-based model's fine-tuned performance.
- **Precision and Recall**: Both metrics show balanced performance, ensuring the model correctly identifies both positive and negative sentiments.
- **Jaccard Score (IoU)**: The Jaccard score of 88.13% confirms the fine-grained capability of the model to capture sentiment accurately, even in cases with high word overlap.

## **Installation Instructions**
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-name>

2. pip install -r requirements.txt
3. Download the joblib files from the provided Google Drive links.
Run the Streamlit dashboard:
streamlit run app.py


