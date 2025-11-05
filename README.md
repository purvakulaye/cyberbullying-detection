# cyberbullying-detection
🧠 Cyberbullying Detection using NLP and Machine Learning
📖 Project Description

The Cyberbullying Detection System is an AI-powered project designed to automatically detect and classify online text as bullying or non-bullying. With the rapid increase of social media usage, cyberbullying has become a major issue affecting mental health and online safety.

This project leverages Natural Language Processing (NLP) and Machine Learning techniques to analyze comments, posts, or messages and identify offensive or bullying content. The system also performs emotion and sentiment analysis, provides bullying intensity percentages, and supports multiple languages through language detection and translation.

🚀 Key Features

🔤 Text Classification: Detects whether a given message or comment is bullying or non-bullying.

🌍 Multilingual Support: Automatically detects the language of the text and translates it into English for analysis.

❤️ Emotion Analysis: Identifies emotions such as anger, sadness, joy, fear, etc.

📊 Sentiment Intensity: Provides a sentiment score and the percentage of bullying vs non-bullying content.

💬 Interactive Web App: Built using Flask/Streamlit for easy text input and live results visualization.

📈 Visualization Dashboard: Displays analysis results using charts for better insights.

🧩 Tech Stack
Component	Tools / Libraries
Language	Python
NLP	NLTK, SpaCy, Transformers, langdetect, googletrans
Machine Learning	Scikit-learn, TensorFlow / PyTorch
Sentiment Analysis	TextBlob, VADER, BERT
Visualization	Matplotlib, Seaborn, Plotly
Web Framework	Flask / Streamlit
Dataset	Kaggle Cyberbullying Dataset, Hate Speech Dataset, Twitter Bullying Dataset
⚙️ Workflow

Data Preprocessing: Cleaning, tokenization, lemmatization, and removing noise.

Feature Extraction: Using TF-IDF or pre-trained embeddings (Word2Vec/BERT).

Model Training: Training ML or deep learning models to classify text.

Prediction & Analysis: Detects bullying, performs emotion and sentiment analysis.

Visualization & Deployment: Displays results in a user-friendly web dashboard.

📊 Example Output

Input: “You’re such a failure. Nobody likes you.”
Output:

Classification: 🚫 Bullying

Sentiment: Negative (-0.85)

Emotion: Anger (92%)

Language: English

Bullying Confidence: 96%

💡 Future Enhancements

Integration with live social media comment streams (Twitter, Instagram).

Real-time alert system for online platforms.

Enhanced multilingual support using large transformer models.

Addition of context-aware sarcasm detection.

👩‍💻 Contributors

Developed by: purva kulaye 
Course: BSc Data Science
Project Title: Automated Cyberbullying Detection using NLP and Machine Learning
