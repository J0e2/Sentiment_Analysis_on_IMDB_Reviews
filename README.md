Sentiment Analysis on IMDB Reviews \n
Overview
This project performs sentiment analysis on IMDB movie reviews using various machine learning models. The goal is to classify movie reviews as either positive or negative based on their text content.

Dataset
Source: IMDB Dataset of 50K Movie Reviews from Kaggle

Size: 50,000 reviews

Classes: Balanced dataset with 25,000 positive and 25,000 negative reviews

Project Structure
1. Data Preprocessing
The text preprocessing pipeline includes:

HTML tag removal

URL removal

Lowercasing

Punctuation and number removal

Tokenization and lemmatization

Stopword removal (including movie-specific terms like "film", "movie", "watch")

Custom text cleaning and processing

2. Feature Engineering
TF-IDF Vectorization with 5,000 features

N-gram range: (1,2) to capture both unigrams and bigrams

3. Models Implemented
Three machine learning models were trained and evaluated:

Multinomial Naive Bayes

Accuracy: 86.06%

Balanced performance across both classes

Logistic Regression ✅ Best Performing Model

Accuracy: 89.24%

Precision: 89-90% for both classes

Recall: 88-90% for both classes

Random Forest Classifier

Accuracy: 85.41%

Slightly lower but still competitive performance

4. Model Evaluation
All models were evaluated using:

Accuracy score

Classification reports (precision, recall, f1-score)

Confusion matrix analysis

Key Results
Best Model: Logistic Regression with 89.24% accuracy

The dataset is well-balanced, leading to consistent performance across both positive and negative classes

All models showed good generalization with similar performance on training and test sets

Dependencies
pandas, numpy

matplotlib, seaborn

scikit-learn

nltk

wordcloud

re, string

Usage
The notebook includes:

Complete data preprocessing pipeline

Model training and evaluation

Example of making predictions on new text

Visualization of results

Applications
This sentiment analysis model can be used for:

Automated movie review classification

Content moderation

Market research and analysis

Customer feedback analysis for streaming platforms

The project demonstrates effective text preprocessing techniques and comparative analysis of multiple machine learning algorithms for sentiment classification tasks.
