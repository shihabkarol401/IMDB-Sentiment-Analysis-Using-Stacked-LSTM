# Sentiment Analysis Using Stacked LSTM on IMDB 50K Movie Reviews

This project implements, evaluates, and deploys a deep learning–based sentiment classification system using a Stacked LSTM architecture trained on the IMDB 50K movie review dataset.

The system includes a complete NLP preprocessing pipeline and a production-style inference interface with probability breakdown and latency monitoring.

Live Demo: https://huggingface.co/spaces/shihabkarol/Sentiment-Analysis-Using-LSTM-on-IMDB-Dataset

## Problem Statement
Build a binary text classification model to predict whether a movie review is **Positive** or **Negative**, and deploy it as a real-time inference system.

## Model Architecture

- Embedding Layer
- Stacked LSTM Layers
- Dense Output Layer (Sigmoid activation)
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Maximum Sequence Length: 250

The LSTM architecture was selected to capture sequential dependencies in text data.

## Model Performance

| Metric | Score |
|--------|--------|
| Validation Accuracy | **89.06%** |
| Test Accuracy | **88.48%** |
| Test Loss | **0.2938** |

The minimal gap between validation and test accuracy indicates stable generalization with limited overfitting.

## NLP Preprocessing Pipeline

The following preprocessing steps were implemented:

- Lowercasing
- Stopword removal (negation preserved)
- HTML & URL cleaning
- Regex-based noise removal
- Lemmatization
- Tokenization
- Fixed-length sequence padding (250)

The tokenizer and trained model were serialized to ensure reproducible deployment.

## Deployment & Inference System

The model is deployed using Hugging Face Spaces with:

- Real-time sentiment prediction
- Positive & Negative probability breakdown
- Model confidence reporting
- Inference latency tracking (ms)
- Stateless prediction pipeline

Model saved in `.keras` format  
Tokenizer serialized via `pickle`

## Tech Stack

- Python
- TensorFlow / Keras
- NLTK
- NumPy
- Gradio
- Hugging Face Spaces

## Project Highlights

- End-to-end ML pipeline from training to deployment  
- Reproducible model artifact serialization  
- Evaluation-aware design (validation vs test comparison)  
- Production-style inference monitoring  

## Potential Improvements

- Add baseline comparison (TF-IDF + Logistic Regression)
- Integrate ROC-AUC and confusion matrix visualization
- Implement threshold tuning
- Containerize deployment with Docker
- Expose REST API using FastAPI
