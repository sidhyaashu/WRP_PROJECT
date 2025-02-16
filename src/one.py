#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced NLP Approach for Automated Knowledge Gap Detection in Student Responses
-------------------------------------------------------------------------------
This script demonstrates:
    - Data pre-processing (tokenization, stopword removal, normalization)
    - Semantic similarity analysis using T5 transformer model
    - Dynamic topic modeling using BERTopic
    - Real-time feedback generation
    - Comprehensive evaluation metrics

Requirements:
    - Python 3.x
    - transformers
    - torch
    - nltk
    - bertopic
    - scikit-learn
    - numpy

To install required packages, you can run:
    pip install transformers torch nltk bertopic scikit-learn numpy
"""

import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from transformers import T5Tokenizer, T5ForConditionalGeneration
from bertopic import BERTopic
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Download nltk stopwords (if not already downloaded)
nltk.download('stopwords')
STOPWORDS = set(stopwords.words("english"))

# Load T5 Model and Tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# --------------------------
# Data Collection (Sample Data)
# --------------------------
# Example student responses and a reference answer
student_responses = [
    "The process of photosynthesis converts light energy into chemical energy.",
    "Photosynthesis is the method by which plants use sunlight to create food, but they need water too.",
    "Plants perform a process to turn sunlight into energy, though the details are not clear.",
    "Plants do something with light energy to make food."
]

reference_answer = "Photosynthesis is the process by which green plants convert light energy into chemical energy by synthesizing sugars from carbon dioxide and water."

# --------------------------
# Pre-Processing Function
# --------------------------
def preprocess_text(text):
    """
    Preprocess text by lowercasing, removing non-alphabetic characters, and stopwords.
    """
    # Lowercase the text
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize and remove stopwords
    tokens = [word for word in text.split() if word not in STOPWORDS]
    # Return cleaned text
    return " ".join(tokens)

# Preprocess all responses
preprocessed_responses = [preprocess_text(resp) for resp in student_responses]
preprocessed_reference = preprocess_text(reference_answer)

# --------------------------
# Semantic Similarity Analysis Function
# --------------------------
def compute_similarity(student_text, reference_text):
    """
    Compute semantic similarity between student text and reference text using T5 model.
    """
    input_text = f"stsb sentence1: {student_text} sentence2: {reference_text}"
    inputs = t5_tokenizer.encode(input_text, return_tensors='pt')
    outputs = t5_model.generate(inputs, max_length=5)
    similarity_score = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return float(similarity_score)

# Compute similarity scores for each student response
print("\nSemantic Similarity Scores:")
for idx, response in enumerate(preprocessed_responses):
    sim_score = compute_similarity(response, preprocessed_reference)
    print(f"Response {idx+1} Similarity: {sim_score:.4f}")

# --------------------------
# Dynamic Topic Modeling with BERTopic
# --------------------------
def perform_topic_modeling(texts):
    """
    Perform dynamic topic modeling using BERTopic on the provided texts.
    """
    # Initialize BERTopic model
    topic_model = BERTopic()
    # Fit the model on the texts
    topics, _ = topic_model.fit_transform(texts)
    return topic_model, topics

# Combine student responses and reference answer for topic modeling analysis
all_texts = preprocessed_responses + [preprocessed_reference]
topic_model, topics = perform_topic_modeling(all_texts)

print("\nIdentified Topics:")
for topic in set(topics):
    print(f"Topic {topic}: {topic_model.get_topic(topic)}")

# --------------------------
# Real-Time Feedback Generation
# --------------------------
def generate_feedback(student_text, reference_text, threshold=0.8):
    """
    Generate feedback based on semantic similarity and identified topics.
    """
    sim_score = compute_similarity(student_text, reference_text)
    if sim_score >= threshold:
        feedback = "Good job! Your response aligns well with the reference answer."
    else:
        feedback = "Your response is missing some key concepts. Consider reviewing the following topics: "
        # Identify missing topics
        student_topics = perform_topic_modeling([student_text])[1]
        reference_topics = perform_topic_modeling([reference_text])[1]
        missing_topics = set(reference_topics) - set(student_topics)
        for topic in missing_topics:
            feedback += f"\n- {topic_model.get_topic(topic)}"
    return feedback

# Generate feedback for each student response
print("\nFeedback for Student Responses:")
for idx, response in enumerate(preprocessed_responses):
    feedback = generate_feedback(response, preprocessed_reference)
    print(f"\nResponse {idx+1} Feedback:\n{feedback}")

# --------------------------
# Evaluation Metrics
# --------------------------
# For demonstration purposes, assume binary classification: 1 for correct, 0 for incorrect
# In practice, these labels should come from a labeled dataset
true_labels = [1, 1, 0, 0]  # Ground truth labels
predicted_labels = [1 if compute_similarity(resp, preprocessed_reference) >= 0.8 else 0 for resp in preprocessed_responses]

precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print("\nEvaluation Metrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-Score:  {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# --------------------------
# Main Pipeline Execution
# --------------------------
def main():
    print("\n--- Running Full Pipeline ---\n")

    # 1. Pre-process Responses
    print("Preprocessing student responses...")
    processed_responses = [preprocess_text
::contentReference[oaicite:0]{index=0}]
 
