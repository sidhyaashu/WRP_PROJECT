#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced NLP Approach for Automated Knowledge Gap Detection in Student Responses
-------------------------------------------------------------------------------
This script demonstrates:
    - Advanced data pre-processing (tokenization, stopword removal, normalization, lemmatization)
    - Semantic similarity analysis using Sentence-BERT
    - Knowledge gap detection by comparing key concepts
    - Detailed feedback generation for open-ended responses
    - Visualization of data at each step of the pipeline

Requirements:
    - Python 3.x
    - spaCy (and the English model: en_core_web_sm)
    - nltk
    - sentence-transformers
    - scikit-learn
    - numpy
    - matplotlib
    - seaborn
    - wordcloud
    - networkx

To install required packages, you can run:
    pip install spacy nltk sentence-transformers scikit-learn numpy matplotlib seaborn wordcloud networkx
    python -m spacy download en_core_web_sm
"""

import re
import spacy
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# Download nltk data (if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load Sentence-BERT Model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# --------------------------
# Data Collection (Sample Data)
# --------------------------
# Example open-ended questions, reference answers, and student responses
questions = [
    "Explain the process of photosynthesis.",
    "Describe the significance of the water cycle in Earth's ecosystem.",
    "Discuss the impact of the Industrial Revolution on modern society."
]

reference_answers = [
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It involves the chlorophyll in leaves and generates oxygen as a byproduct.",
    "The water cycle is crucial as it distributes fresh water across the globe, supporting life. It involves processes like evaporation, condensation, precipitation, and infiltration, maintaining ecological balance.",
    "The Industrial Revolution marked a major turning point in history; it led to advancements in technology, manufacturing, and transportation, significantly influencing modern society's economic and social structures."
]

student_responses = [
    "Plants use sunlight to make food from carbon dioxide and water, releasing oxygen.",
    "Water evaporates, forms clouds, and comes back as rain, which is important for life.",
    "The Industrial Revolution changed how things were made and had effects on today."
]

# Assign identifiers
student_ids = [f'Student {i+1}' for i in range(len(student_responses))]

# --------------------------
# Pre-Processing Function
# --------------------------
def preprocess_text(text):
    """
    Preprocess text by lowercasing, removing non-alphabetic characters, 
    stopwords, and performing lemmatization.
    """
    # Lowercase the text
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in STOPWORDS]
    # Return cleaned text
    return " ".join(tokens)

# Preprocess all texts
preprocessed_responses = [preprocess_text(resp) for resp in student_responses]
preprocessed_references = [preprocess_text(ans) for ans in reference_answers]
preprocessed_questions = [preprocess_text(q) for q in questions]

# --------------------------
# Visualization: Word Frequencies
# --------------------------
def plot_word_frequencies(texts, title):
    """
    Plot the word frequency distribution of the provided texts.
    """
    all_words = " ".join(texts).split()
    freq_dist = nltk.FreqDist(all_words)
    freq_df = nltk.FreqDist(all_words).most_common(15)
    words = [word for word, freq in freq_df]
    freqs = [freq for word, freq in freq_df]

    plt.figure(figsize=(10,6))
    sns.barplot(x=freqs, y=words, palette='viridis')
    plt.title(f'Word Frequencies: {title}')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.show()

# Visualize word frequencies in student responses
plot_word_frequencies(preprocessed_responses, "Student Responses")

# Visualize word frequencies in reference answers
plot_word_frequencies(preprocessed_references, "Reference Answers")

# --------------------------
# Semantic Similarity Analysis
# --------------------------
# Encode responses and reference answers
response_embeddings = sbert_model.encode(preprocessed_responses)
reference_embeddings = sbert_model.encode(preprocessed_references)

# Compute similarity scores
similarity_scores = []
for resp_emb, ref_emb in zip(response_embeddings, reference_embeddings):
    sim_score = util.cos_sim(resp_emb, ref_emb).item()
    similarity_scores.append(sim_score)

# Display similarity scores
print("\nSemantic Similarity Scores:")
for idx, score in enumerate(similarity_scores):
    print(f"Response {idx+1} Similarity: {score:.4f}")

# --------------------------
# Visualization: Similarity Heatmap
# --------------------------
def plot_similarity_heatmap(response_embeddings, reference_embeddings):
    """
    Plot a heatmap of similarity scores between all student responses and reference answers.
    """
    sim_matrix = cosine_similarity(response_embeddings, reference_embeddings)
    plt.figure(figsize=(8,6))
    sns.heatmap(sim_matrix, annot=True, cmap='coolwarm',
                xticklabels=[f'Answer {i+1}' for i in range(len(reference_answers))],
                yticklabels=student_ids, fmt=".2f")
    plt.title('Semantic Similarity Heatmap')
    plt.xlabel('Reference Answers')
    plt.ylabel('Student Responses')
    plt.show()

plot_similarity_heatmap(response_embeddings, reference_embeddings)

# --------------------------
# Knowledge Gap Detection
# --------------------------
def extract_key_concepts(text):
    """
    Extract key concepts (nouns and noun phrases) from the text.
    """
    doc = nlp(text)
    concepts = set()
    for chunk in doc.noun_chunks:
        concepts.add(chunk.lemma_.lower())
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']:
            concepts.add(token.lemma_.lower())
    return concepts

# Identify missing concepts in student responses
def identify_missing_concepts(response_text, reference_text):
    response_concepts = extract_key_concepts(response_text)
    reference_concepts = extract_key_concepts(reference_text)
    missing_concepts = reference_concepts - response_concepts
    extra_concepts = response_concepts - reference_concepts
    return missing_concepts, extra_concepts

# --------------------------
# Visualization: Concept Venn Diagram
# --------------------------
def plot_concept_venn_diagram(response_idx):
    """
    Plot a Venn diagram of concepts in the student's response and reference answer.
    """
    from matplotlib_venn import venn2

    response_concepts = extract_key_concepts(preprocessed_responses[response_idx])
    reference_concepts = extract_key_concepts(preprocessed_references[response_idx])

    plt.figure(figsize=(8,6))
    venn2([response_concepts, reference_concepts], set_labels=('Student Response', 'Reference Answer'))
    plt.title(f'Concept Overlap for Response {response_idx+1}')
    plt.show()

# Visualize concept overlap for each response
for idx in range(len(student_responses)):
    plot_concept_venn_diagram(idx)

# --------------------------
# Detailed Feedback Generation
# --------------------------
def generate_feedback(response_idx, threshold=0.75):
    """
    Generate detailed feedback for a student's response.
    """
    sim_score = similarity_scores[response_idx]
    student_text = student_responses[response_idx]
    reference_text = reference_answers[response_idx]
    missing_concepts, extra_concepts = identify_missing_concepts(preprocessed_responses[response_idx], preprocessed_references[response_idx])

    feedback = f"Your Response:\n{student_text}\n"
    feedback += f"\nSimilarity Score: {sim_score:.4f}\n"
    if sim_score >= threshold:
        feedback += "Great work! Your response covers the key concepts.\n"
    else:
        if missing_concepts:
            feedback += "\nKey concepts you missed:\n"
            feedback += ", ".join(missing_concepts) + "\n"
        if extra_concepts:
            feedback += "\nAdditional concepts in your response:\n"
            feedback += ", ".join(extra_concepts) + "\n"
        if not missing_concepts and not extra_concepts:
            feedback += "Your response could be elaborated further to include more details.\n"
    return feedback

# Generate feedback for each student response
print("\nDetailed Feedback:")
for idx in range(len(student_responses)):
    print(f"\nFeedback for {student_ids[idx]}:\n{generate_feedback(idx)}")

# --------------------------
# Visualization: Missing Concepts Word Cloud
# --------------------------
def visualize_missing_concepts(response_idx):
    """
    Create a word cloud of missing concepts for a student's response.
    """
    missing_concepts, _ = identify_missing_concepts(preprocessed_responses[response_idx], preprocessed_references[response_idx])
    if missing_concepts:
        text = " ".join(missing_concepts)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Missing Concepts in {student_ids[response_idx]}')
        plt.show()

# Visualize missing concepts for each response
for idx in range(len(student_responses)):
    visualize_missing_concepts(idx)

# --------------------------
# Visualization: Similarity Score Distribution
# --------------------------
def plot_similarity_scores(similarity_scores):
    """
    Plot similarity scores for student responses.
    """
    plt.figure(figsize=(8, 6))
    sns.barplot(x=student_ids, y=similarity_scores, palette='viridis')
    plt.axhline(y=0.75, color='red', linestyle='--', label='Threshold')
    plt.xlabel('Student Responses')
    plt.ylabel('Similarity Score')
    plt.title('Semantic Similarity Scores of Student Responses')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

plot_similarity_scores(similarity_scores)

# --------------------------
# Visualization: Concept Network Graph
# --------------------------
def plot_concept_network(response_idx):
    """
    Plot a network graph of concepts in the reference answer and student's response.
    """
    missing_concepts, _ = identify_missing_concepts(preprocessed_responses[response_idx], preprocessed_references[response_idx])

    ref_concepts = extract_key_concepts(preprocessed_references[response_idx])
    doc = nlp(preprocessed_references[response_idx])

    edges = []
    for token in doc:
        if token.text in ref_concepts:
            for child in token.children:
                if child.text in ref_concepts:
                    edges.append((token.text, child.text))

    G = nx.Graph()
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, k=0.5, iterations=20)
    plt.figure(figsize=(10, 8))
    
    node_colors = []
    for node in G.nodes():
        if node in missing_concepts:
            node_colors.append('red')
        else:
            node_colors.append('green')
    
    nx.draw(G, pos, with_labels=True, node_size=5000, node_color=node_colors, font_size=10)
    plt.title(f'Concept Network for Reference Answer {response_idx+1}')
    plt.show()

# Plotting concept network for each reference answer
for idx in range(len(reference_answers)):
    plot_concept_network(idx)
