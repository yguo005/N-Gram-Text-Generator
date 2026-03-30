"""
Streamlit Web Interface for N-Gram Text Generation

This module provides a web-based interface using Streamlit for the N-gram text
generation model. It allows users to interactively generate text based on their input.

Features:
1. User Interface:
   - Text input for starting words (prefix)
   - Slider for selecting output text length
   - Button to generate text
   - Multiple variations of generated text

2. Model Integration:
   - Uses the N-gram model from ngram.py
   - Processes text using NLTK's Gutenberg corpus
   - Generates multiple text variations for the same input

3. Interactive Elements:
   - Real-time text generation
   - Adjustable text length (5-30 words)
   - Informative help tooltips
   - About section with model details

Usage:
    To run the app:
    $ streamlit run app.py
    
    or
    
    $ python3 -m streamlit run app.py

Requirements:
    - streamlit>=1.28.0
    - nltk>=3.8.1
    - Python 3.x
"""

import streamlit as st
import nltk
from nltk.corpus import gutenberg
import random
from nltk import ngrams, FreqDist

# Import everything needed from ngram.py
from ngram import (
    preprocess_text,
    preprocess_tokenize_gutenberg_corpus,
    build_ngram_model,
    generate_sentence,
    vocabulary  # Import the global vocabulary
)

# Download required NLTK data
def download_nltk_data():
    nltk.download('gutenberg')
    nltk.download('punkt')

# Main Streamlit app
def main():
    st.title("Text Generator")
    st.write("Generate text using an N-gram language model")
    
    # Initialize the model
    with st.spinner("Loading the model... "):
        download_nltk_data()
        tokens = preprocess_tokenize_gutenberg_corpus()
        n = 2 
        
        # Create n-grams and their frequencies
        ngram_list = list(ngrams(tokens, n))
        ngram_frequencies = FreqDist(ngram_list)
        
        # Build the transition model
        transition_model = build_ngram_model(ngram_frequencies, n)
    
    # User inputs
    st.subheader("Generate Text")
    col1, col2 = st.columns(2)
    
    with col1:
        prefix_input = st.text_input(
            "Enter your starting words:",
            value="default text",
            help="Enter 1-3 words to start your generated text"
        )
        
    with col2:
        sentence_length = st.slider(
            "Choose output length:",
            min_value=5,
            max_value=30,
            value=15, # default value
            help="Number of words in the generated text"
        )
    
    # Generate button
    if st.button("Generate Text"):
        if prefix_input.strip():
            prefix = prefix_input.lower().split()
            
            # Generate multiple variations
            st.subheader("Generated Variations:")
            for i in range(3):  # Generate 3 different variations
                generated_text = generate_sentence(
                    prefix,  
                    sentence_length, 
                    n, 
                    transition_model 
                )
                st.write(f"{i+1}. {generated_text}")
        else:
            st.error("Please enter some starting words")
    
    # Add information about the model
    with st.expander("About this Text Generator"):
        st.write("""
        This text generator uses an N-gram language model trained on classic literature from the Gutenberg corpus.
        It generates text by:
        1. Taking your input words as a starting point
        2. Predicting the next word based on patterns in classic literature
        3. Repeating this process until reaching the desired length
        
        The model uses 2-grams (sequences of 2 words) to maintain better coherence in the generated text.
        """)

if __name__ == "__main__":
    main() 