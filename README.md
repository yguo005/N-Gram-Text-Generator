# N-Gram Text Generator

This project builds a simple N-gram language model using the NLTK Gutenberg corpus and provides a Streamlit web app to generate text from user-provided starting words.

## Features

- Trains a 2-gram model on Gutenberg corpus text
- Applies preprocessing, tokenization, and vocabulary limiting
- Uses Laplace smoothing and backoff for unseen contexts
- Generates multiple text variations from a given prefix
- Includes a Streamlit interface for interactive generation

## Project Structure

- `app.py`: Streamlit UI and model loading/generation flow
- `ngram.py`: N-gram model logic, text generation, and evaluation helpers
- `requirements.txt`: Python dependencies

## Requirements

- Python 3.9+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```




git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

If your system asks for authentication, use GitHub Desktop, a personal access token, or SSH.
