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

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Notes

- On first run, NLTK datasets are downloaded (`gutenberg`, `punkt`, `punkt_tab`).
- `ngram.py` includes print-based exploratory output and evaluation code that runs at import time.

## How to Push to GitHub

If this is your first push from this folder:

```bash
git init
git add .
git commit -m "Initial commit"
```

Create a new empty GitHub repository (without README/license/gitignore), then connect and push:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

If your system asks for authentication, use GitHub Desktop, a personal access token, or SSH.