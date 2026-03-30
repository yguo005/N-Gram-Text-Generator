"""
N-Gram Language Model for Text Generation

This module implements an N-gram language model trained on the Gutenberg corpus. 
The model can generate new text based on a given prefix.

Key Components:
1. Data Preprocessing:
   - Loads text from the Gutenberg corpus
   - Cleans and tokenizes the text
   - Creates a vocabulary of common words

2. N-gram Model:
   - Creates n-grams from the text (sequences of n words)
   - Builds a probability model for word transitions
   - Uses Laplace smoothing to handle unseen combinations

3. Text Generation:
   - Takes a prefix (starting words) as input
   - Generates text by predicting next words based on context
   - Uses probabilistic selection for natural variation

Usage:
    from ngram import generate_sentence, build_ngram_model
    
    # Generate text with a prefix
    text = generate_sentence(
        prefix=['a', 'king'],
        sentence_length=15,
        n=2,
        transition_model=model
    )

Features:
- Handles unknown words using <UNK> token
- Implements backoff for unseen contexts
- Limits vocabulary to most common words for efficiency
- Uses Laplace smoothing for better probability estimates


"""



# 1. Data collection and preprocessing:
import nltk
from nltk.corpus import gutenberg
import re
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Print available texts in the Gutenberg corpus
print("Available texts in the Gutenberg corpus:")
print(gutenberg.fileids())

# a. collect a dataset of text.
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('punkt_tab')
# gutenberg.raw()

# b. clean and preprocess the data
def preprocess_text(text):
    text = text.lower()
    # Remove non-alphanumeric characters (keeping spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# c. Tokenize the text into words.
def preprocess_tokenize_gutenberg_corpus():
  all_tokens = []
  for file_id in gutenberg.fileids():
    raw_text = gutenberg.raw(file_id)
    # remove non-alphanumeric and convert to lower case
    cleaned_text = preprocess_text(raw_text)
    # tokenize the cleaned text
    tokens = word_tokenize(cleaned_text)
    all_tokens.extend(tokens) # extend(iterable): Adds all the elements of the iterable to the end of the list. If iterable is a list (like tokens), extend() merges the two lists by adding each element of tokens individually to the end of all_tokens; append(item): Adds a single item to the end of the list. If used all_tokens.append(tokens), would be adding the entire list tokens as a single element to all_tokens. This is not what i want when building a language model, as need a flat list of all words
  return all_tokens

# get the list of tokens from the entire corpus
tokens = preprocess_tokenize_gutenberg_corpus()

# After vocabulary creation, limit vocabulary size
from collections import Counter
word_counts = Counter(tokens)
VOCAB_LIMIT = 2000  # try 5000 at first
common_words = dict(word_counts.most_common(VOCAB_LIMIT))
vocabulary = set(common_words.keys())
vocabulary_size = len(vocabulary)

# 2. Model implementation:
# (a) Create n-grams from the tokenized text and calculate their frequencies in the dataset
import random
from nltk import ngrams, FreqDist

n = 2  # try 2,3,4
# generate n-grams
ngram_list = list(ngrams(tokens, n))
print("First 10 n-grams: ",ngram_list[:10])
# calculate frequecies of each unique n-gram
ngram_frequencies = FreqDist(ngram_list)
print("\nmost common n-grams: ", ngram_frequencies.most_common(10)) # ngram_frequencies.most_common(k): This method of a FreqDist object returns a list of the k most common items (n-grams) and their frequencies, sorted from most common to least common

# b. Write a function to calculate the probability of a word following a given (n − 1)gram
# P(w | w1, w2, ..., wn-1) = Count(w1, w2, ..., wn-1, w) / Count(w1, w2, ..., wn-1)
# e. Implement Laplace smoothing to handle zero probabilities for unseen n-grams
# P(w | w1, w2, ..., wn-1) = (Count(w1, w2, ..., wn-1, w) + 1) / (Count(w1, w2, ..., wn-1) + V)

# Create a more efficient n-gram transition model
def build_ngram_model(ngram_frequencies, n):
    # Create a dictionary to store next word probabilities for each context
    transition_model = defaultdict(dict)
    
    # Group n-grams by their context (first n-1 words)
    for ngram, count in ngram_frequencies.items():
        context = ngram[:-1]  # up to (n-1) context words
        next_word = ngram[-1]  # last word
        transition_model[context][next_word] = count
    
    # Convert counts to probabilities with Laplace smoothing
    MAX_CHOICES = 10  # Limit number of next word choices for efficiency
    V = len(vocabulary)  # Vocabulary size for Laplace smoothing
    
    for context in transition_model:
        words_with_counts = transition_model[context]
        total_count = sum(words_with_counts.values())
        
        # Apply Laplace smoothing: (count + 1)/(total + V)
        probs = {word: (count + 1)/(total_count + V) 
                for word, count in words_with_counts.items()}
        
        # Add small probability for unseen words
        default_prob = 1/(total_count + V)
        
        # Get top choices including smoothed probabilities
        top_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:MAX_CHOICES]
        
        # Store only the top choices with their probabilities
        transition_model[context] = dict(top_words)
        # Add default probability for unseen words
        transition_model[context]['<UNK>'] = default_prob
    
    return transition_model

# build the transition model
transition_model = build_ngram_model(ngram_frequencies, n)
print("Transition model built!")

def generate_sentence(prefix, sentence_length, n, transition_model):
    generated_sentence = list(prefix)
    
    while len(generated_sentence) < sentence_length:
        # Get the current context (last n-1 words)
        context = tuple(generated_sentence[-(n-1):])
        
        # If context not in model, back off to shorter context
        while len(context) > 0 and context not in transition_model:
            context = context[1:]
        
        # If no context found, pick a random word from vocabulary
        if not context or not transition_model[context]:
            next_word = random.choice(list(vocabulary)[:100])  # Use only common words
        else:
            # Choose next word based on pre-computed probabilities
            # Without randomness, given the same input, always get the same output
            words_probs = transition_model[context]
            next_word = random.choices(
                list(words_probs.keys()), # possible next words
                weights=list(words_probs.values()), # their probabilities
                k=1  # select one word
            )[0] #random.choices() always returns a list, [0] gets the first (and only) element from this list
        
        generated_sentence.append(next_word)
    
    # Join the words to form a sentence
    sentence_string = " ".join(generated_sentence)
    if sentence_string and sentence_string[0].islower():
        sentence_string = sentence_string[0].upper() + sentence_string[1:]
    if not sentence_string.endswith(('.', '!', '?')):
        sentence_string += '.'
    
    return sentence_string

# 3. aTest the model with different types of prefixes
print("\n=== Testing Model Generation ===")

# Test Case 1: Two-word prefixes (n-1 words, where n=3)
two_word_prefixes = [
    ['the', 'king'],
    ['she', 'was'],
    ['in', 'the'],
    ['they', 'were'],
    ['he', 'said']
]

print("\n1. Testing with two-word prefixes (optimal length):")
for prefix in two_word_prefixes:
    print(f"\nPrefix: '{' '.join(prefix)}'")
    # Generate both short and long sentences
    print("Short sentence:", generate_sentence(prefix, 8, n, transition_model))
    print("Long sentence:", generate_sentence(prefix, 15, n, transition_model))

# Test Case 2: Single-word prefixes (shorter than n-1)
single_word_prefixes = [
    ['the'],
    ['and'],
    ['but'],
    ['she'],
    ['king']
]

print("\n2. Testing with single-word prefixes (shorter than optimal):")
for prefix in single_word_prefixes:
    print(f"\nPrefix: '{' '.join(prefix)}'")
    print("Generated:", generate_sentence(prefix, 10, n, transition_model))

# Test Case 3: Common phrases as prefixes
phrase_prefixes = [
    ['once', 'upon'],
    ['long', 'ago'],
    ['dear', 'friend'],
    ['great', 'king'],
    ['young', 'man']
]

print("\n3. Testing with common phrases:")
for prefix in phrase_prefixes:
    print(f"\nPrefix: '{' '.join(prefix)}'")
    print("Generated:", generate_sentence(prefix, 12, n, transition_model))

# Test Case 4: Different sentence lengths
print("\n4. Testing different sentence lengths with same prefix:")
test_prefix = ['the', 'queen']
print(f"\nPrefix: '{' '.join(test_prefix)}'")
for length in [5, 10, 15, 20]:
    print(f"\nLength {length}:", generate_sentence(test_prefix, length, n, transition_model))

print("\n=== Testing Complete ===")

# 3. b Compute the perplexity of the model on a test set that was not used during training
def compute_perplexity(test_tokens, n, transition_model):
    import math
    # Get all n-grams from test set
    test_ngrams = list(ngrams(test_tokens, n))
    log_probability_sum = 0

    for ngram in test_ngrams:
        context = ngram[:-1] # previous n-1 words
        next_word = ngram[-1] # word to predict

        # get probability from transition model
        if context in transition_model:
            # if word exists in transition model for this context
            if next_word in transition_model[context]:
                prob = transition_model[context][next_word]
            else:
                # use <UNK> probability for unseen words
                prob = transition_model[context].get('<UNK>', 1e-10) # small default to no <unk>
        else:
            # if cotext not found, use a very small probability
            prob = 1e-10
        log_probability_sum += math.log2(prob)
    
    # Calculate perplexity
    N = len(test_ngrams)
    if N ==0:
        return float('inf') # positive infinity, can't calculate perplexity (would cause division by zero
    
    ave_log_probability = -1 * log_probability_sum / N
    perplexity = math.pow(2, ave_log_probability)
    return perplexity

# test the perplexity computation on a test set
def evaluate_model_perpexity():
    # Split the corpus into training and test sets (90-10 split)
    split_point = int(len(tokens) * 0.95)  # initially try 90% for training, 10% for testing
    training_tokens = tokens[:split_point]
    test_tokens = tokens[split_point:]

    print("\n=====Model Evaluation ===")
    print(f"Training set size: {len(training_tokens)} tokens ({(len(training_tokens)/len(tokens))*100:.1f}%)")
    print(f"Test set size: {len(test_tokens)} tokens ({(len(test_tokens)/len(tokens))*100:.1f}%)")

    # build model on training data
    training_ngrams = list(ngrams(training_tokens, n))
    training_frequencies = FreqDist(training_ngrams)
    training_model = build_ngram_model(training_frequencies, n)

    # compute perplexity on test set
    test_perplexity = compute_perplexity(test_tokens, n, training_model)
    print(f"\nModel perplexity on test set: {test_perplexity: .2f}")

    # compute perplexity on different context for coparision
    print("\nPerplexity on different contexts:")
    test_contexts = [
        tokens[split_point:split_point +100], # take the first 100 tokens right after the training/test split
        tokens[-100:], # take the last 100 tokens of the corpus
        tokens[split_point+1000:split_point+1100] # take 100 tokens from themiddle of test set
    ]

    for i, context in enumerate(test_contexts, 1):
        context_perplexity = compute_perplexity(context, n, training_model)
        print(f"context{i} perplexity: {context_perplexity: .2f}")

# evaluate the model
evaluate_model_perpexity()



