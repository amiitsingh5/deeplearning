from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample sentences for training
sentences = [
    "Word embeddings are important in natural language processing.",
    "They help capture semantic relationships between words.",
    "Word2Vec is a popular algorithm for generating embeddings.",
]

# Tokenize the sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, sg=1, min_count=1)

# Access the vocabulary and count the number of words
num_words = len(model.wv.key_to_index)

print(f"Number of words in the trained Word2Vec model: {num_words}")
