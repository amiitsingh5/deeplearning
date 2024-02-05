from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample sentences for training

sentences = ["My name is Amit and i like birds"]

"""
sentences = [
    "Word embeddings are important in natural language processing.",
    "They help capture semantic relationships between words.",
    "Word2Vec is a popular algorithm for generating embeddings.",
]
"""
# Tokenize the sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
print("1. tokenized_sentences : ", tokenized_sentences)
# Train Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=1000, window=5, sg=1, min_count=1)
# sg=0 specifies CBOW,
print("2. model :", model)
# Access word embeddings
word_embedding = model.wv['name'] #retriving word from trained model
# retrieving the 'word embedding vector' for the word 'word' from the trained Word2Vec model.
print(word_embedding)
