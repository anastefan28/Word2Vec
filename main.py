from collections import Counter

import numpy as np
from eval import most_similar
from train import train
from model import init_vectors
from data import tokenise, build_vocab, build_noise_distribution, generate_skipgram_pairs

with open("shakespeare.txt") as f:
    text = f.read()

tokens = tokenise(text)
word2id, id2word = build_vocab(tokens, min_count=5)

print(f"Total tokens : {len(tokens)}")
print(f"Vocabulary size: {len(word2id)}")
print(f"Top 10 words {id2word[:10]}")
# for (word,id) in word2id.items():
#     print(f"{word}: {id}")
# for (id,word) in enumerate(id2word):
#     print(f"{id}: {word}")

    
counts = Counter(tokens)
noise_dist = build_noise_distribution(id2word, counts)

for i, word in enumerate(id2word[:10]):
    print(f"{word} {counts[word]} {noise_dist[i]:.4f}")

embed_dim = 50
vocab_size = len(word2id)

corpus_ids = [word2id[w] for w in tokens if w in word2id]
pairs = generate_skipgram_pairs(corpus_ids, window=2)

print(f"Total training pairs: {len(pairs)}")
print("First 10 pairs:")
for word, neighbour in pairs[:10]:
    print(f"({id2word[word]}, {id2word[neighbour]})")

input_vectors, output_vectors = init_vectors(vocab_size, embed_dim)
  
# print(f"{input_vectors[0][:5]}")   
# print(f"{output_vectors[0][:5]}")   

pairs_array = np.array(pairs)
input_vectors, output_vectors = train(pairs_array, input_vectors, output_vectors, noise_dist, vocab_size, lr=0.025, epochs=10, k_negatives=15)

test_words = ["king", "love", "man", "war", "god"]

with open("results.txt", "w") as f:
    for word in test_words:
        if word in word2id:
            neighbours = most_similar(word, input_vectors, output_vectors, word2id, id2word, topn=5)
            f.write(f"\n'{word}' most similar:\n")
            for neighbour, score in neighbours:
                f.write(f"  {neighbour} {score}\n")