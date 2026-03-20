import numpy as np

def get_embedding(word, input_vectors, output_vectors, word2id):
    idx = word2id[word]
    vector = input_vectors[idx] + output_vectors[idx]
    return vector

def most_similar(word, input_vectors, output_vectors, word2id, id2word, topn=5):
    query = get_embedding(word, input_vectors, output_vectors, word2id)
    query = query / (np.linalg.norm(query) + 1e-10)
    similarities = []
    for idx, other_word in enumerate(id2word):
        if other_word == word:
            continue
        vector = get_embedding(other_word, input_vectors, output_vectors, word2id)
        vector = vector / (np.linalg.norm(vector) + 1e-10)
        score = np.dot(query, vector)
        similarities.append((other_word, score))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:topn]