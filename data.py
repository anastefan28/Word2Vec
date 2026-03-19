import re
from collections import Counter
import numpy as np

def tokenise(text):
    return re.findall(r"[a-z]+", text.lower())

def build_vocab(tokens, min_count=5):
    counts = Counter(tokens)
    id2word = []
    for word, count in counts.most_common():
        if count >= min_count:
            id2word.append(word)
    word2id = {}
    for index, word in enumerate(id2word):
        word2id[word] = index
    
    return word2id, id2word

def build_noise_distribution(id2word, counts):
    freq = []
    for word in id2word:
        freq.append(counts[word])
    
    freq = np.array(freq, dtype=float)
    freq_powered = freq ** 0.75
    noise_dist = freq_powered / freq_powered.sum()
    return noise_dist

def generate_skipgram_pairs(corpus_ids, window):
    result = []
    for i in range(len(corpus_ids)):
        current_word = corpus_ids[i]
        start = i - window
        end = i + window + 1
        for j in range(start, end):
            if j < 0 or j >= len(corpus_ids):
                continue
            if j == i:
                continue
            neighbour = corpus_ids[j]
            result.append((current_word, neighbour))
    
    return result

