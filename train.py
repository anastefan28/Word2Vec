import numpy as np
from model import forward, compute_loss, compute_gradients, update_parameters

def train(pairs, input_vectors, output_vectors, noise_dist, vocab_size, lr, epochs, k_negatives):
    total_steps = len(pairs) * epochs
    current_step = 0
    for epoch in range(epochs):
        np.random.shuffle(pairs)
        epoch_loss = 0

        for current_word, neighbour in pairs:
            negative_ids = np.random.choice(vocab_size, size=k_negatives, replace=True, p=noise_dist)
            centre_vector, positive_vector, negative_vectors, positive_score, negative_scores = forward(current_word, 
                                                                                                neighbour, negative_ids, input_vectors, output_vectors)
            loss = compute_loss(positive_score, negative_scores)
            epoch_loss += loss
            
            grad_centre, grad_positive, grad_negatives = compute_gradients(centre_vector, positive_vector, 
                                                                           negative_vectors,positive_score, negative_scores)
            input_vectors, output_vectors = update_parameters(current_word, neighbour, negative_ids,grad_centre, grad_positive, grad_negatives,input_vectors, output_vectors, lr)
            current_step += 1
            lr = max(0.025 * (1 - current_step / total_steps), 0.0001)

        avg_loss = epoch_loss / len(pairs)
        print(f"Epoch {epoch}/{epochs} — avg loss: {avg_loss:.4f} — lr: {lr:.6f}")
    
    return input_vectors, output_vectors
