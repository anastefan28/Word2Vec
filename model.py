import numpy as np

def init_vectors(vocab_size, dim):
    input_vectors = np.random.uniform(-0.5 / dim, 0.5 / dim, (vocab_size, dim))
    output_vectors = np.zeros((vocab_size, dim), dtype=float)

    return input_vectors, output_vectors

def forward(centre_word_id, context_word_id, negative_word_ids, input_vectors, output_vectors):
    centre = input_vectors[centre_word_id]
    context = output_vectors[context_word_id]
    negatives = output_vectors[negative_word_ids]
    positive_score = np.dot(context, centre)
    negative_scores = np.matmul(negatives, centre)

    return centre, context, negatives, positive_score, negative_scores


def sigmoid(x):
    res = np.zeros_like(x)
    positive = x >= 0
    res[positive] = 1 / (1 + np.exp(-x[positive]))
    negative = x < 0
    res[negative] = np.exp(x[negative]) / (1 + np.exp(x[negative]))
    return res

def compute_loss(positive_score, negative_scores):
    positive_loss = np.log(sigmoid(positive_score) + 1e-10)
    negative_loss = np.sum(np.log(sigmoid(-negative_scores) + 1e-10))
    total_loss = -(positive_loss + negative_loss)

    return total_loss


def compute_gradients(centre, context, negatives, positive_score, negative_scores):
    context_error = sigmoid(positive_score) - 1
    negative_error = sigmoid(negative_scores)
    grad_context = context_error * centre
    grad_negatives = negative_error.reshape(-1, 1) * centre
    grad_centre = context_error * context + np.matmul(negative_error, negatives)

    return grad_centre, grad_context, grad_negatives

def update_parameters(centre_word_id, context_word_id, negative_word_ids,grad_centre, grad_context, grad_negatives,
                      input_vectors, output_vectors, learning_rate):
    input_vectors[centre_word_id] -= learning_rate * grad_centre
    output_vectors[context_word_id] -= learning_rate * grad_context
    output_vectors[negative_word_ids] -= learning_rate * grad_negatives

    return input_vectors, output_vectors