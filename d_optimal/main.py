import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def sigmoid(x):
    """Helper function to compute the sigmoid of x."""
    return 1 / (1 + np.exp(-x))

class LogisticData:
    def __init__(self, weights, seed=42):
        """
        Initializes the LogisticData class with specified weights and seed.
        
        Args:
        - weights (np.array): True weights for data generation.
        - seed (int): Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.weights = weights
    
    def generate_data(self, N):
        """
        Generates synthetic data for logistic regression.
        
        Args:
        - N (int): Number of data points.
        
        Returns:
        - tuple: Generated data and labels.
        """
        data = self.rng.standard_normal((N, len(self.weights)))
        probs = sigmoid(data @ self.weights)
        labels = (self.rng.random(N) < probs).astype(int)
        return data, labels

def fisher_matrix(difference_vector, weights):
    """
    Computes the Fisher information matrix for a single data point.
    
    Args:
    - difference_vector (np.array): Difference vector (input data point).
    - weights (np.array): Weights for the logistic model.
    
    Returns:
    - np.array: Fisher information matrix for the data point.
    """
    # YOUR CODE HERE (~2-4 lines)
    pass
    # END OF YOUR CODE

# Initialization
true_weights = np.array([-0.3356, -1.4104, 0.3144, -0.5591, 1.0426, 0.6036, -0.7549, -1.1909, 1.4779, -0.7513])
data_dim = len(true_weights)
dataset_generator = LogisticData(weights=true_weights)

# Number of iterations for sampling 500 points
num_iterations = 200

# Store covariance matrix norms for comparison
cov_norms_greedy = []
cov_norms_random = []

def greedy_fisher(data, selected_indices):
    """
    Selects the data point that maximizes the Fisher information determinant.
    
    Args:
    - data (np.array): The data matrix.
    - curr_fisher_matrix (np.array): Fisher matrix of already selected indices. NOTE: This is a global variable so you have access to it despite not being passed in as an argument!
    - selected_indices (list): List of already selected indices.
    
    Returns:
    - int: Index of the selected data point.
    """
    best_det = -np.inf
    best_index = -1
    
    # Iterate over data points to find the one maximizing Fisher determinant.
    for i, difference_vector in enumerate(data):
        # YOUR CODE HERE (~5-10 lines)
        # Make sure to skip already selected data points!
        pass
        # END OF YOUR CODE
    return best_index

def posterior_inv_cov(X, laplace_center):
    """
    Computes the posterior inverse covariance matrix using Laplace approximation.
    
    Args:
    - X (np.array): Data matrix.
    - laplace_center (np.array): Center point (weights).
    
    Returns:
    - np.array: Posterior inverse covariance matrix.
    """
    # Calculate probabilities for logistic regression model.
    probs = sigmoid(X @ laplace_center)
    W = np.diag(probs * (1 - probs))
    
    # Compute inverse covariance matrix assuming standard Gaussian prior.
    inv_cov = X.T @ W @ X + np.eye(len(true_weights))
    return inv_cov

for _ in tqdm(range(num_iterations)):
    # Generate a new sample of 500 data points
    data, _ = dataset_generator.generate_data(N=500)
    
    # Greedy selection of best 30 data points
    selected_indices = []
    curr_fisher_matrix = np.zeros((data_dim, data_dim))

    for _ in range(30):
        # Select the data point maximizing Fisher information determinant.
        best_index = greedy_fisher(data, curr_fisher_matrix, selected_indices)
        selected_indices.append(best_index)
        curr_fisher_matrix += fisher_matrix(data[best_index], true_weights)

    # Prepare greedy and random samples
    X_greedy = data[selected_indices]

    # Generate 30 random samples for comparison
    random_indices = np.random.choice(len(data), 30, replace=False)
    X_random = data[random_indices]

    # Compute posterior inverse covariance matrices for both strategies
    posterior_inv_cov_greedy = posterior_inv_cov(X_greedy, laplace_center=true_weights) 
    posterior_inv_cov_random = posterior_inv_cov(X_random, laplace_center=true_weights)

    # Calculate covariance matrices (inverse of posterior inverse covariance)
    cov_matrix_greedy = np.linalg.inv(posterior_inv_cov_greedy)
    cov_matrix_random = np.linalg.inv(posterior_inv_cov_random)

    # Measure the norm (Frobenius norm) of the covariance matrices
    cov_norm_greedy = np.linalg.norm(cov_matrix_greedy, 'fro')
    cov_norm_random = np.linalg.norm(cov_matrix_random, 'fro')

    # Store norms for analysis
    cov_norms_greedy.append(cov_norm_greedy)
    cov_norms_random.append(cov_norm_random)

# Display comparison results
print(f'Greedy mean: {np.mean(cov_norms_greedy)}')
print(f'Random mean: {np.mean(cov_norms_random)}')
print(f'Greedy win rate: {(np.array(cov_norms_greedy) < np.array(cov_norms_random)).mean()}')

# Plot the distributions of covariance matrix norms
plt.hist(cov_norms_greedy, bins=30, alpha=0.7, color='blue', label='Greedy')
plt.hist(cov_norms_random, bins=30, alpha=0.7, color='red', label='Random')
plt.xlabel('L2 Norm of Covariance Matrix')
plt.ylabel('Frequency')
plt.title('Comparison of Covariance Norms (Greedy vs. Random) Across Iterations')
plt.legend()
plt.show()
