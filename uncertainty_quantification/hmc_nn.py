# Use a GPU when running this file! JAX should automatically default to GPU.
import jax.numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
from ece import expected_calibration_error

# DO NOT CHANGE! This function can be ignored.
def set_numpyro(new_sampler):
    numpyro.sample = new_sampler

# Define the neural network model with one hidden layer
def nn_model(x_data, y_data, hidden_dim=10):
    """
    Defines a Bayesian neural network with one hidden layer.

    Args:
    - x_data (np.array): Input data.
    - y_data (np.array): Target labels.
    - hidden_dim (int): Number of units in the hidden layer.

    Returns:
    - hidden_activations: Activations from the hidden layer.
    - logits: Logits for the output layer.
    """
    input_dim = x_data.shape[1]
    
    # Prior over the weights and biases for the hidden layer
    w_hidden = numpyro.sample('w_hidden', dist.Normal(np.zeros((input_dim, hidden_dim)), np.ones((input_dim, hidden_dim))))
    b_hidden = numpyro.sample('b_hidden', dist.Normal(np.zeros(hidden_dim), np.ones(hidden_dim)))
    
    # Compute the hidden layer activations using ReLU
    # YOUR CODE HERE (~1 line)
    # Implement the hidden layer computation, applying a ReLU activation.
    hidden_activations = np.maximum(0, x_data @ w_hidden + b_hidden)
    # END OF YOUR CODE 
    
    # Prior over the weights and biases for the output layer
    w_output = numpyro.sample('w_output', dist.Normal(np.zeros(hidden_dim), np.ones(hidden_dim)))
    b_output = numpyro.sample('b_output', dist.Normal(0, 1))
    
    # Compute the logits for the output layer
    # YOUR CODE HERE (~1 line)
    # Calculate the logits as the linear combination of hidden activations and output layer weights.
    logits = hidden_activations @ w_output + b_output
    # END OF YOUR CODE

    # Likelihood (Bernoulli likelihood with logits)
    numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y_data)
    return hidden_activations, logits

def sigmoid(x):
    """Helper function to compute the sigmoid of x."""
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    # Load training and testing data
    x_train = np.load('./data/differences_train.npy')
    x_test = np.load('./data/differences_test.npy')
    y_train = np.load('./data/labels_train.npy')
    y_test = np.load('./data/labels_test.npy')

    # HMC Sampler Configuration
    hmc_kernel = NUTS(nn_model)

    # Running HMC with the MCMC interface in NumPyro
    num_samples = 200  # Number of samples
    warmup_steps = 100  # Number of burn-in steps
    rng_key = random.PRNGKey(0)  # Random seed

    # MCMC object with HMC kernel
    mcmc = MCMC(hmc_kernel, num_samples=num_samples, num_warmup=warmup_steps)
    mcmc.run(rng_key, x_train, y_train)

    # Get the sampled weights (theta samples)
    samples = mcmc.get_samples()

    # Extract the weight samples
    w_hidden_samples = samples['w_hidden']
    b_hidden_samples = samples['b_hidden']
    w_output_samples = samples['w_output']
    b_output_samples = samples['b_output']

    # Compute the averaged weights and biases
    w_hidden_mean = np.mean(w_hidden_samples, axis=0)
    b_hidden_mean = np.mean(b_hidden_samples, axis=0)
    w_output_mean = np.mean(w_output_samples, axis=0)
    b_output_mean = np.mean(b_output_samples, axis=0)

    # Forward pass through the network for testing set
    # YOUR CODE HERE (~2 lines)
    # Compute hidden layer activations and logits for the test set using the mean weights and biases.
    hidden, test_logits = nn_model(x_test, y_test, hidden_dim=10)
    # END OF YOUR CODE
    test_predictions = test_logits > 0
    test_accuracy = np.mean(test_predictions == y_test)
    print(f'Test Accuracy: {test_accuracy}')

    # Forward pass through the network for training set
    # YOUR CODE HERE (~2 lines)
    # Compute hidden layer activations and logits for the training set.
    hidden, train_logits = nn_model(x_train, y_train, hidden_dim=10)
    # END OF YOUR CODE
    train_predictions = train_logits > 0
    train_accuracy = np.mean(train_predictions == y_train)
    print(f'Train Accuracy: {train_accuracy}')

    # Compute expected calibration error on testing set
    expected_calibration_error(sigmoid(test_logits), y_test, model_name="HMC")
