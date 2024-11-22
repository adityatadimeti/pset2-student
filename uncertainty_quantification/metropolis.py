print("hi")
import torch
print("hi")
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from ece import expected_calibration_error

# Load training and testing data
x_train = torch.tensor(np.load('./data/differences_train.npy'))
x_test = torch.tensor(np.load('./data/differences_test.npy'))
y_train = torch.tensor(np.load('./data/labels_train.npy'))
y_test = torch.tensor(np.load('./data/labels_test.npy'))

# Likelihood function for logistic regression (per data point)
def likelihood(theta, x, y):
    """
    Computes the likelihood of the data given the logistic regression parameters.
    
    Args:
    - theta (torch.Tensor): Model parameters.
    - x (torch.Tensor): Input data.
    - y (torch.Tensor): True labels.

    Returns:
    - torch.Tensor: Likelihood values for each data point.
    """
    # YOUR CODE HERE (~3 lines)
    # Calculate logits as the linear combination of inputs and parameters.
    # Use the sigmoid function to compute the probability of the positive class.
    return torch.sigmoid(x @ theta) ** y * (1 - torch.sigmoid(x @ theta)) ** (1 - y)
    # END OF YOUR CODE

# Prior probability (theta ~ N(0, I)) - only depends on theta, not per sample
def prior(theta, sigma):
    """
    Computes the prior probability of theta under a Gaussian distribution with variance sigma^2.

    Args:
    - theta (torch.Tensor): Model parameters.
    - sigma (float): Standard deviation of the prior distribution.

    Returns:
    - torch.Tensor: Prior probability value.
    """
    # YOUR CODE HERE (~2 lines)
    # Implement Gaussian prior with zero mean and identity covariance.
    # Note that the normalization constant is not needed for Metropolis-Hastings.
    return torch.exp(-0.5 * (theta / sigma) ** 2)
    # END OF YOUR CODE

# Metropolis-Hastings sampler
def metropolis_hastings(x, y, num_samples, burn_in, tau, sigma):
    """
    Runs the Metropolis-Hastings algorithm to sample from the posterior distribution.

    Args:
    - x (torch.Tensor): Input data.
    - y (torch.Tensor): True labels.
    - num_samples (int): Total number of samples to draw.
    - burn_in (int): Number of initial samples to discard.
    - tau (float): Proposal standard deviation.
    - sigma (float): Prior standard deviation.

    Returns:
    - torch.Tensor: Collected samples post burn-in.
    - float: Acceptance ratio.
    """
    # Initialize theta (starting point of the chain) and containers for samples and acceptance count
    theta = torch.zeros(x.shape[1])
    samples = []
    acceptances = 0
    
    # Run the Metropolis-Hastings algorithm
    for t in tqdm(range(num_samples), desc="MCMC Iteration"):
        # YOUR CODE HERE (~12-16 lines)
        # 1. Propose new theta from the proposal distribution (e.g., Gaussian around current theta).
        # 2. Compute prior and likelihood for current and proposed theta
        # 3. Calculate the acceptance ratio as the product of likelihood and prior ratios.
        # 4. Accept or reject the proposal based on the acceptance probability.
        # 5. Store the sample after the burn-in period

        theta_proposal = theta + torch.normal(0, tau, size=theta.shape)

        current_likelihoods = likelihood(theta, x, y)
        proposed_likelihoods = likelihood(theta_proposal, x, y)
        current_prior = torch.prod(prior(theta, sigma))
        proposed_prior = torch.prod(prior(theta_proposal, sigma))

        likelihood_ratio = torch.prod(proposed_likelihoods / current_likelihoods)
        prior_ratio = proposed_prior / current_prior

        acceptance_ratio = likelihood_ratio * prior_ratio
        if torch.rand(1) < acceptance_ratio:
            theta = theta_proposal
            acceptances += 1
        if t >= burn_in:
            samples.append(theta)

        # END OF YOUR CODE
    
    return torch.stack(samples), acceptances / num_samples

# Run Metropolis-Hastings on training data
num_samples = 10000
burn_in = 1000
tau = 0.01  # Proposal variance (tune this for convergence)
sigma = 2.0  # Prior variance

# Collect samples and compute acceptance ratio
samples, acceptance_ratio = metropolis_hastings(x_train, y_train, num_samples=num_samples, burn_in=burn_in, tau=tau, sigma=sigma)
averaged_weights = samples.mean(axis=0)
print(f'Predicted weights: {averaged_weights}')
print(f'Acceptance Ratio: {acceptance_ratio}')

# Evaluate accuracy on training set
train_predictions = (x_train @ averaged_weights > 0).float()
train_acc = (train_predictions == y_train).float().mean()
print(f'Train Accuracy: {train_acc}')

# Evaluate accuracy on testing set
test_predictions = (x_test @ averaged_weights > 0).float()
acc = (test_predictions == y_test).float().mean()
print(f'Test Accuracy: {acc}')

# Compute expected calibration error on testing set
expected_calibration_error(torch.sigmoid(x_test @ averaged_weights).numpy(), y_test.numpy(), model_name="Metropolis-Hastings")
