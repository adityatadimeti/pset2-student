import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

class LogisticActiveLearning:
    def __init__(self, test_size=0.2):
        """
        Initializes LogisticActiveLearning model, sets device, and prepares data.
        
        Args:
        - test_size (float): Proportion of the dataset used for validation.
        """
        # Make device customizable
        self.device = torch.device("cpu")
        X, y = make_classification(n_samples=10000, random_state=42)

        # Convert data and labels to tensors
        x_data = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_data = torch.tensor(y, dtype=torch.float32).to(self.device)
        self.N, self.D = x_data.shape

        # Split into training and validation sets
        train_indices, val_indices = train_test_split(range(self.N), test_size=test_size, random_state=42)
        self.x_train = x_data[train_indices]
        self.y_train = y_data[train_indices]
        self.x_val = x_data[val_indices]
        self.y_val = y_data[val_indices]

        # Initialize mean and inverse covariance for the prior
        self.weights_mean = torch.zeros(self.D, requires_grad=True, device=self.device)
        self.weights_inv_cov = torch.eye(self.D).to(self.device)  # Start with identity inverse covariance

    def negative_log_posterior(self, w, x, y):
        """
        Computes the negative log-posterior (negative log-prior + log-likelihood).
        
        Args:
        - w (torch.Tensor): Model weights.
        - x (torch.Tensor): Input data point.
        - y (torch.Tensor): True label.
        
        Returns:
        - torch.Tensor: Negative log-posterior value.
        """
        # YOUR CODE HERE (~4-6 lines)
        # Compute log-prior term using inverse covariance
        pass
        # END OF YOUR CODE

    def optimize_weights(self, w, x, y, num_steps=50, lr=1e-2):
        """
        Optimizes weights using Adam optimizer.
        
        Args:
        - w (torch.Tensor): Initial weights.
        - x (torch.Tensor): Input data point.
        - y (torch.Tensor): True label.
        - num_steps (int): Number of optimization steps.
        - lr (float): Learning rate.
        
        Returns:
        - torch.Tensor: Updated weights.
        - torch.Tensor: Hessian inverse covariance.
        """
        optimizer = Adam([w], lr=lr)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            loss = self.negative_log_posterior(w, x, y)
            loss.backward()
            optimizer.step()

        # Compute the Hessian of log-posterior, serving as inverse covariance
        inv_cov = self.compute_hessian(w.detach(), x, y)
        return w.detach().clone(), inv_cov

    def compute_hessian(self, w, x, y):
        """
        Computes the Hessian of the negative log-posterior, used as the inverse covariance.
        
        Args:
        - w (torch.Tensor): Model weights.
        - x (torch.Tensor): Input data point.
        - y (torch.Tensor): True label.
        
        Returns:
        - torch.Tensor: Hessian of the negative log-posterior.
        """
        # YOUR CODE HERE (~5-8 lines)
        # Hessian of the prior term
        pass
        # END OF YOUR CODE

    def acquisition_fn(self, x):
        """
        Computes posterior means and inverse covariances for y=1 and y=0 without modifying original parameters.
        
        Args:
        - x (torch.Tensor): Input data point.
        
        Returns:
        - dict: Posterior properties for y=1 and y=0 cases.
        """
        weights_y1 = self.weights_mean.clone().detach().requires_grad_(True)
        weights_y0 = self.weights_mean.clone().detach().requires_grad_(True)

        # Optimize weights and get Hessian for both y=1 and y=0 cases
        posterior_mean_y1, inv_cov_y1 = self.optimize_weights(weights_y1, x, 1, num_steps=50)
        posterior_mean_y0, inv_cov_y0 = self.optimize_weights(weights_y0, x, 0, num_steps=50)

        # Calculate probabilities for the acquisition function
        prob_y1 = torch.sigmoid(torch.dot(self.weights_mean.detach(), x))
        prob_y0 = 1 - prob_y1

        return {
            'prob_y1': prob_y1,
            'prob_y0': prob_y0,
            'posterior_mean_y1': posterior_mean_y1,
            'posterior_inv_cov_y1': inv_cov_y1,
            'posterior_mean_y0': posterior_mean_y0,
            'posterior_inv_cov_y0': inv_cov_y0
        }

    def expected_information_gain(self, x):
        """
        Computes expected information gain for a given point `x`.
        
        Args:
        - x (torch.Tensor): Input data point.
        
        Returns:
        - torch.Tensor: Expected Information Gain (EIG) value.
        """
        acquisition = self.acquisition_fn(x)

        # Compute KL divergences for y=1 and y=0 using inverse covariances
        kl_y1 = kl_divergence_gaussians(
            acquisition['posterior_mean_y1'],
            acquisition['posterior_inv_cov_y1'],
            self.weights_mean.detach(),
            self.weights_inv_cov
        )

        kl_y0 = kl_divergence_gaussians(
            acquisition['posterior_mean_y0'],
            acquisition['posterior_inv_cov_y0'],
            self.weights_mean.detach(),
            self.weights_inv_cov
        )

        # Expected Information Gain (EIG)
        eig = None # YOUR CODE HERE (1 line)
        return eig

    def active_learning(self, selected_indices, subset_size=50):
        """
        Active learning loop that selects the most informative data point based on EIG.
        
        Args:
        - selected_indices (list): Indices of previously selected samples.
        - subset_size (int): Number of samples to consider in each subset.

        Returns:
        - best_x, best_x_idx, best_acquisition: Selected data point and acquisition details.
        """
        best_eig = -float('inf')
        best_x = None
        best_x_idx = -1
        best_acquisition = None

        subset_indices = [i for i in torch.randperm(len(self.x_train)).tolist() if i not in selected_indices][:subset_size]

        # YOUR CODE HERE (~ 10 lines)
        pass
        # END OF YOUR CODE
        return best_x, best_x_idx, best_acquisition

    def validate(self):
        """
        Computes accuracy on the validation set by predicting labels and comparing to true labels.
        
        Returns:
        - float: Validation accuracy.
        """
        with torch.no_grad():
            logits = self.x_val @ self.weights_mean
            predictions = torch.sigmoid(logits) >= 0.5  # Convert logits to binary predictions
            accuracy = (predictions == self.y_val).float().mean().item()
            print(f"Validation accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def train(self, num_iterations=10, subset_size=50):
        """
        Train the model using active learning with subset sampling.
        
        Args:
        - num_iterations (int): Number of active learning iterations.
        - subset_size (int): Number of samples to consider in each subset.
        """
        selected_indices = []
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")

            # Select the most informative data point from a random subset
            best_x, best_x_idx, acquisition = self.active_learning(selected_indices, subset_size=subset_size)
            selected_indices.append(best_x_idx)
            print(f"Selected data point with EIG.")

            # Get the true label for the selected data point
            y = self.y_train[best_x_idx].item()

            # Update posterior mean and inverse covariance based on true label
            if y == 1:
                self.weights_mean = acquisition['posterior_mean_y1']
                self.weights_inv_cov = acquisition['posterior_inv_cov_y1']
            else:
                self.weights_mean = acquisition['posterior_mean_y0']
                self.weights_inv_cov = acquisition['posterior_inv_cov_y0']

            print(f"Covariance L2: {torch.inverse(self.weights_inv_cov).norm()}")

            # Validate model performance on the validation set
            self.validate()

# KL divergence between two multivariate normal distributions
def kl_divergence_gaussians(mu1, sigma1_inv, mu2, sigma2_inv):
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Args:
    - mu1, mu2 (torch.Tensor): Mean vectors of the distributions.
    - sigma1_inv, sigma2_inv (torch.Tensor): Inverse covariance matrices of the distributions. PLEASE NOTE THE INVERSE!
    
    Returns:
    - torch.Tensor: KL divergence value.
    """
    # YOUR CODE HERE (~ 9-12 lines)
    pass
    # END OF YOUR CODE

# Example usage
model = LogisticActiveLearning()
model.train(num_iterations=100, subset_size=50)
