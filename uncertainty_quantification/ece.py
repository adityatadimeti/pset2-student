import numpy as np
import matplotlib.pyplot as plt

def expected_calibration_error(probs, labels, model_name: str, n_bins=20, n_ticks=10, plot=True):
    """
    Computes the Expected Calibration Error (ECE) for a model and plots a refined reliability diagram
    with confidence histogram and additional calibration statistics.
    
    Args:
    - probs (np.array): Array of predicted probabilities for the positive class (for binary classification).
    - labels (np.array): Array of true labels (0 or 1).
    - model_name (str): Name of the model for labeling the plot.
    - n_bins (int): Number of bins to divide the probability interval [0,1] into.
    - n_ticks (int): Number of ticks to show along the x-axis.
    - plot (bool): If True, generates the reliability plot; otherwise, only computes ECE.

    Returns:
    - float: Computed ECE value.
    """
    
    # Ensure probabilities are in the range [0, 1]
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be in the range [0, 1]"
    
    # Initialize bin edges, centers, and storage for accuracy, confidence, and counts
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = 1.0 / n_bins

    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    # Populate bin statistics: accuracy, confidence, and count
    # YOUR CODE HERE (~7 lines)
    # Loop over each bin and:
    # - Find indices of probabilities that fall within the bin.
    # - Count the number of items in the bin.
    # - Calculate the accuracy (average of true labels) within the bin.
    # - Calculate the confidence (average of predicted probabilities) within the bin.
    for bin in range(n_bins):
        next_bin = bin + 1
        bindices = np.where((probs < bin_edges[next_bin]) & (probs >= bin_edges[bin]))[0]
        bin_size = len(bindices)
        bin_counts[bin] = bin_size
        accs[bin] = np.mean(labels[bindices])
        confs[bin] = np.mean(probs[bindices])
    # END OF YOUR CODE
    
    # Compute ECE: weighted average of |accuracy - confidence| across bins
    # YOUR CODE HERE (1 line)
    # - Use the bin counts to calculate a weighted average of the differences between accuracy and confidence.
    ece_value = np.sum(bin_counts / len(labels) * np.abs(accs - confs))
    # END OF YOUR CODE
    
    # Return only ECE if plot is not required
    if not plot:
        return ece_value

    # Compute average confidence and accuracy for reference lines
    avg_confidence = np.mean(probs)
    avg_accuracy = np.mean(labels)
    
    # Create reliability diagram and histogram
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 10))
    
    # Reliability diagram (top plot)
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    for i in range(n_bins):
        # Draw the gap bar starting from the diagonal line (perfect calibration)
        ax1.bar(bin_centers[i], abs(accs[i] - confs[i]), width=bar_width, bottom=min(accs[i], confs[i]), 
                color='red', alpha=0.3, label='Accuracy-Confidence Gap' if i == 0 else "")
        # Draw the accuracy bar as a small black line on top of the gap bar
        ax1.plot([bin_centers[i] - bar_width / 2, bin_centers[i] + bar_width / 2], 
                 [accs[i], accs[i]], color='black', linewidth=2)

    # Add a black line as a sample for accuracy in the legend
    ax1.plot([], [], color='black', linewidth=2, label='Accuracy Marker')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{model_name}\nECE={ece_value:.2f}')
    ax1.legend()

    # Set tick marks based on `n_ticks` evenly spaced along the x-axis
    tick_positions = np.linspace(0, 1, n_ticks + 1)
    ax1.set_xticks(tick_positions)
    ax2.set_xticks(tick_positions)
    ax1.set_xticklabels([f'{x:.2f}' for x in tick_positions])
    ax2.set_xticklabels([f'{x:.2f}' for x in tick_positions])

    # Confidence histogram with average markers
    ax2.bar(bin_centers, bin_counts, width=bar_width, color='blue', alpha=0.6)
    ax2.axvline(x=avg_confidence, color='gray', linestyle='--', linewidth=2, label='Avg. confidence')
    ax2.axvline(x=avg_accuracy, color='black', linestyle='-', linewidth=2, label='Avg. accuracy')
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    
    return ece_value

if __name__ == "__main__":
    # Test with random probabilities and labels
    probs = np.random.rand(10000)  # Random probabilities between 0 and 1
    labels = np.random.binomial(1, (probs + 1) / 2)

    # Run the function and display the result
    ece_value = expected_calibration_error(probs, labels, "Test Model", plot=True)
    print(f"ECE Value: {ece_value}")
