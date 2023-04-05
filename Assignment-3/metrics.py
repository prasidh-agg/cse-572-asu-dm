# Import necessary libraries
import numpy as np

# This function calculates the purity of clustering results
def calculatePurity(labels, fit_transform_data):
    purity = 0
    for label in np.unique(labels):  # Iterate through unique labels
        label_points = np.where(labels == label)
        local_purity = 0
        count = 0
        unique, count = np.unique(
            fit_transform_data[label_points], return_counts=True)

        for index in range(0, unique.shape[0]):
            exp = count[index] / float(len(label_points[0]))
            if exp > local_purity:
                local_purity = exp
        purity += local_purity * (len(label_points[0]) / float(len(labels)))

    return purity

# This function calculates the entropy of clustering results
def calculateEntropy(labels, fit_transform_data):
    Entropy = 0
    for label in np.unique(labels):  # Iterate through unique labels
        # Find the indices of points with the current label
        label_points = np.where(labels == label)
        local = 0
        # Count occurrences of each unique value
        unique, count = np.unique(
            fit_transform_data[label_points], return_counts=True)

        # Iterate through unique values and calculate their probability
        for index in range(0, unique.shape[0]):
            # Calculate probability
            exp = count[index] / float(len(label_points[0]))
            # Calculate entropy contribution for the current value
            local += -1*exp*np.log(exp)
        # Add weighted entropy to the total entropy
        Entropy += local * (len(label_points[0]) / float(len(labels)))

    return Entropy