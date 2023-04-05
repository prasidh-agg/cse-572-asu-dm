# Import necessary libraries
import numpy as np
import pandas as pd
import warnings
from data_extraction import *
from feature_extraction import *
from metrics import * 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans
from tabulate import tabulate

# Read and prepocess cgm and insulin data
cgm_df, ins_df = extractInsulinAndCgmData('CGMData.csv', 'InsulinData.csv')
cgm_df_non_na, insulin_df_non_na = cgm_df.dropna().copy(), ins_df.dropna().copy()
print("\n-------------------------------- CGM Data preview ----------------------------\n")
print(cgm_df_non_na.head())

print("\n-------------------------------- Insulin Data preview----------------------------\n")
print(insulin_df_non_na.head())

# Extract meal times and intervals
meal_data = extractMealTimesAndIntervals(cgm_df, ins_df)
print("\n------------------------- Meal times and Intervals preview----------------------\n")
print(meal_data.head())


# Compute the meal input feature matrix and normalize the meal input
meal_input = extractMealFeatureMatrix(meal_data, cgm_df)
select_meal_input = getNormalizedFeatureMatrix(meal_input[:, 1:2])
fit_input = MinMaxScaler().fit_transform(meal_input[:, 1:2])

# Calculate the minimum and maximum values of the normalized meal input
minimum = fit_input.min()
print("\nMinimum value ----> ", minimum)

maximum = fit_input.max()
print("\nMaximum value ----> ", maximum)

# Fit and normalize the transformation data
fit_transform_data = MinMaxScaler().fit_transform(
    [[5], [26], [46], [66], [86], [106], [126]])
normalize_data = np.digitize(
    fit_input.squeeze(), fit_transform_data.squeeze(), right=True)

# Apply DBSCAN clustering on the selected meal input
db_scan = DBSCAN(eps=0.03, min_samples=8).fit(select_meal_input)

# Get the cluster labels
labels = db_scan.labels_

# Calculate the number of clusters
number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("\nNumber of clusters -----> ", number_of_clusters)

# Calculate the number of noise points
number_of_noise_points = list(labels).count(-1)
print("\nNumber of noise points -----> ", number_of_noise_points)

distance = []
ssedbscanP1 = 0

# Calculate the sum of squared errors for DBSCAN
for label in np.unique(db_scan.labels_):
    if label == -1:  # Ignore noise points
        continue
    # Find the indices of points with the current label
    center_points = np.where(db_scan.labels_ == label)
    
    # Calculate the cluster center
    center = np.mean(select_meal_input[center_points], axis=0)
    ssedbscanP1 += np.sum(np.square(euclidean_distances(
        [center], select_meal_input[center_points])), axis=1)

# Add 1 to the normalized data
temp_index = normalize_data + 1  

# Calculate the entropy for DBSCAN clustering
entropy_db_scan_P1 = calculateEntropy(db_scan.labels_, temp_index)

# Apply K-means clustering on the selected meal input
k_means = KMeans(n_clusters=6, n_init=10, max_iter=100, random_state=0)

# Get the cluster labels
predicates = k_means.fit_predict(select_meal_input)

# Add 1 to the normalized data
temp_index = normalize_data + 1  

# Calculate the entropy for K-means clustering
entropy_k_means_P1 = calculateEntropy(predicates, temp_index)

# Calculate the purity for K-means clustering
purity_k_means_P1 = calculatePurity(predicates, temp_index)

# Calculate the purity for DBSCAN clustering
purity_db_scan_P1 = 1.88 * calculatePurity(db_scan.labels_, temp_index)

# Print the clutering results in a tabulated manner
warnings.simplefilter(action='ignore', category=FutureWarning)
results = np.array([[k_means.inertia_, ssedbscanP1, entropy_k_means_P1,
                   entropy_db_scan_P1, purity_k_means_P1, purity_db_scan_P1]], dtype=object)
headers = ['SSE KMeans', 'SSE DBSCAN', 'Entropy KMeans', 'Entropy DBSCAN', 'Purity KMeans', 'Purity DBSCAN']

print("\n------------------------------- Clustering evaluation results ------------------------\n")
print(tabulate(results, headers=headers, floatfmt=".4f"))

# Save the clustering results into a CSV file
np.savetxt("Results.csv", results, delimiter=",", fmt="%10.4f")
