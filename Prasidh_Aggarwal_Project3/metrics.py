# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.cluster import contingency_matrix

def kMeansMetrics(scaledMealFeatureMatrix, mealDataMatrix):
    # Creating a KMeans object with 7 clusters, 20 iterations for the initialization, and a maximum of 100 iterations for each run
    kMeans = KMeans(n_clusters=7, n_init=20, max_iter=100)

    # Fitting the model to the scaledMealFeatureMatrix data
    kMeans.fit(scaledMealFeatureMatrix)

    # Getting the cluster labels for each data point
    labels = kMeans.labels_
    clusterCenters = kMeans.cluster_centers_

    # Getting the ground truth labels from the mealDataMatrix
    ground_truth = mealDataMatrix['label']
    print("\n------------------------- Bin Matrix ----------------------\n")
    print(ground_truth)

    # Calculating the contingency matrix for the predicted labels and ground truth labels
    contigencyMatrix = contingency_matrix(ground_truth, labels)

    # Getting the number of samples in the mealDataMatrix
    num_samples = len(mealDataMatrix)

    # Calculating the Sum of Squared Errors (SSE) for KMeans
    kmeans_sse = kMeans.inertia_

    # Calculating the purity for KMeans
    kmeans_purity = np.sum(np.amax(contigencyMatrix, axis=0)) / num_samples

    # Calculating the entropy for KMeans
    kmeans_entropy = -np.sum((np.sum(contigencyMatrix, axis=1) / num_samples) *
                    np.log2(np.sum(contigencyMatrix, axis=1) / num_samples))

    # Returning the SSE, purity, and entropy for KMeans
    return kmeans_sse, kmeans_purity, kmeans_entropy

    
def dbScanMetrics(scaledMealFeatureMatrix, mealDataMatrix):
    # Creating a DBSCAN object with an epsilon of 1.4 and a minimum of 4 samples per cluster
    dbScan = DBSCAN(eps=1.4, min_samples=4).fit(scaledMealFeatureMatrix)
    
    # Getting the predicted labels for each data point
    dbScanLabels = dbScan.labels_.astype(float)

    # Changing the value of -1 (noise points) to NaN
    dbScanLabels[dbScanLabels == -1] = np.nan

    # Getting the ground truth labels from the mealDataMatrix
    ground_truth = mealDataMatrix['label']
    

    # Calculating the contingency matrix for the predicted labels and ground truth labels
    contigencyMatrix = pd.crosstab(ground_truth, dbScanLabels, dropna = False)

    # Getting the number of samples in the mealDataMatrix
    num_samples = len(mealDataMatrix)

    # Changing the NaN values back to -1
    dbScanLabels = np.nan_to_num(dbScanLabels, nan=-1)

    # Adding the predicted labels to the scaledMealFeatureMatrix
    scaledMealFeatureMatrix['label'] = dbScanLabels

    # Calculating the SSE for each cluster (excluding noise points)
    dbscan_sse = scaledMealFeatureMatrix[scaledMealFeatureMatrix['label'] != -1].groupby(['label']).apply(lambda x: ((x.iloc[:,:-1] - x.iloc[:,:-1].mean()) ** 2).sum().sum()).sum()

    # Calculate the purity for DBSCAN
    dbscan_purity = np.sum(np.amax(contigencyMatrix, axis=0)) / num_samples

    # Calculate the entropy for DBSCAN
    dbscan_entropy = -np.sum((np.sum(contigencyMatrix, axis=1) / num_samples) *
                    np.log2(np.sum(contigencyMatrix, axis=1) / num_samples))

    return dbscan_sse, dbscan_purity, dbscan_entropy
