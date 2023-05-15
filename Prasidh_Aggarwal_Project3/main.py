# Import necessary libraries
import numpy as np
import pandas as pd
import warnings
from data_extraction import *
from feature_extraction import *
from metrics import * 
from sklearn.preprocessing import StandardScaler

from tabulate import tabulate

# Read and prepocess Insulin and CGM data
insulin_df, cgm_df = extractInsulinAndCgmData('InsulinData.csv', 'CGMData.csv')
insulin_df_non_na, cgm_df_non_na = insulin_df.dropna().copy(), cgm_df.dropna().copy()

print("\n-------------------------------- Insulin Data preview ----------------------------\n")
print(insulin_df_non_na.head())


print("\n-------------------------------- CGM Data preview ----------------------------\n")
print(cgm_df_non_na.head())

# Extract meal times and intervals
meal_df = extractMealStartTimes(insulin_df, cgm_df)
print("\n------------------------- Meal times preview ----------------------\n")
print(meal_df.head())

mealDataMatrix = computeMealDataMatrix(cgm_df, meal_df)
print("\n------------------------- Meal data matrix preview ----------------------\n")
print(mealDataMatrix)

mealFeatureMatrix = computeMealFeatureMatrix(mealDataMatrix)
print("\n------------------------- Meal feature matrix preview ----------------------\n")
print(mealFeatureMatrix)

scaler = StandardScaler()
scaledMealFeatureMatrix = pd.DataFrame(scaler.fit_transform(mealFeatureMatrix))


kMeans_sse, kMeans_purity, kMeans_entropy = kMeansMetrics(scaledMealFeatureMatrix, mealDataMatrix)
dbScan_sse, dbScan_purity, dbScan_entropy = dbScanMetrics(scaledMealFeatureMatrix, mealDataMatrix)

# Save the clustering evaluation results in a CSV file
warnings.simplefilter(action='ignore', category=FutureWarning)
results = np.array([[kMeans_sse, dbScan_sse, kMeans_entropy, dbScan_entropy, kMeans_purity, dbScan_purity]])
headers = ['SSE KMeans', 'SSE DBSCAN', 'Entropy KMeans', 'Entropy DBSCAN', 'Purity KMeans', 'Purity DBSCAN']

print("\n------------------------------- Clustering evaluation results ------------------------\n")
print(tabulate(results, headers=headers, floatfmt=".4f"))
np.savetxt("Results.csv", results, delimiter=",", fmt="%10.4f")

