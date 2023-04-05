#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import fft, ifft, rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from joblib import dump, load

# Load the test dataset
test_data = pd.read_csv('test.csv', header=None)

# Function to create a feature matrix for no-meal data
def create_no_meal_feature_matrix(no_meal_data):
    
    """
    Extracts features from cleaned no-meal data using FFT and additional calculations.

    Parameters:
    -----------
    no_meal_data: pd.DataFrame
        A pandas DataFrame containing cleaned no-meal data.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame containing extracted no-meal features.
    """
    # Identify and remove rows with more than 5 missing values
    index_to_remove = no_meal_data.isna().sum(axis=1).replace(
        0, np.nan).dropna().where(lambda x: x > 5).dropna().index

    # Clean the non-meal data
    no_meal_clean_data = no_meal_data.drop(
        no_meal_data.index[index_to_remove]).reset_index().drop(columns='index')
    no_meal_clean_data = no_meal_clean_data.interpolate(
        method='linear', axis=1)

    # Identify and remove rows with missing values after interpolation
    index_to_drop_again = no_meal_clean_data.isna().sum(
        axis=1).replace(0, np.nan).dropna().index
    no_meal_clean_data = no_meal_clean_data.drop(
        no_meal_clean_data.index[index_to_drop_again]).reset_index().drop(columns='index')

    # Initialize an empty DataFrame to store the feature matrix
    feature_matrix = pd.DataFrame()

    no_meal_clean_data['tau_time'] = (
        24-no_meal_clean_data.iloc[:, 0:19].idxmax(axis=1))*5
    no_meal_clean_data['difference_in_glucose_normalized'] = (no_meal_clean_data.iloc[:, 0:19].max(
        axis=1)-no_meal_clean_data.iloc[:, 24])/(no_meal_clean_data.iloc[:, 24])
    # Calculate power and index of maximum values in the frequency domain
    first_max_power = []
    first_max_index = []
    second_max_power = []
    second_max_index = []

    for i in range(len(no_meal_clean_data)):
        array = abs(
            rfft(no_meal_clean_data.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sorted_array = abs(
            rfft(no_meal_clean_data.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        first_max_power.append(sorted_array[-2])
        second_max_power.append(sorted_array[-3])
        first_max_index.append(array.index(sorted_array[-2]))
        second_max_index.append(array.index(sorted_array[-3]))

    # Add calculated features to the feature matrix
    feature_matrix['tau_time'] = no_meal_clean_data['tau_time']
    feature_matrix['difference_in_glucose_normalized'] = no_meal_clean_data['difference_in_glucose_normalized']
    feature_matrix['first_max_power'] = first_max_power
    feature_matrix['second_max_power'] = second_max_power
    feature_matrix['first_max_index'] = first_max_index
    feature_matrix['second_max_index'] = second_max_index

    # Calculate first and second differential data and standard deviation
    first_differential_data = []
    second_differential_data = []
    standard_deviation = []

    for i in range(len(no_meal_clean_data)):
        first_differential_data.append(
            np.diff(no_meal_clean_data.iloc[:, 0:24].iloc[i].tolist()).max())
        second_differential_data.append(
            np.diff(np.diff(no_meal_clean_data.iloc[:, 0:24].iloc[i].tolist())).max())
        standard_deviation.append(np.std(no_meal_clean_data.iloc[i]))

    # Add calculated features to the feature matrix
    feature_matrix['2ndDifferential'] = second_differential_data
    feature_matrix['standard_deviation'] = standard_deviation

    return feature_matrix


# Create the feature matrix for the test dataset
dataset = create_no_meal_feature_matrix(test_data)

# Load the pre-trained Decision Tree Classifier model
with open('DecisionTreeClassifier.pickle', 'rb') as pre_trained:
    pickle_file = load(pre_trained)
    predict = pickle_file.predict(dataset)
    pre_trained.close()

pd.DataFrame(predict).to_csv('Result.csv', index=False, header=False)
