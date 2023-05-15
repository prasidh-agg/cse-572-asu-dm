# Import necessary libraries
import numpy as np
import pandas as pd
from data_extraction import *
from scipy.stats import entropy, iqr
from scipy.signal import periodogram

# Define a function to extract meal data and create a feature matrix
def computeMealDataMatrix(cgm_df, meal_df):
    
    # list to store extracted meal data
    meal_data_list = []
    
    # list to store the corresponding carb inputs
    ground_truth = []
    
    for _, row in meal_df.iterrows():
        
        # Define the start time of the meal as 30 minutes before the meal time
        meal_start = row['Date_Time'] - pd.DateOffset(minutes=30)
        
        # Define the end time of the meal as 2 hours after the meal time
        meal_end = row['Date_Time'] + pd.DateOffset(hours=2)
        
        # Extract the meal data within the meal period
        meal = cgm_df.loc[(cgm_df['Date_Time'] >= meal_start) & (cgm_df['Date_Time'] < meal_end)]
        
        # Remove meal periods with <30 readings
        if (meal_df.shape[0] < 30):
            continue
            
        # Remove missing glucose sensor readings
        meal = meal[meal['Sensor_Glucose'].notna()]
        
        # Sort meal data by time
        meal = meal.set_index('Date_Time').sort_index().reset_index()

        # Remove readings <300 seconds apart
        mask = (meal['Date_Time'].shift(-1, fill_value=inf) - meal['Date_Time'] \
                >= dt.timedelta(seconds=300))
        
        meal = meal[mask]

        # Only include meal_period if it has exactly 30 readings
        if (meal.shape[0] == 30):
            meal_data_list.append(meal['Sensor_Glucose'])
            ground_truth.append(row['Carb_Input'])
        
    # Concatenate the meal data and transpose it to get a feature matrix
    feature_matrix = pd.concat(meal_data_list, axis=1).transpose()
    
    # Add the carb input as a label to the feature matrix
    feature_matrix['label'] = ground_truth 
    return feature_matrix


def computeMealFeatureMatrix(meal_data_matrix):
    
    # Exclude ground truth
    _input = meal_data_matrix.iloc[:, :-1]
    
    features = pd.DataFrame()
    
    velocity = _input.diff(axis=1).dropna(axis=1, how='all')
    features['velocity_min'] = velocity.min(axis=1)
    features['velocity_max'] = velocity.max(axis=1)
    features['velocity_mean'] = velocity.mean(axis=1)

    acceleration = velocity.diff(axis=1).dropna(axis=1, how='all')
    features['acceleration_min'] = acceleration.min(axis=1)
    features['acceleration_max'] = acceleration.max(axis=1)
    features['acceleration_mean'] = acceleration.mean(axis=1)

    features['entropy'] = _input.apply(lambda row: entropy(row, base=2), axis=1)
    features['iqr'] = _input.apply(lambda row: entropy(row, base=2), axis=1)
    
    fft_values = _input.apply(lambda row: np.fft.fft(row), axis=1)

    # Get the indices of the frequencies sorted by decreasing amplitude
    fft_indices = fft_values.apply(lambda row: np.argsort(np.abs(row))[::-1])

    # Select the first 6 peaks of each row
    fft_peaks = fft_indices.apply(lambda row: row[:6])
    fft_peaks = fft_peaks.apply(pd.Series)
    fft_peaks.columns = ['fft_max_' + str(i+1) for i in fft_peaks.apply(pd.Series).columns]
    
    features = pd.concat([features, fft_peaks], axis=1)
    
    _input = meal_data_matrix.iloc[:, :-1]
    psd = _input.apply(lambda row: periodogram(row)[1], axis=1)
    psd = psd.apply(lambda row: [np.mean(row[0:5]), np.mean(row[5:10]), np.mean(row[10:16])])
    psd = psd.apply(pd.Series)
    psd.columns = ['psd1', 'psd2', 'psd3']

    features = pd.concat([features, psd], axis=1)
    
    return features
