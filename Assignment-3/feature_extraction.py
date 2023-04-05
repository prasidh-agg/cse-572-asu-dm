# Import necessary libraries
import numpy as np
import pandas as pd
from data_extraction import *
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans

# This function takes in meal data and insulin data, calculates meal features, and returns a feature matrix
def extractMealFeatureMatrix(df_meal, insulin_df):
    
    # Initialize an empty list to store meal features
    meal_input = []  
    
    # Reset the index of the meal data
    df_meal.reset_index()  

    # Iterate through the rows of the meal data
    for index, x in df_meal.iterrows():
        # Define a time window of 2 hours after the meal and 30 minutes before the meal
        stop = x['Date_Time'] + pd.DateOffset(hours=2)
        start = x['Date_Time'] + pd.DateOffset(minutes=-30)
        
        # Select the insulin data within the time window
        meal = insulin_df.loc[(insulin_df['Date_Time'] >= start) & (
            insulin_df['Date_Time'] < stop)]
        
        # Set the index to 'Date_Time'
        meal.set_index('Date_Time', inplace=True)
        
        # Sort the dataframe by 'Date_Time' and reset the index
        meal = meal.sort_index().reset_index()
        isCorrect, meal = extractSensorTimeIntervals(
            meal, 30)  # Check for correct sensor time intervals
        if isCorrect == False:
            continue
        
        # Extract and reshape the 'Sensor Glucose (mg/dL)' values into a row array
        meal_feature = meal[[
            'Sensor Glucose (mg/dL)']].to_numpy().reshape(1, 30)
        
        # Insert the index at the beginning
        meal_feature = np.insert(meal_feature, 0, index, axis=1)
        
        # Insert the meal value after the index
        meal_feature = np.insert(meal_feature, 1, x['meal'], axis=1)
        
        # Append the meal feature array to the meal input list
        meal_input.append(meal_feature)

    # Return the meal input list as a NumPy array
    return np.array(meal_input).squeeze()

# This function takes in input data, computes various statistics, and returns a normalized feature matrix
def getNormalizedFeatureMatrix(input):
    data_frame = pd.DataFrame(data=input)
    
    # Calculate the minimum value of each row
    df = pd.DataFrame(data=data_frame.min(axis=1), columns=['minimum'])
    
    # Calculate the median value of each row
    df['median'] = data_frame.median(axis=1)
    df['sum'] = data_frame.sum(axis=1)  # Calculate the sum of each row
    
    # Calculate the maximum value of each row
    df['maximum'] = data_frame.max(axis=1)
    
    # Calculate the range (max-min) of each row
    df['min_max'] = df['maximum']-df['minimum']
    
    # Return the normalized feature matrix
    return MinMaxScaler().fit_transform(df)  

