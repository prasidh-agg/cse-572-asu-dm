#!/usr/bin/env python
# coding: utf-8

from datetime import timedelta
from joblib import dump, load
import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft, rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle


# Load all the Insulin and CGM data from the given CSV files
def load_data():
    """
    Load insulin and CGM data from CSV files and return four data frames containing data for two patients.

    Returns
    -------
    tuple of pandas.DataFrame
        A tuple containing four pandas.DataFrame objects:
        insulin_data : pandas.DataFrame
            A data frame containing insulin data for patient 1.
        insulin_data2 : pandas.DataFrame
            A data frame containing insulin data for patient 2.
        cgm_data : pandas.DataFrame
            A data frame containing CGM data for patient 1.
        cgm_data2 : pandas.DataFrame
            A data frame containing CGM data for patient 2.
    """
    # Import insulin data
    insulin_data = pd.read_csv('InsulinData.csv', low_memory=False, usecols=[
                               'Date', 'Time', 'BWZ Carb Input (grams)'])
    insulin_data['date_time_stamp'] = pd.to_datetime(
        insulin_data['Date'] + ' ' + insulin_data['Time'])
    insulin_data2 = pd.read_csv('Insulin_patient2.csv', low_memory=False, usecols=[
                                'Date', 'Time', 'BWZ Carb Input (grams)'])
    insulin_data2['date_time_stamp'] = pd.to_datetime(
        insulin_data2['Date'] + ' ' + insulin_data2['Time'])

    # Import CGM data
    cgm_data = pd.read_csv('CGMData.csv', low_memory=False, usecols=[
                           'Date', 'Time', 'Sensor Glucose (mg/dL)'])
    cgm_data['date_time_stamp'] = pd.to_datetime(
        cgm_data['Date'] + ' ' + cgm_data['Time'])
    cgm_data2 = pd.read_csv('CGM_patient2.csv', low_memory=False, usecols=[
                            'Date', 'Time', 'Sensor Glucose (mg/dL)'])
    cgm_data2['date_time_stamp'] = pd.to_datetime(
        cgm_data2['Date'] + ' ' + cgm_data2['Time'])

    return insulin_data, insulin_data2, cgm_data, cgm_data2


# Define a function to create meal data given insulin and CGM data frames and a date identifier
def extract_meal_data(insulin_data, cgm_data, date_identifier):
    """
    This function creates a Pandas DataFrame containing the meal data for each patient based on their insulin and
    CGM data.

    Parameters:
    insulin_data: A Pandas DataFrame containing insulin data with columns 'Date', 'Time', and 'BWZ Carb Input (grams)'.
    cgm_data : A Pandas DataFrame containing CGM data with columns 'Date', 'Time', and 'Sensor Glucose (mg/dL)'.
    date_identifier: An integer specifying the format of the date string in the meal data output. 
                            1 for the format 'm/d/YYYY' and 2 for the format 'YYYY-mm-dd'.
    Returns:
    pd.DataFrame: A Pandas DataFrame containing the meal data for each patient.
    """
    # Create a copy of the insulin data frame and set the datetime column as the index
    insulin_data_copy = insulin_data.copy()
    insulin_data_copy = insulin_data_copy.set_index('date_time_stamp')

    # Create a new data frame that sorts the insulin data by datetime and removes any rows with null values
    insulin_data_cleaned = insulin_data_copy.sort_values(
        by='date_time_stamp', ascending=True).dropna().reset_index()

    # Replace any zero values in the 'BWZ Carb Input (grams)' column with NaN and drop any rows with NaN values
    insulin_data_cleaned['BWZ Carb Input (grams)'].replace(
        0.0, np.nan, inplace=True)
    insulin_data_cleaned = insulin_data_cleaned.dropna()
    insulin_data_cleaned = insulin_data_cleaned.reset_index().drop(columns='index')

    # Create a list of valid timestamps that have at least 2 hours between them
    valid_meal_timestamps_list = []
    value = 0

    for index, i in enumerate(insulin_data_cleaned['date_time_stamp']):
        try:
            value = (
                insulin_data_cleaned['date_time_stamp'][index+1]-i).seconds / 60.0
            if value >= 120:
                valid_meal_timestamps_list.append(i)
        except KeyError:
            break

    # Create a list of meal data for each valid timestamp
    meal_data_list = []

    if date_identifier == 1:
        date_format = '%-m/%-d/%Y'
        time_format = '%-H:%-M:%-S'
    elif date_identifier == 2:
        date_format = '%Y-%m-%d'
        time_format = '%H:%M:%S'
    else:
        raise ValueError('Invalid date identifier')

    for index, i in enumerate(valid_meal_timestamps_list):
        # Define the start and end times for the meal period
        start = pd.to_datetime(i - timedelta(minutes=30))
        end = pd.to_datetime(i + timedelta(minutes=120))
        # Get the date of the meal as a string in the format of 'm/d/YYYY'
        get_date = i.date().strftime(date_format)
        # Add the CGM data for the meal period to the list
        meal_data_list.append(cgm_data.loc[cgm_data['Date'] == get_date].set_index('date_time_stamp').between_time(
            start_time=start.strftime(time_format), end_time=end.strftime(time_format))['Sensor Glucose (mg/dL)'].values.tolist())
    # Return the list of meal data as a Pandas DataFrame
    return pd.DataFrame(meal_data_list)


# Define a function to create no-meal data given insulin and CGM data frames
def extract_no_meal_data(insulin_data, cgm_data):
    """
    Extracts no-meal data from CGM (continuous glucose monitoring) data for valid timestamps, as defined by insulin data.
    A valid timestamp is defined as a timestamp in the insulin data that is at least 4 hours apart from the next valid timestamp.

    Parameters:
    -----------
    insulin_data: pd.DataFrame
        A pandas DataFrame containing insulin data with at least two columns: 'date_time_stamp' and 'Insulin Value'.
    cgm_data: pd.DataFrame
        A pandas DataFrame containing CGM data with at least two columns: 'date_time_stamp' and 'Sensor Glucose (mg/dL)'.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame containing no-meal data for each valid timestamp in insulin_data. The DataFrame contains rows
        of 24-hour intervals of CGM data for each valid timestamp.
    """
    # Create a copy of the insulin data frame and sort it by datetime, replacing any zero values with NaN
    insulin_data_copy = insulin_data.copy()
    insulin_data_cleaned = insulin_data_copy.sort_values(
        by='date_time_stamp', ascending=True).replace(0.0, np.nan).dropna().copy()
    insulin_data_cleaned = insulin_data_cleaned.reset_index().drop(columns='index')

    # Create a list of valid timestamps that have at least 4 hours between them
    valid_no_meal_timestamps_list = []
    for idx, i in enumerate(insulin_data_cleaned['date_time_stamp']):
        try:
            value = (
                insulin_data_cleaned['date_time_stamp'][idx + 1]-i).seconds // 3600
            if value >= 4:
                valid_no_meal_timestamps_list.append(i)
        except KeyError:
            break

    # Create a list of no-meal data for each valid timestamp
    no_meal_list = []
    for idx, i in enumerate(valid_no_meal_timestamps_list):
        counter = 1
        try:
            # Calculate the number of 24-hour intervals between the two valid timestamps
            len_of_dataset = len(cgm_data.loc[(cgm_data['date_time_stamp'] >= valid_no_meal_timestamps_list[idx]+pd.Timedelta(
                hours=2)) & (cgm_data['date_time_stamp'] < valid_no_meal_timestamps_list[idx+1])])//24
            # Add each 24-hour interval of CGM data to the list
            while (counter <= len_of_dataset):
                if counter == 1:
                    no_meal_list.append(cgm_data.loc[(cgm_data['date_time_stamp'] >= valid_no_meal_timestamps_list[idx]+pd.Timedelta(hours=2)) & (
                        cgm_data['date_time_stamp'] < valid_no_meal_timestamps_list[idx+1])]['Sensor Glucose (mg/dL)'][:counter*24].values.tolist())
                    counter += 1
                else:
                    no_meal_list.append(cgm_data.loc[(cgm_data['date_time_stamp'] >= valid_no_meal_timestamps_list[idx]+pd.Timedelta(hours=2)) & (
                        cgm_data['date_time_stamp'] < valid_no_meal_timestamps_list[idx+1])]['Sensor Glucose (mg/dL)'][(counter-1)*24:(counter)*24].values.tolist())
                    counter += 1
        except IndexError:
            break

    # Return the list of no-meal data as a Pandas DataFrame
    return pd.DataFrame(no_meal_list)


# Clean data by dropping invalid rows and interpolating missing data
def drop_indices_and_clean_data(data):
    """
    Cleans meal data by dropping invalid rows and interpolating missing values. Also calculates additional columns 
    and drops any remaining NaN values.

    Parameters:
    -----------
    data: pd.DataFrame
        A pandas DataFrame containing meal data.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame with cleaned and preprocessed meal data.
    """
    # Find the index of meal periods with more than 6 missing values, drop them from the meal data data frame and interpolate the remaining missing values
    index_to_drop = data.isna().sum(axis=1).replace(
        0, np.nan).dropna().where(lambda x: x > 6).dropna().index
    data_cleaned = data.drop(
        data.index[index_to_drop]).reset_index().drop(columns='index')
    data_cleaned = data_cleaned.interpolate(method='linear', axis=1)

    # Find the index of meal periods with missing values again, drop them from the meal data dataframe and drop any remaining NaN values
    index_to_drop_again = data_cleaned.isna().sum(
        axis=1).replace(0, np.nan).dropna().index
    data_cleaned = data_cleaned.drop(
        data.index[index_to_drop_again]).reset_index().drop(columns='index')
    data_cleaned['tau_time'] = (data_cleaned.iloc[:, 22:25].idxmin(
        axis=1)-data_cleaned.iloc[:, 5:19].idxmax(axis=1))*5
    data_cleaned['difference_in_glucose_normalized'] = (data_cleaned.iloc[:, 5:19].max(
        axis=1)-data_cleaned.iloc[:, 22:25].min(axis=1))/(data_cleaned.iloc[:, 22:25].min(axis=1))
    data_cleaned = data_cleaned.dropna().reset_index().drop(columns='index')
    return data_cleaned


# Define a function to create a meal feature matrix given a meal data data frame
def extract_meal_feature_matrix(meal_data):
    """
    Extracts meal features from cleaned meal data using FFT and additional calculations.

    Parameters:
    -----------
    meal_data: pd.DataFrame
        A pandas DataFrame containing cleaned meal data.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame containing extracted meal features.
    """
    meal_cleaned_data = drop_indices_and_clean_data(meal_data)

    # Initialize empty lists for the different features in the meal feature matrix
    first_max_power = []
    first_max_index = []
    second_max_power = []
    second_max_index = []

    # Calculate the power of the first, second, and third max frequencies for each meal period using the FFT
    for i in range(len(meal_cleaned_data)):
        array = abs(
            rfft(meal_cleaned_data.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sorted_array = abs(
            rfft(meal_cleaned_data.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        first_max_power.append(sorted_array[-2])
        second_max_power.append(sorted_array[-3])
        first_max_index.append(array.index(sorted_array[-2]))
        second_max_index.append(array.index(sorted_array[-3]))

    # Initialize an empty meal feature matrix data frame
    meal_feature_matrix = pd.DataFrame()

    # Add the power of the second and third max frequencies as features to the meal feature matrix
    meal_feature_matrix['tau_time'] = meal_cleaned_data['tau_time']
    meal_feature_matrix['difference_in_glucose_normalized'] = meal_cleaned_data['difference_in_glucose_normalized']
    meal_feature_matrix['first_max_power'] = first_max_power
    meal_feature_matrix['second_max_power'] = second_max_power
    meal_feature_matrix['first_max_index'] = first_max_index
    meal_feature_matrix['second_max_index'] = second_max_index

    # Find the time of the minimum value between the 22nd and 25th column and the maximum value between the 5th and 19th column for each meal period
    tm = meal_cleaned_data.iloc[:, 22:25].idxmin(axis=1)
    maximum = meal_cleaned_data.iloc[:, 5:19].idxmax(axis=1)

    # Initialize empty lists for the second differential, standard deviation, and maximum difference between consecutive values features
    list1 = []
    second_differential_data = []
    standard_deviation = []

    # Calculate the second differential, standard deviation, and maximum difference between consecutive values features for each meal period
    for i in range(len(meal_cleaned_data)):
        list1.append(
            np.diff(meal_cleaned_data.iloc[:, maximum[i]:tm[i]].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(
            np.diff(meal_cleaned_data.iloc[:, maximum[i]:tm[i]].iloc[i].tolist())).max())
        standard_deviation.append(np.std(meal_cleaned_data.iloc[i]))

    # Add the second differential, standard deviation, and maximum difference between consecutive values
    meal_feature_matrix['2ndDifferential'] = second_differential_data
    meal_feature_matrix['standard_deviation'] = standard_deviation
    return meal_feature_matrix

# Define a function to create a no meal feature matrix given a no meal data data frame
def extract_no_meal_feature_matrix(no_meal_data):
    """
    Extracts no meal features from cleaned no meal data using FFT and additional calculations.

    Parameters:
    -----------
    no_meal_data: pd.DataFrame
        A pandas DataFrame containing cleaned no meal data.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame containing extracted no meal features.
    """
    no_meal_cleaned_data = drop_indices_and_clean_data(no_meal_data)

    no_meal_feature_matrix = pd.DataFrame()
    # Calculate power max and index for each frequency spectrum
    first_max_power = []
    first_max_index = []
    second_max_power = []
    second_max_index = []
    for i in range(len(no_meal_cleaned_data)):
        array = abs(
            rfft(no_meal_cleaned_data.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sorted_array = abs(
            rfft(no_meal_cleaned_data.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        first_max_power.append(sorted_array[-2])
        second_max_power.append(sorted_array[-3])
        first_max_index.append(array.index(sorted_array[-2]))
        second_max_index.append(array.index(sorted_array[-3]))
    no_meal_feature_matrix['tau_time'] = no_meal_cleaned_data['tau_time']
    no_meal_feature_matrix['difference_in_glucose_normalized'] = no_meal_cleaned_data['difference_in_glucose_normalized']
    no_meal_feature_matrix['first_max_power'] = first_max_power
    no_meal_feature_matrix['second_max_power'] = second_max_power
    no_meal_feature_matrix['first_max_index'] = first_max_index
    no_meal_feature_matrix['second_max_index'] = second_max_index
    # Calculate first and second differential and standard deviation for each row
    first_differential_data = []
    second_differential_data = []
    standard_deviation = []
    for i in range(len(no_meal_cleaned_data)):
        first_differential_data.append(
            np.diff(no_meal_cleaned_data.iloc[:, 0:24].iloc[i].tolist()).max())
        second_differential_data.append(
            np.diff(np.diff(no_meal_cleaned_data.iloc[:, 0:24].iloc[i].tolist())).max())
        standard_deviation.append(np.std(no_meal_cleaned_data.iloc[i]))
    no_meal_feature_matrix['2ndDifferential'] = second_differential_data
    no_meal_feature_matrix['standard_deviation'] = standard_deviation

    return no_meal_feature_matrix


insulin_data, insulin_data2, cgm_data, cgm_data2 = load_data()

# Create new meal data data frame for two patients
meal_data = extract_meal_data(insulin_data, cgm_data, 1)
meal_data2 = extract_meal_data(insulin_data2, cgm_data2, 2)

meal_data = meal_data.iloc[:, 0:30]
meal_data2 = meal_data2.iloc[:, 0:30]

# Create new no-meal data frame for two patients
no_meal_data = extract_no_meal_data(insulin_data, cgm_data)
no_meal_data2 = extract_no_meal_data(insulin_data2, cgm_data2)

print('Meal data for sample 1: ', meal_data)
print('--------------------------------------')
print('No-Meal data for sample 1: ', no_meal_data)
print('--------------------------------------')

# Create a meal feature matrix for first two patients
meal_feature_matrix = extract_meal_feature_matrix(meal_data)
meal_feature_matrix2 = extract_meal_feature_matrix(meal_data2)

# Concatenate the two meal feature matrices and drop the index column
meal_feature_matrix = pd.concat(
    [meal_feature_matrix, meal_feature_matrix2]).reset_index().drop(columns='index')

# Create no-meal feature matrices for both patients using the create_no_meal_feature_matrix function
no_meal_feature_matrix = extract_no_meal_feature_matrix(no_meal_data)
no_meal_feature_matrix2 = extract_no_meal_feature_matrix(no_meal_data2)

# Concatenate the two feature matrices and drop the index column
no_meal_feature_matrix = pd.concat(
    [no_meal_feature_matrix, no_meal_feature_matrix2]).reset_index().drop(columns='index')

print('Meal feature matrix for sample 1', meal_feature_matrix)
print('--------------------------------------')
print('No-Meal feature matrix for sample 1', no_meal_feature_matrix)
print('--------------------------------------')

# Add label column to the meal and non-meal feature matrices, concatenate them, and shuffle the resulting data frame
meal_feature_matrix['label'] = 1

# Assign label 0 to non-meal data and concatenate it with meal data
no_meal_feature_matrix['label'] = 0
total_data = pd.concat(
    [meal_feature_matrix, no_meal_feature_matrix]).reset_index().drop(columns='index')

# Shuffle the dataset and split it into training and testing sets using K-Fold cross-validation
dataset = shuffle(total_data, random_state=1).reset_index().drop(
    columns='index')
kfold = KFold(n_splits=10, shuffle=False)

# Separate the features from the labels and train a decision tree classifier
principaldata = dataset.drop(columns='label')
scores_rf = []
model = DecisionTreeClassifier(criterion="entropy")
for train_index, test_index in kfold.split(principaldata):
    x_train, x_test, y_train, y_test = principaldata.loc[train_index], principaldata.loc[
        test_index], dataset.label.loc[train_index], dataset.label.loc[test_index]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    scores_rf.append(model.score(x_test, y_test))
    # Print the individual metrics for each fold
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.3f}\n")

print('Prediction Score is : ', np.mean(scores_rf) * 100)

# Train the classifier on the entire dataset and save it to a file
classifier = DecisionTreeClassifier(criterion="entropy")
x, y = principaldata, dataset['label']
classifier.fit(x, y)
dump(classifier, 'DecisionTreeClassifier.pickle')
