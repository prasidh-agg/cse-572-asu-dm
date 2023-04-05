# Import necessary libraries
import pandas as pd
pd.options.mode.chained_assignment = None # type: ignore

# Read and preprocess both insulin and CGM data
def extractInsulinAndCgmData(insulin_df, cgm_df):
    insulin_data = pd.read_csv(insulin_df, parse_dates=[
                               ['Date', 'Time']], keep_date_col=True, low_memory=False)
    insulin_df = insulin_data[['Date_Time', 'Index',
                               'Sensor Glucose (mg/dL)', 'Date', 'Time']]
    insulin_df['Index'] = insulin_df.index

    cgm_data = pd.read_csv(cgm_df, parse_dates=[
                           ['Date', 'Time']], low_memory=False)
    cgm_df = cgm_data[['Date_Time', 'BWZ Carb Input (grams)']]
    cgm_df = cgm_df.rename(columns={'BWZ Carb Input (grams)': 'meal'})
    return insulin_df, cgm_df

# Extract meal times from CGM data
def extractMealTimes(cgm_data_frame):
    insulin_df = cgm_data_frame.copy()
    insulin_df = insulin_df.loc[insulin_df['meal'].notna(
    ) & insulin_df['meal'] != 0]
    insulin_df.set_index(['Date_Time'], inplace=True)
    insulin_df = insulin_df.sort_index().reset_index()
    insulin_df_diff = insulin_df.diff(axis=0)
    insulin_df_diff = insulin_df_diff.loc[insulin_df_diff['Date_Time'].dt.seconds >= 7200]
    insulin_df = insulin_df.join(
        insulin_df_diff, lsuffix='_caller', rsuffix='_other')
    insulin_df = insulin_df.loc[insulin_df['Date_Time_other'].notna(), [
        'Date_Time_caller', 'meal_caller']]
    insulin_df = insulin_df.rename(columns={'meal_caller': 'meal'})
    return insulin_df

# Calculate intervals between meals
def calculateMealIntervals(cgm_data_point, insulin_meal_point):
    insulin_df = cgm_data_point.copy()
    cgm_df = insulin_meal_point.copy()
    insulin_df = insulin_df.loc[insulin_df['Sensor Glucose (mg/dL)'].notna()]
    insulin_df.set_index(['Date_Time'], inplace=True)
    insulin_df = insulin_df.sort_index().reset_index()
    cgm_df.set_index(["Date_Time_caller"], inplace=True)
    cgm_df = cgm_df.sort_index().reset_index()
    result = pd.merge_asof(cgm_df, insulin_df, left_on='Date_Time_caller',
                           right_on='Date_Time', direction="forward")
    return result


# Extract meal intervals and meal times
def extractMealTimesAndIntervals(cgm_df, ins_df):
    ins_time = extractMealTimes(ins_df)
    result = calculateMealIntervals(cgm_df, ins_time)
    return result


# Calculate sensor time intervals for meal data
def extractSensorTimeIntervals(df, val):
    cgm_data = df.loc[df['Sensor Glucose (mg/dL)'].notna()
                      ]['Sensor Glucose (mg/dL)'].count()

    if cgm_data < val:
        return False, None

    before_time = None
    val = 0
    for x in df.iterrows():
        if before_time == None:
            before_time = x[1]['Date_Time']
            val += 1
            continue

        if (x[1]['Date_Time'] - before_time).seconds < 300:
            df.at[val, 'Sensor Glucose (mg/dL)'] = -999
            val += 1
            continue

        before_time = x[1]['Date_Time']
        val += 1

    df = df.loc[df['Sensor Glucose (mg/dL)'] != -999]

    if df['Sensor Glucose (mg/dL)'].count() == val:
        return True, df
    else:
        return False, None