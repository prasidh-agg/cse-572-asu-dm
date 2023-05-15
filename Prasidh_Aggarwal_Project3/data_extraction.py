# Import necessary libraries
import pandas as pd
import datetime as dt
pd.options.mode.chained_assignment = None # type: ignore

inf = dt.datetime(2100, 12, 12)

# Read and preprocess both insulin and CGM data
def extractInsulinAndCgmData(insulin_df, cgm_df):
    insulin_data = pd.read_csv(insulin_df, parse_dates=[
                               ['Date', 'Time']], keep_date_col=True, low_memory=False)
    insulin_df = insulin_data[['Date_Time', 'BWZ Carb Input (grams)']]
    insulin_df = insulin_df.rename(columns={'BWZ Carb Input (grams)': 'Carb_Input'})

    cgm_data = pd.read_csv(cgm_df, parse_dates=[
                           ['Date', 'Time']], keep_date_col=True, low_memory=False)
    cgm_df = cgm_data[['Date_Time', 'Index','Sensor Glucose (mg/dL)', 'Date', 'Time']]
    cgm_df = cgm_df.rename(columns={'Sensor Glucose (mg/dL)': 'Sensor_Glucose'})

    return insulin_df, cgm_df


def extractMealStartTimes(insulin_df, cgm_df):
    insulin_df = insulin_df[(insulin_df['Carb_Input'].notna()) & (insulin_df['Carb_Input'] != 0)]
    insulin_df = insulin_df.set_index('Date_Time').sort_index().reset_index()
    
    # Include only those meal periods for which the next carb intake is atleast after 2 hrs
    mask = (insulin_df['Date_Time'].shift(-1, fill_value=inf) - insulin_df['Date_Time'] \
            >= dt.timedelta(hours=2))
    
    insulin_df = insulin_df[mask]
    
    # Column rename is required for the following merge
    insulin_df = insulin_df.rename(columns = {'Date_Time': 'Pseudo_Start_Time'})
    
    cgm_df = cgm_df[cgm_df['Sensor_Glucose'].notna()]
    cgm_df = cgm_df.set_index('Date_Time').sort_index().reset_index()
    
    meal_df = pd.merge_asof(insulin_df, cgm_df, left_on='Pseudo_Start_Time', \
                            right_on='Date_Time', direction='forward')[['Date_Time', 'Carb_Input']]
    
    min, max = meal_df['Carb_Input'].min(), meal_df['Carb_Input'].max()
    
    # Binning BWZ Carb Input (grams) into bins of range 20
    meal_df['Carb_Input'] = (meal_df['Carb_Input'] - min) // 20
    
    return meal_df