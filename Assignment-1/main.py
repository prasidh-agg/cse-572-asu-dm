import pandas as pd
import numpy as np

#Read the Datasets
cgm_data=pd.read_csv('../CGMData.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])
insulin_data=pd.read_csv('../InsulinData.csv',low_memory=False)

#Combine date and time to make a dateTimeStamp column
cgm_data['date_time_stamp']=pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])
insulin_data['date_time_stamp']=pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'])


#Find the data corresponding to the auto modes and the manual modes using the alarms
#from the insulin data
auto_mode_begin=insulin_data.sort_values(by='date_time_stamp', ascending=True).loc[insulin_data['Alarm']=='AUTO MODE ACTIVE PLGM OFF'].iloc[0]['date_time_stamp']

auto_mode_data=cgm_data.sort_values(by='date_time_stamp', ascending=True).loc[cgm_data['date_time_stamp']>=auto_mode_begin]
manual_mode_data=cgm_data.sort_values(by='date_time_stamp', ascending=True).loc[cgm_data['date_time_stamp']<auto_mode_begin]


#Create a copy of the auto mode data and index it using date_time_stamp
auto_mode_data_date_index=auto_mode_data.copy()
auto_mode_data_date_index=auto_mode_data_date_index.set_index('date_time_stamp')

#Create a copy of the manual mode data and index it using date_time_stamp
manual_mode_data_date_index=manual_mode_data.copy()
manual_mode_data_date_index=manual_mode_data_date_index.set_index('date_time_stamp')

#Filter the groups where the count is greater than 0.8 * 288
#Drop missing values, extract the date index 
#Convert it to a Python list, which is stored in the glucose_data variable.
auto_glucose_data=auto_mode_data_date_index.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x:x>int(0.8*288)).dropna().index.tolist()


manual_glucose_data=manual_mode_data_date_index.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(lambda x:x>int(0.8*288)).dropna().index.tolist()
auto_mode_data_date_index=auto_mode_data_date_index.loc[auto_mode_data_date_index['Date'].isin(auto_glucose_data)]

manual_mode_data_date_index=manual_mode_data_date_index.loc[manual_mode_data_date_index['Date'].isin(manual_glucose_data)]

#Calculate the percentage of time in a day that the patient's glucose levels were in hyperglycemia.
percent_time_hyper_wholeday_auto=(auto_mode_data_date_index.between_time('0:00:00','23:59:59').loc[auto_mode_data_date_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hyper_daytime_auto=(auto_mode_data_date_index.between_time('6:00:00','23:59:59').loc[auto_mode_data_date_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hyper_overnight_auto=(auto_mode_data_date_index.between_time('0:00:00','5:59:59').loc[auto_mode_data_date_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hyper_wholeday_manual=(manual_mode_data_date_index.between_time('0:00:00','23:59:59').loc[manual_mode_data_date_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hyper_daytime_manual=(manual_mode_data_date_index.between_time('6:00:00','23:59:59').loc[manual_mode_data_date_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hyper_overnight_manual=(manual_mode_data_date_index.between_time('0:00:00','5:59:59').loc[manual_mode_data_date_index['Sensor Glucose (mg/dL)']>180].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#Calculate the percentage of time in a day that the patient's glucose levels were in critical hyperglycemia.
percent_time_hyper_critical_wholeday_auto=(auto_mode_data_date_index.between_time('0:00:00','23:59:59').loc[auto_mode_data_date_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hyper_critical_daytime_auto=(auto_mode_data_date_index.between_time('6:00:00','23:59:59').loc[auto_mode_data_date_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hyper_critical_overnight_auto=(auto_mode_data_date_index.between_time('0:00:00','5:59:59').loc[auto_mode_data_date_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hyper_critical_wholeday_manual=(manual_mode_data_date_index.between_time('0:00:00','23:59:59').loc[manual_mode_data_date_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hyper_critical_daytime_manual=(manual_mode_data_date_index.between_time('6:00:00','23:59:59').loc[manual_mode_data_date_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hyper_critical_overnight_manual=(manual_mode_data_date_index.between_time('0:00:00','5:59:59').loc[manual_mode_data_date_index['Sensor Glucose (mg/dL)']>250].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#Calculate the percentage of time in a day that the patient's glucose levels were between 70 and 180mg/dL.
percent_time_range_wholeday_auto=(auto_mode_data_date_index.between_time('0:00:00','23:59:59').loc[(auto_mode_data_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data_date_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_range_daytime_auto=(auto_mode_data_date_index.between_time('6:00:00','23:59:59').loc[(auto_mode_data_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data_date_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_range_overnight_auto=(auto_mode_data_date_index.between_time('0:00:00','5:59:59').loc[(auto_mode_data_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data_date_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_range_wholeday_manual=(manual_mode_data_date_index.between_time('0:00:00','23:59:59').loc[(manual_mode_data_date_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data_date_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_range_daytime_manual=(manual_mode_data_date_index.between_time('6:00:00','23:59:59').loc[(manual_mode_data_date_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data_date_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_range_overnight_manual=(manual_mode_data_date_index.between_time('0:00:00','5:59:59').loc[(manual_mode_data_date_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data_date_index['Sensor Glucose (mg/dL)']<=180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#Calculate the percentage of time in a day that the patient's glucose levels were between 70 and 150mg/dL.
percent_time_range_sec_wholeday_auto=(auto_mode_data_date_index.between_time('0:00:00','23:59:59').loc[(auto_mode_data_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data_date_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_range_sec_daytime_auto=(auto_mode_data_date_index.between_time('6:00:00','23:59:59').loc[(auto_mode_data_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data_date_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_range_sec_overnight_auto=(auto_mode_data_date_index.between_time('0:00:00','5:59:59').loc[(auto_mode_data_date_index['Sensor Glucose (mg/dL)']>=70) & (auto_mode_data_date_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_range_sec_wholeday_manual=(manual_mode_data_date_index.between_time('0:00:00','23:59:59').loc[(manual_mode_data_date_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data_date_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_range_sec_daytime_manual=(manual_mode_data_date_index.between_time('6:00:00','23:59:59').loc[(manual_mode_data_date_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data_date_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_range_sec_overnight_manual=(manual_mode_data_date_index.between_time('0:00:00','5:59:59').loc[(manual_mode_data_date_index['Sensor Glucose (mg/dL)']>=70) & (manual_mode_data_date_index['Sensor Glucose (mg/dL)']<=150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#Calculate the percentage of time in a day that the patient's glucose levels were in hypoglycemia level 1.
percent_time_hypo1_wholeday_auto=(auto_mode_data_date_index.between_time('0:00:00','23:59:59').loc[auto_mode_data_date_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hypo1_daytime_auto=(auto_mode_data_date_index.between_time('6:00:00','23:59:59').loc[auto_mode_data_date_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hypo1_overnight_auto=(auto_mode_data_date_index.between_time('0:00:00','5:59:59').loc[auto_mode_data_date_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hypo1_wholeday_manual=(manual_mode_data_date_index.between_time('0:00:00','23:59:59').loc[manual_mode_data_date_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hypo1_daytime_manual=(manual_mode_data_date_index.between_time('6:00:00','23:59:59').loc[manual_mode_data_date_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hypo1_overnight_manual=(manual_mode_data_date_index.between_time('0:00:00','5:59:59').loc[manual_mode_data_date_index['Sensor Glucose (mg/dL)']<70].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

#Calculate the percentage of time in a day that the patient's glucose levels were in hypoglycemia level 2.
percent_time_hypo2_wholeday_auto=(auto_mode_data_date_index.between_time('0:00:00','23:59:59').loc[auto_mode_data_date_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hypo2_daytime_auto=(auto_mode_data_date_index.between_time('6:00:00','23:59:59').loc[auto_mode_data_date_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hypo2_overnight_auto=(auto_mode_data_date_index.between_time('0:00:00','5:59:59').loc[auto_mode_data_date_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hypo2_wholeday_manual=(manual_mode_data_date_index.between_time('0:00:00','23:59:59').loc[manual_mode_data_date_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hypo2_daytime_manual=(manual_mode_data_date_index.between_time('6:00:00','23:59:59').loc[manual_mode_data_date_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

percent_time_hypo2_overnight_manual=(manual_mode_data_date_index.between_time('0:00:00','5:59:59').loc[manual_mode_data_date_index['Sensor Glucose (mg/dL)']<54].groupby('Date')['Sensor Glucose (mg/dL)'].count()/288*100)

results=pd.DataFrame(
    {
        'percent_time_hyper_overnight':[ percent_time_hyper_overnight_manual.mean(axis=0),percent_time_hyper_overnight_auto.mean(axis=0)],
        'percent_time_hyper_critical_overnight':[ percent_time_hyper_critical_overnight_manual.mean(axis=0),percent_time_hyper_critical_overnight_auto.mean(axis=0)],
        'percent_time_range_overnight':[ percent_time_range_overnight_manual.mean(axis=0),percent_time_range_overnight_auto.mean(axis=0)],
        'percent_time_range_sec_overnight':[ percent_time_range_sec_overnight_manual.mean(axis=0),percent_time_range_sec_overnight_auto.mean(axis=0)],
        'percent_time_hypo1_overnight':[ percent_time_hypo1_overnight_manual.mean(axis=0),percent_time_hypo1_overnight_auto.mean(axis=0)],
        'percent_time_hypo2_overnight':[ np.nan_to_num(percent_time_hypo2_overnight_manual.mean(axis=0)),percent_time_hypo2_overnight_auto.mean(axis=0)],         
        'percent_time_hyper_daytime':[ percent_time_hyper_daytime_manual.mean(axis=0),percent_time_hyper_daytime_auto.mean(axis=0)],
        'percent_time_hyper_critical_daytime':[ percent_time_hyper_critical_daytime_manual.mean(axis=0),percent_time_hyper_critical_daytime_auto.mean(axis=0)],
        'percent_time_range_daytime':[ percent_time_range_daytime_manual.mean(axis=0),percent_time_range_daytime_auto.mean(axis=0)],
        'percent_time_range_sec_daytime':[ percent_time_range_sec_daytime_manual.mean(axis=0),percent_time_range_sec_daytime_auto.mean(axis=0)],
        'percent_time_hypo1_daytime':[ percent_time_hypo1_daytime_manual.mean(axis=0),percent_time_hypo1_daytime_auto.mean(axis=0)],
        'percent_time_hypo2_daytime':[ percent_time_hypo2_daytime_manual.mean(axis=0),percent_time_hypo2_daytime_auto.mean(axis=0)],
        'percent_time_hyper_wholeday':[ percent_time_hyper_wholeday_manual.mean(axis=0),percent_time_hyper_wholeday_auto.mean(axis=0)],
        'percent_time_hyper_critical_wholeday':[ percent_time_hyper_critical_wholeday_manual.mean(axis=0),percent_time_hyper_critical_wholeday_auto.mean(axis=0)],
        'percent_time_range_wholeday':[ percent_time_range_wholeday_manual.mean(axis=0),percent_time_range_wholeday_auto.mean(axis=0)],
        'percent_time_range_sec_wholeday':[ percent_time_range_sec_wholeday_manual.mean(axis=0),percent_time_range_sec_wholeday_auto.mean(axis=0)],
        'percent_time_hypo1_wholeday':[ percent_time_hypo1_wholeday_manual.mean(axis=0),percent_time_hypo1_wholeday_auto.mean(axis=0)],
        'percent_time_hypo2_wholeday':[ percent_time_hypo2_wholeday_manual.mean(axis=0),percent_time_hypo2_wholeday_auto.mean(axis=0)]
    }, index=['manual_mode','auto_mode']
)

results.to_csv('../Results.csv',header=False,index=False)