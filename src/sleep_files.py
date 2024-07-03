import matplotlib
matplotlib.use('Agg')

import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os,fnmatch
import matplotlib as mpl
import matplotlib.cm as cm
import calendar
import seaborn as sns

def get_files_from_paths():
    sleeps = glob.glob("*SLEEP*")
    return sleeps
sleeps = get_sleep_data()    

def setup_build_plot():
  everything = get_files_from_paths()
  df_by_day = []
  for s in everything:
      with open(s, 'r') as f:
          df_raw = {}
          sleep_ = json.load(f)
          start_  = sleep_["startTime"][-10:-1]
          stop_  = sleep_["endTime"][-10:-1]
          duration_  = float(sleep_["duration"][0:-1])/3600.0
          df_raw["start"] = pd.to_datetime(sleep_["startTime"], utc=True).tz_convert('Australia/ACT')#.tz_localize(None)#+" "+df_raw["start"].str[:8]).dt.floor('h')
          df_raw["stop"] = pd.to_datetime(sleep_["endTime"], utc=True).tz_convert('Australia/ACT')#.tz_localize(None) #+" "+df_raw["stop"].str[:8]).dt.floor('h')
          df_raw["duration"] = duration_#pd.to_datetime(sleep_["duration"])
          df_raw["activity_dttm"] = pd.to_datetime(df_raw["start"])#.str[:8]).dt.floor('h')  
          df_by_day.append(df_raw)  
data0 = df_to_pivot_table(df_by_day)        
return data0  



def df_to_pivot_table(df_by_day):
  df = pd.DataFrame(df_by_day)
  df.index = pd.RangeIndex(len(df.index))
  new_sleep = pd.DataFrame(columns=["sleep"])
  cnt = 0 
  for i in range(0,len(df["stop"])*2):
      if i<len(df["stop"])-1:
          new_sleep.loc[cnt,"sleep"] = df.loc[i,"start"]
          cnt+=1
          new_sleep.loc[cnt,"sleep"] = df.loc[i,"stop"]
          cnt+=1
  temp = pd.Series([i.hour for i in new_sleep["sleep"]]).apply(lambda x: '{:02d}:00'.format(x))
  data0 = pd.crosstab([i.weekday() for i in new_sleep["sleep"]],temp )#.hour.apply(lambda x: '{:02d}:00'.format(x))).fillna(0)
  data0.fillna(0)  
  data0.index = [calendar.day_name[i][0:3] for i in data0.index]
  data0 = data0.T
  return data0



def setup_build_plot():
  everything = get_files_from_paths()
  df_by_day = []
  for s in everything:
      with open(s, 'r') as f:
          df_raw = {}
          sleep_ = json.load(f)
          start_  = sleep_["startTime"][-10:-1]
          stop_  = sleep_["endTime"][-10:-1]
          duration_  = float(sleep_["duration"][0:-1])/3600.0
          df_raw["start"] = pd.to_datetime(sleep_["startTime"], utc=True).tz_convert('Australia/ACT')#.tz_localize(None)#+" "+df_raw["start"].str[:8]).dt.floor('h')
          df_raw["stop"] = pd.to_datetime(sleep_["endTime"], utc=True).tz_convert('Australia/ACT')#.tz_localize(None) #+" "+df_raw["stop"].str[:8]).dt.floor('h')
          df_raw["duration"] = duration_
          df_raw["activity_dttm"] = pd.to_datetime(df_raw["start"])#.str[:8]).dt.floor('h')  
          df_by_day.append(df_raw)  
data0 = df_to_pivot_table(df_by_day)        
return data0  

data0 = setup_build_plot()
fig = plt.figure()
plt.pcolor(df)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.savefig("basic_for_README_exercise.png")


def get_variance(data0):
  variance_collection = []
  for i,row in enumerate(df.iterrows()):
       variance_collection.append(np.var(df.iloc[i].values)) 
  return variance_collection
  
variance_collection = get_variance(data0)
print(np.sum(variance_collection))     



fig = plt.figure()
plt.plot([i for i in range(0,len(variance_collection))], variance_collection)
plt.savefig("variance_collection.png")


fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# plot data
theta, r = np.meshgrid(np.linspace(0,2*np.pi,len(data)+1),np.arange(len(data.columns)+1))
ax.pcolormesh(theta,r,data.T.values, cmap="Reds")
fig.colorbar(p,ax=ax)

# set ticklabels
pos,step = np.linspace(0,2*np.pi,len(data),endpoint=False, retstep=True)
pos += step/2.
ax.set_xticks(pos)
ax.set_xticklabels(data.index)

ax.set_yticks(np.arange(len(data.columns)))
ax.set_yticklabels(data.columns)
plt.savefig("for_README_exercise.png")
















