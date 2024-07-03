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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import calendar
import seaborn as sns

sleeps = glob.glob("*SLEEP*")
df_by_day = []
for s in sleeps:
    with open(s, 'r') as f:
        df_raw = {}#pd.DataFrame(columns=["start","stop","duration"])

        sleep_ = json.load(f)
        start_  = sleep_["startTime"][-10:-1]

        stop_  = sleep_["endTime"][-10:-1]
        duration_  = float(sleep_["duration"][0:-1])/3600.0
        df_raw["start"] = pd.to_datetime(sleep_["startTime"], utc=True).tz_convert('Australia/ACT')#.tz_localize(None)#+" "+df_raw["start"].str[:8]).dt.floor('h')
        df_raw["stop"] = pd.to_datetime(sleep_["endTime"], utc=True).tz_convert('Australia/ACT')#.tz_localize(None) #+" "+df_raw["stop"].str[:8]).dt.floor('h')
        df_raw["duration"] = duration_#pd.to_datetime(sleep_["duration"])
        print(df_raw["stop"].weekday)
        print(df_raw["stop"].hour)
        df_raw["activity_dttm"] = pd.to_datetime(df_raw["start"])#.str[:8]).dt.floor('h')
        df_by_day.append(df_raw)
df = pd.DataFrame(df_by_day)
df.index = pd.RangeIndex(len(df.index))
new_sleep = pd.DataFrame(columns=["sleep"])
print(new_sleep)
print(df)
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
df = data = data0


variance_collections = []
for i,row in enumerate(df.iterrows()):
     variance_collections.append(np.var(df.iloc[i].values)) 
print(np.sum(variance_collections))     

fig = plt.figure()
plt.plot([i for i in range(0,len(variance_collections))], variance_collections)
plt.savefig("variance_collection_sleep.png")


variance_collections
theta = 2 * np.pi * np.array(variance_collections)

#fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#ax.plot(theta,variance_collections)
variance_collections = (variance_collections - np.mean(variance_collections)) / np.std(variance_collections)
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, variance_collections)#, cmap='hsv', alpha=0.75)
plt.savefig("rotational_variance_collection_sleep.png")

fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# plot data
theta, r = np.meshgrid(np.linspace(0,2*np.pi,len(data)+1),np.arange(len(data.columns)+1))
p = ax.pcolormesh(theta,r,data.T.values, cmap="Reds")#,colors=colors)
fig.colorbar(p,ax=ax)

# create a colors dict

# plot
ax.legend(bbox_to_anchor=(1, 1.02), loc='upper left')

# set ticklabels
pos,step = np.linspace(0,2*np.pi,len(data),endpoint=False, retstep=True)
pos += step/2.
ax.set_xticks(pos)
ax.set_xticklabels(data.index)
ax.set_yticks(np.arange(len(data.columns)))
ax.set_yticklabels(data.columns)
plt.savefig("for_README.png")
