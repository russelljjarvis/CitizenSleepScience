from sleepFiles import *


def get_files_from_paths():
  runs = glob.glob("*RUN*")
  walks = glob.glob("*WALK*")
  biking = glob.glob("*BIKING*")  
  everything = []
  everything.extend(runs)
  everything.extend(walks)
  everything.extend(biking)
  return everything

def setup_build_plot():
  everything = get_files_from_paths()
  
  df_by_day = []
  for s in everything:
      with open(s, 'r') as f:
          df_raw = {}#pd.DataFrame(columns=["start","stop","duration"])
  
          sleep_ = json.load(f)
          start_  = sleep_["startTime"][-10:-1]
  
          stop_  = sleep_["endTime"][-10:-1]
          #print("start ",sleep_["startTime"])
          #print("stop ",sleep_["endTime"])
          #print("duration", sleep_["duration"])
          duration_  = float(sleep_["duration"][0:-1])/3600.0
          #import pdb
          #pdb.set_trace()
          #pd.to_datetime(df['time'], utc=True ).dt.tz_convert('Australia/ACT')
          df_raw["start"] = pd.to_datetime(sleep_["startTime"], utc=True).tz_convert('Australia/ACT')#.tz_localize(None)#+" "+df_raw["start"].str[:8]).dt.floor('h')
          #import pdb
          #pdb.set_trace()
          df_raw["stop"] = pd.to_datetime(sleep_["endTime"], utc=True).tz_convert('Australia/ACT')#.tz_localize(None) #+" "+df_raw["stop"].str[:8]).dt.floor('h')
          df_raw["duration"] = duration_#pd.to_datetime(sleep_["duration"])
          print(df_raw["stop"].weekday)
          print(df_raw["stop"].hour)
  
          df_raw["activity_dttm"] = pd.to_datetime(df_raw["start"])#.str[:8]).dt.floor('h')
  
          df_by_day.append(df_raw)
  
          #print(duration_)
          #print(start_)
          #print(stop_)
  df = pd.DataFrame(df_by_day)
  df.index = pd.RangeIndex(len(df.index))
  
  
  
  # generate the table with timestamps
  #np.random.seed(1)
  #times = pd.Series(pd.to_datetime("Nov 1 '16 at 0:42") + 
  #                  pd.to_timedelta(np.random.rand(10000)*60*24*40, unit='m'))
  # generate counts of each (weekday, hour)
  #data = pd.crosstab(times.dt.weekday, 
  #                   times.dt.hour.apply(lambda x: '{:02d}:00'.format(x))).fillna(0)
  
  
  
  #data.index = [calendar.day_name[i][0:3] for i in data.index]
  #data = data.T
  
  
  #print([i.weekday() for i in df["stop"]], [i.hour for i in df["stop"]])
  #import pdb
  #pdb.set_trace()
  
  
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
  
  print(new_sleep)
  #new_sleep["sleep"].values = new_sleep["sleep"].values + df["stop"].values
  #new_sleep["sleep"].values = new_sleep["sleep"].values + df["start"].values
  
  
  temp = pd.Series([i.hour for i in new_sleep["sleep"]]).apply(lambda x: '{:02d}:00'.format(x))
  data0 = pd.crosstab([i.weekday() for i in new_sleep["sleep"]],temp )#.hour.apply(lambda x: '{:02d}:00'.format(x))).fillna(0)
  data0.fillna(0)
  
  data0.index = [calendar.day_name[i][0:3] for i in data0.index]
  data0 = data0.T
  #df = data = data0
  return data0  
data0 = setup_build_plot()
fig = plt.figure()
plt.pcolor(df)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.savefig("basic_for_README_exercise.png")


#dft = df.T
variance_collectione = []
for i,row in enumerate(df.iterrows()):
     variance_collectione.append(np.var(df.iloc[i].values)) 
  #return variance_collectione, data
print(np.sum(variance_collectione))     



fig = plt.figure()
plt.plot([i for i in range(0,len(variance_collectione))], variance_collectione)
plt.savefig("variance_collection.png")


fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# plot data
theta, r = np.meshgrid(np.linspace(0,2*np.pi,len(data)+1),np.arange(len(data.columns)+1))
ax.pcolormesh(theta,r,data.T.values, cmap="Reds")

# set ticklabels
pos,step = np.linspace(0,2*np.pi,len(data),endpoint=False, retstep=True)
pos += step/2.
ax.set_xticks(pos)
ax.set_xticklabels(data.index)

ax.set_yticks(np.arange(len(data.columns)))
ax.set_yticklabels(data.columns)
plt.savefig("for_README_exercise.png")
