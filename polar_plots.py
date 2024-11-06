import matplotlib
matplotlib.use('Agg')

import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar
import seaborn as sns
from matplotlib.patches import FancyArrow
import copy

# Functions for data processing
def df_to_pivot_table(df_by_day):
    df = pd.DataFrame(df_by_day)
    df.index = pd.RangeIndex(len(df.index))
    new_sleep = pd.DataFrame(columns=["sleep"])
    cnt = 0 
    for i in range(0, len(df["stop"]) * 2):
        if i < len(df["stop"]) - 1:
            new_sleep.loc[cnt, "sleep"] = df.loc[i, "start"]
            cnt += 1
            new_sleep.loc[cnt, "sleep"] = df.loc[i, "stop"]
            cnt += 1
    temp = pd.Series([i.hour for i in new_sleep["sleep"]]).apply(lambda x: '{:02d}:00'.format(x))
    data0 = pd.crosstab([i.weekday() for i in new_sleep["sleep"]], temp)
    data0.fillna(0, inplace=True)  
    data0.index = [calendar.day_name[i][0:3] for i in data0.index]
    return data0.T

def get_files_from_paths():
    return glob.glob("*SLEEP*")

def setup_build_plot():
    everything = get_files_from_paths()
    df_by_day = []
    for s in everything:
        with open(s, 'r') as f:
            sleep_ = json.load(f)
            df_raw = {
                "start": pd.to_datetime(sleep_["startTime"], utc=True).tz_convert('Australia/ACT'),
                "stop": pd.to_datetime(sleep_["endTime"], utc=True).tz_convert('Australia/ACT'),
                "duration": float(sleep_["duration"][:-1]) / 3600.0
            }
            df_by_day.append(df_raw)  
    return df_to_pivot_table(df_by_day)

# Load and normalize data
data0 = setup_build_plot()
data0.index = pd.to_datetime(data0.index, format='%H:%M')
data0 /= data0.values.max()  # Normalize data to range [0, 1]


#df = setup_build_plot()
fig = plt.figure()
plt.pcolor(data0)
plt.yticks(np.arange(0.5, len(data0.index), 1), data0.index)
plt.xticks(np.arange(0.5, len(data0.columns), 1), data0.columns)
plt.savefig("basic_for_README_exercise.png")

# Split pivot tables for sleep initiation and wake time
sleep_initiation_table = data0.copy()
sleep_initiation_table.loc[sleep_initiation_table.between_time('03:30', '15:30').index, :] = 0.0
wake_table = data0.copy()
wake_table.loc[wake_table.between_time('15:30', '03:30').index, :] = 0.0

# Plotting the heatmaps
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
fig.suptitle("Normalized Sleep Initiation and Wake Time Distributions")
sns.heatmap(sleep_initiation_table.values, ax=axs[0], cmap="YlGnBu", cbar=True)
axs[0].set_title("Sleep Initiation Distribution by Hour")
sns.heatmap(wake_table.values, ax=axs[1], cmap="YlGnBu", cbar=True)
axs[1].set_title("Wake Time Distribution by Hour")
sns.heatmap(data0.values, ax=axs[2], cmap="YlGnBu", cbar=True)
axs[2].set_title("Overall Sleep Distribution by Hour")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("normalized_distributionHeat.png")

# Calculate resultant vectors for sleep initiation and wake clusters
angles_sleep = np.linspace(0, 2 * np.pi, len(sleep_initiation_table.index), endpoint=False)
angles_wake = np.linspace(0, 2 * np.pi, len(wake_table.index), endpoint=False)
magnitude_matrix_sleep = sleep_initiation_table.values.T
magnitude_matrix_wake = wake_table.values.T

# Initialize total vectors for sleep initiation and wake times
sleep_x, sleep_y, wake_x, wake_y = 0, 0, 0, 0
for day_idx, day in enumerate(magnitude_matrix_sleep):
    for hour_idx, magnitude in enumerate(day):
        angle = angles_sleep[hour_idx]
        sleep_x += magnitude * np.cos(angle)
        sleep_y += magnitude * np.sin(angle)
for day_idx, day in enumerate(magnitude_matrix_wake):
    for hour_idx, magnitude in enumerate(day):
        angle = angles_wake[hour_idx]
        wake_x += magnitude * np.cos(angle)
        wake_y += magnitude * np.sin(angle)

# Calculate magnitudes and angles for resultant vectors
sleep_magnitude = np.sqrt(sleep_x**2 + sleep_y**2)
wake_magnitude = np.sqrt(wake_x**2 + wake_y**2)
sleep_angle = np.arctan2(sleep_y, sleep_x)
wake_angle = np.arctan2(wake_y, wake_x)

# Calculate mean angles for sleep initiation and wake time based on max row sums
"""
row_sums_wake = wake_table.sum(axis=1)
row_sums_init = sleep_initiation_table.sum(axis=1)
closest_row_index_wake = (row_sums_wake - row_sums_wake.max()).abs().idxmin()
closest_row_index_init = (row_sums_init - row_sums_init.max()).abs().idxmin()
max_wake_time = wake_table.loc[closest_row_index_wake].index.hour + wake_table.loc[closest_row_index_wake].index.minute / 60
max_init_time = sleep_initiation_table.loc[closest_row_index_init].index.hour + sleep_initiation_table.loc[closest_row_index_init].index.minute / 60

max_wake_angle = 2 * np.pi * (max_wake_time / 24)
max_sleep_angle = 2 * np.pi * (max_init_time / 24)
"""
# Calculate mean angles for sleep initiation and wake time based on max row sums
row_sums_wake = wake_table.sum(axis=1)
row_sums_init = sleep_initiation_table.sum(axis=1)

# Get the closest row indices based on max row sums
closest_row_index_wake = row_sums_wake.idxmax()
closest_row_index_init = row_sums_init.idxmax()

# Extract hour and minute for the max wake and sleep initiation times
max_wake_time = closest_row_index_wake.hour + closest_row_index_wake.minute / 60
max_init_time = closest_row_index_init.hour + closest_row_index_init.minute / 60

# Calculate angles for these mean times
max_wake_angle = 2 * np.pi * (max_wake_time / 24)
max_sleep_angle = 2 * np.pi * (max_init_time / 24)


# Normalize magnitudes
max_magnitude = max(sleep_magnitude, wake_magnitude)
if max_magnitude > 0:  # Prevent division by zero
    sleep_magnitude /= max_magnitude
    wake_magnitude /= max_magnitude

# Function to add arrows
def add_arrow(ax, angle, magnitude, color, label):
    arrow = FancyArrow(
        angle, 0, 0, magnitude, width=0.02, head_width=0.1, head_length=0.1,
        length_includes_head=True, color=color, label=label
    )
    ax.add_patch(arrow)



def get_variance(data0):
    variance_collection = []
    for i,row in enumerate(data0.iterrows()):
        variance_collection.append(np.var(data0.iloc[i].values)) 
    return variance_collection
  
variance_collection = get_variance(data0)
#print(np.sum(variance_collection))     



fig = plt.figure()
plt.plot([i for i in range(0,len(variance_collection))], variance_collection)
plt.savefig("variance_collection.png")


fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# Scale and plot variance as an "orbiting ring" around the plot
variance_max = max(variance_collection)  # Find the maximum variance for normalization
#scaled_variance = [7+v / variance_max * (len(data0.columns) + 1) for v in variance_collection]  # Scale to fit outer ring
scaled_variance = [7+v / variance_max * (len(data0.columns) + 1) for v in variance_collection]  # Scale to fit outer ring

# Define positions for the variance ring (outermost radius)
theta_variance = np.linspace(0, 2 * np.pi, len(variance_collection), endpoint=True)


# plot data
theta, r = np.meshgrid(np.linspace(0,2*np.pi,len(data0)+1),np.arange(len(data0.columns)+1))
p = ax.pcolormesh(theta,r,data0.T.values, cmap="Reds")
fig.colorbar(p,ax=ax)

# set ticklabels
pos,step = np.linspace(0,2*np.pi,len(data0),endpoint=False, retstep=True)
pos += step/2.



ax.set_xticks(pos)
ax.set_xticklabels(data0.index.hour)

ax.set_yticks(np.arange(len(data0.columns)))
ax.set_yticklabels(data0.columns)


# Plot the variance as a line around the outermost radius
ax.plot(theta_variance, scaled_variance, color="purple", linewidth=2, label="Variance Ring")


# Plot all four vectors
add_arrow(ax, sleep_angle, sleep_magnitude*6.8, color='blue', label="Sleep Initiation Vector (Resultant)")
add_arrow(ax, wake_angle, wake_magnitude*6.8, color='green', label="Wake Time Vector (Resultant)")
#add_arrow(ax, max_sleep_angle, sleep_magnitude*6.8, color='cyan', label="Mean Sleep Initiation Time Vector")
#add_arrow(ax, max_wake_angle, wake_magnitude*6.8, color='orange', label="Mean Wake Time Vector")
#plt.savefig("for_README_exercise.png")

# Save the plot
plt.savefig("vector_plot_with_all_vectors.png")
#plt.show()




def get_variance(data0):
    variance_collection = []
    for i, row in enumerate(data0.iterrows()):
        variance_collection.append(np.var(data0.iloc[i].values))
    return variance_collection

variance_collection = get_variance(data0)

# Compute mean and standard deviation of variance for normalization
mean_var = np.mean(variance_collection)
std_var = np.std(variance_collection)

# Normalize variance as (x - mean) / std and add 7 for outer ring scaling
scaled_variance = [7.75 + (v - mean_var) / std_var for v in variance_collection]
# Duplicate the first value to the end to make the ring continuous
scaled_variance.append(scaled_variance[0])

# Define theta positions for the variance ring (outermost radius)
theta_variance = np.linspace(np.pi/24, 2 * np.pi+np.pi/24, len(scaled_variance), endpoint=True)

# Create the polar plot
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# Plot data as a polar heatmap
theta, r = np.meshgrid(np.linspace(0, 2 * np.pi, len(data0) + 1), np.arange(len(data0.columns) + 1))
p = ax.pcolormesh(theta, r, data0.T.values, cmap="Reds")
fig.colorbar(p, ax=ax)

# Set tick labels for hours and radial values
pos, step = np.linspace(0, 2 * np.pi, len(data0), endpoint=False, retstep=True)
pos += step / 2.
ax.set_xticks(pos)
ax.set_xticklabels(data0.index.hour)
ax.set_yticks(np.arange(len(data0.columns)))
ax.set_yticklabels(data0.columns)

# Plot the variance as a continuous ring around the outermost radius
ax.plot(theta_variance, scaled_variance, color="purple", linewidth=2, label="Variance Ring")

# Define function to add arrows for vectors
def add_arrow(ax, angle, magnitude, color, label):
    arrow = FancyArrow(
        angle, 0, 0, magnitude, width=0.02, head_width=0.1, head_length=0.1,
        length_includes_head=True, color=color, label=label
    )
    ax.add_patch(arrow)

# Plot all four vectors with adjusted magnitudes
add_arrow(ax, sleep_angle, sleep_magnitude * 6.8, color='blue', label="Sleep Initiation Vector (Resultant)")
add_arrow(ax, wake_angle, wake_magnitude * 6.8, color='green', label="Wake Time Vector (Resultant)")
# Uncomment if you need mean vectors as well
# add_arrow(ax, max_sleep_angle, sleep_magnitude * 6.8, color='cyan', label="Mean Sleep Initiation Time Vector")
# add_arrow(ax, max_wake_angle, wake_magnitude * 6.8, color='orange', label="Mean Wake Time Vector")

# Adjust legend placement outside the plot
ax.legend(loc="upper right", bbox_to_anchor=(-0.5, -0.8))

# Save the plot
plt.savefig("vector_plot_with_all_vectors_and_variance_ring.png")
plt.show()


def get_variance(data0):
    variance_collection = []
    for i, row in enumerate(data0.iterrows()):
        variance_collection.append(np.var(data0.iloc[i].values))
    return variance_collection

variance_collection = get_variance(data0)

# Compute mean and standard deviation of variance for normalization
mean_var = np.mean(variance_collection)
std_var = np.std(variance_collection)

# Normalize variance as (x - mean) / std and add 7 for outer ring scaling
scaled_variance = [7.75 + (v - mean_var) / std_var for v in variance_collection]
# Duplicate the first value to the end to make the ring continuous
scaled_variance.append(scaled_variance[0])

# Define theta positions for the variance ring (outermost radius)
theta_variance = np.linspace(0, 2 * np.pi, len(scaled_variance), endpoint=True)
#theta_variance = np.linspace(np.pi/24, 2 * np.pi+np.pi/24, len(scaled_variance), endpoint=True)

# Set up figure with two polar subplots: one for main plot, one for variance ring
fig = plt.figure(figsize=(8, 8))
ax_main = fig.add_subplot(111, projection='polar')
ax_variance = fig.add_subplot(111, projection='polar', frame_on=False)

# Main plot
ax_main.set_theta_zero_location("N")
ax_main.set_theta_direction(-1)

# Plot data as a polar heatmap on the main axis
theta, r = np.meshgrid(np.linspace(0, 2 * np.pi, len(data0) + 1), np.arange(len(data0.columns) + 1))
p = ax_main.pcolormesh(theta, r, data0.T.values, cmap="Reds")
fig.colorbar(p, ax=ax_main)

# Set tick labels for hours and radial values
pos, step = np.linspace(0, 2 * np.pi, len(data0), endpoint=False, retstep=True)
pos += step / 2.
ax_main.set_xticks(pos)
ax_main.set_xticklabels(data0.index.hour)
ax_main.set_yticks(np.arange(len(data0.columns)))
ax_main.set_yticklabels(data0.columns)

# Define function to add arrows for vectors
def add_arrow(ax, angle, magnitude, color, label):
    arrow = FancyArrow(
        angle, 0, 0, magnitude, width=0.02, head_width=0.1, head_length=0.1,
        length_includes_head=True, color=color, label=label
    )
    ax.add_patch(arrow)

# Plot all four vectors on the main plot
add_arrow(ax_main, sleep_angle, sleep_magnitude * 6.8, color='blue', label="Sleep Initiation Vector (Resultant)")
add_arrow(ax_main, wake_angle, wake_magnitude * 6.8, color='green', label="Wake Time Vector (Resultant)")
# Uncomment if you need mean vectors as well
# add_arrow(ax_main, max_sleep_angle, sleep_magnitude * 6.8, color='cyan', label="Mean Sleep Initiation Time Vector")
# add_arrow(ax_main, max_wake_angle, wake_magnitude * 6.8, color='orange', label="Mean Wake Time Vector")

# Configure the variance ring plot
#ax_variance.set_ylim(7.5, 8.5)  # Set radius range outside main plot area
ax_variance.plot(theta_variance, scaled_variance, color="purple", linewidth=2, label="Variance Ring")

# Hide polar grid and ticks in variance plot to make it appear as an orbit
ax_variance.grid(False)
ax_variance.set_xticklabels([])
ax_variance.set_yticklabels([])

# Move legend outside of main plot
ax_main.legend(loc="upper right", bbox_to_anchor=(1.3, 1.3))

# Save the plot
plt.savefig("vector_plot_with_all_vectors_and_variance_ring.png")
plt.show()

# Functions for data processing
def get_variance(data0):
    variance_collection = []
    for i, row in enumerate(data0.iterrows()):
        variance_collection.append(np.var(data0.iloc[i].values))
    return variance_collection

# Calculate normalized variance
variance_collection = get_variance(data0)
mean_var = np.mean(variance_collection)
std_var = np.std(variance_collection)

# Scale variance to start at 7 with normalized values
scaled_variance = [3.5 + (v - mean_var) / std_var for v in variance_collection]
scaled_variance.append(scaled_variance[0])  # To close the circle

# Define theta positions for the variance ring
theta_variance = np.linspace(np.pi/24, 2 * np.pi+ np.pi/24, len(scaled_variance), endpoint=True)

# Set up figure with two polar subplots
fig = plt.figure(figsize=(14, 7))
ax_main = fig.add_subplot(121, projection='polar')  # Main polar plot on the left
ax_variance = fig.add_subplot(122, projection='polar')  # Polar plot for variance on the right

# Main polar heatmap with vector arrows on the left
ax_main.set_theta_zero_location("N")
ax_main.set_theta_direction(-1)
theta, r = np.meshgrid(np.linspace(0, 2 * np.pi, len(data0) + 1), np.arange(len(data0.columns) + 1))
p = ax_main.pcolormesh(theta, r, data0.T.values, cmap="Reds")
fig.colorbar(p, ax=ax_main)

# Set tick labels for hours and radial values
pos, step = np.linspace(0, 2 * np.pi, len(data0), endpoint=False, retstep=True)
pos += step / 2.
ax_main.set_xticks(pos)
ax_main.set_xticklabels(data0.index.hour)
ax_main.set_yticks(np.arange(len(data0.columns)))
ax_main.set_yticklabels(data0.columns)

# Define function to add arrows for vectors
def add_arrow(ax, angle, magnitude, color, label):
    arrow = FancyArrow(
        angle, 0, 0, magnitude, width=0.02, head_width=0.1, head_length=0.1,
        length_includes_head=True, color=color, label=label
    )
    ax.add_patch(arrow)

# Plot all four vectors on the main plot
add_arrow(ax_main, sleep_angle, sleep_magnitude * 6.8, color='blue', label="Sleep Initiation Vector (Resultant)")
add_arrow(ax_main, wake_angle, wake_magnitude * 6.8, color='green', label="Wake Time Vector (Resultant)")

# Adjust legend placement outside the main plot
ax_main.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

# Right subplot: Polar plot for variance with scaled values starting from 7
ax_variance.set_theta_zero_location("N")
ax_variance.set_theta_direction(-1)
ax_variance.plot(theta_variance, scaled_variance, color="purple", linewidth=2, label="Variance Over Time (Polar)")
#ax_variance.set_ylim(7, max(scaled_variance) + 1)  # Radius starts from 7




ax_variance.set_xticks(pos)
ax_variance.grid(False)  # Hide grid
#ax_variance.set_xticks([])  # Remove angular (theta) ticks
ax_variance.set_yticks([])  # Remove radial ticks
#ax_variance.spines['polar'].set_visible(False)  # Hide the polar spine

# Customize ticks for variance polar plot
ax_variance.set_xticklabels(data0.index.hour)  # Match hour labels to main plot
#ax_variance.set_yticks([7, 8, 9])  # Example radial ticks
#ax_variance.set_yticklabels(["Low", "Medium", "High"])  # Custom labels for variance range
ax_variance.legend(loc="upper right")#, bbox_to_anchor=(1.1, 1.1))

# Save and display the plot
plt.tight_layout()
plt.savefig("polar_plots_with_variance_ring.png")
plt.show()