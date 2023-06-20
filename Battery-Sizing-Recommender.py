#!/usr/bin/env python
# coding: utf-8

# ## Battery Suggestion Based on the Total Consumption per Day
# 
# This code analyzes customer consumption data and energy prices to suggest the optimal battery size. It begins by reading consumption data from an Excel file and prompts the user to select a price source (Easy or Entsoe). The code then reads the corresponding price data and performs data processing steps.
# 
# First, the code calculates the average consumption per hour on weekdays, considering only Monday to Friday data. It also calculates the average price per hour during the same period. The code then identifies the peak positive and negative consumption hours and calculates the average and total consumption during peak and off-peak hours.
# 
# Next, the code defines a function to simulate different battery sizes and calculates the associated costs. It performs simulations for a range of suggested battery sizes and determines the best battery size based on the lowest total cost.
# 
# The code provides insights into the average and total consumption during peak and off-peak hours. It also suggests the best battery size for customers, considering their consumption patterns and energy prices.

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import calendar
import holidays


# ### File paths
# 
# It is necessary for the user to input the file path of the consumption data.
# 
# **IMPORTANT**
# 
# The consumption data file must have at least the following columns:
# 
# - Datetime
# - Consumption
# - Operating Hours### Paths

# In[2]:


# Consumption Data
consumption_data = input("Enter the consumption data file path (CSV): ")
#consumption_data = 'C:/Users/Diego Galdino/Desktop/New folder/Consumption2.xlsx'

# Price Data
price_Easy = 'C:/Users/Diego Galdino/OneDrive - Groene Cadans B.V/Algorithms/Energy-Price-Easy/EasyEnergyPrice.csv'
price_Entosoe = 'C:/Users/Diego Galdino/OneDrive - Groene Cadans B.V/Algorithms/Energy-Price-ENTSOE/EntsoeEnergyPrice.csv'


# ### Read the files

# In[3]:


# Read consumption data
df_consumption = pd.read_excel(consumption_data)

while True:
    # Select price source
    price_source = input("Select the price source:\n1 - Easy\n2 - Entsoe\n")

    # Read price data based on the selected source
    if price_source == "1":
        df_price = pd.read_csv(price_Easy)
        break
    elif price_source == "2":
        df_price = pd.read_csv(price_Entosoe)
        break
    else:
        print("Invalid selection. Try again.")

# Restrict to required columns
required_columns = ['Datetime', 'Consumption', 'Operating Hours']
df_consumption = df_consumption[required_columns]

# Check if the required columns are present
missing_columns = set(required_columns) - set(df_consumption.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")


# ### Adjusting the Consumption Data

# In[4]:


# Create a DataFrame copy for NOT change the originial
data_C = df_consumption.copy()

# Group by weekday
data_C['Weekday'] = data_C['Datetime'].dt.weekday

# Calculate average consumption per hour from Monday to Friday
average_consumption_weekday = data_C.loc[data_C['Weekday'] < 5].groupby(data_C['Datetime'].dt.hour)['Consumption'].mean()* 4

# Create a list of all hours from 0 to 23
all_hours = range(24)

# Convert average_consumption_weekday to a DataFrame
average_consumption_weekday = average_consumption_weekday.to_frame()

# Reset the index to get the hour values as a column
average_consumption_weekday = average_consumption_weekday.reset_index()

# Rename the columns to "Datetime" and "Consumption"
average_consumption_weekday.columns = ["Datetime", "Consumption"]

# Fill missing hours with 0 average consumption
average_consumption_weekday = average_consumption_weekday.reindex(all_hours, fill_value=0)


# ### Adjusting the Price Data

# In[5]:


# Create a copy of the DataFrame
data_P = df_price.copy()

# Create a list of all weekdays
weekdays = range(0, 5)  # Monday to Sunday

# Convert the "Date" and "Hour" columns to datetime
data_P['Datetime'] = pd.to_datetime(data_P['Date'] + ' ' + data_P['Hour'], format='%Y-%m-%d %H:%M')

# Group by weekday
data_P['Weekday'] = data_P['Datetime'].dt.weekday

# Calculate average price per hour from Monday to Friday
average_price_weekday = data_P.loc[data_P['Weekday'] < 5].groupby(data_P['Datetime'].dt.hour)['Import Grid (EUR/kWh)'].mean()

# Create a list of all hours from 0 to 23
all_hours = range(24)

# Convert average_price_weekday to a DataFrame
average_price_weekday = average_price_weekday.to_frame()

# Reset the index to get the hour values as a column
average_price_weekday = average_price_weekday.reset_index()

# Rename the columns to "Datetime" and "Import Grid (EUR/kWh)"
average_price_weekday.columns = ["Datetime", "Import Grid (EUR/kWh)"]

# Fill missing hours with 0 average price
average_price_weekday = average_price_weekday.reindex(all_hours, fill_value=0)
average_price_weekday = average_price_weekday.reindex(all_hours, fill_value=0)


# ### Calculating the Total and Average Consumption

# In[6]:


# Calculate the difference between each hour and its previous hour
hourly_diff = average_consumption_weekday['Consumption'].diff()

# Find the index of the peak positive and negative values
peak_positive_index = np.argmax(hourly_diff)

# Get the corresponding hour value
peak_positive_hour = average_consumption_weekday['Datetime'].iloc[peak_positive_index]

# Save the last value before the peak positive hour
last_value_before_peak = average_consumption_weekday['Consumption'].iloc[peak_positive_index -2]

# Define the peak negative threshold as 50% above the last value before the peak
peak_negative_threshold = last_value_before_peak * 1.1

# Find the hour when the consumption returns to the threshold
peak_negative_hour = average_consumption_weekday.loc[average_consumption_weekday['Consumption'] >= peak_negative_threshold, 'Datetime'].idxmax()

# Format the peak positive and negative hours as time
peak_positive_hour_formatted = datetime.strptime(str(peak_positive_hour), "%H").strftime("%H:%M")
peak_negative_hour_formatted = datetime.strptime(str(peak_negative_hour), "%H").strftime("%H:%M")


# In[7]:


# Filter the consumption data between the peak positive and negative hours
peak_consumption = average_consumption_weekday.loc[(average_consumption_weekday['Datetime'] >= peak_positive_hour) & (average_consumption_weekday['Datetime'] < peak_negative_hour), 'Consumption']

# Filter the consumption data out the peak positive and negative hours
off_peak_consumption = average_consumption_weekday.loc[(average_consumption_weekday['Datetime'] >= peak_negative_hour) | (average_consumption_weekday['Datetime'] < peak_positive_hour), 'Consumption']

# Average and Total consumption during peak hours
average_peak_consumption = peak_consumption.mean()
total_peak_consumption = peak_consumption.sum()

# Average and Total consumption outside peak hours
average_off_peak_consumption = off_peak_consumption.mean()
total_off_peak_consumption = off_peak_consumption.sum()

# Print the average peak consumption with two decimal places
print("Average Peak Consumption: {:.2f}".format(average_peak_consumption))
print("Average Out Peak Consumption: {:.2f}".format(average_off_peak_consumption))
print("\n")

# Print the total peak consumption with two decimal places
print("Total Peak Consumption: {:.2f}".format(total_peak_consumption))
print("Total Out Peak Consumption: {:.2f}".format(total_off_peak_consumption))


# ### Simulations for Decide the best Battery Size

# In[8]:


# Function to simulate different battery sizes
def simulate_battery_size(battery_size):
    excess_consumption = total_peak_consumption - battery_size
    if excess_consumption > 0:
        battery_consumption = excess_consumption
    else:
        battery_consumption = 0
    battery_cost = battery_consumption * average_price_weekday['Import Grid (EUR/kWh)'].values[0]
    return battery_consumption, battery_cost

# Simulate different battery sizes and calculate the costs
suggested_battery_sizes = list(range(0, 5000, 100))  # Example suggested battery sizes in kWh
costs = []
simulations = []
for size in suggested_battery_sizes:
    battery_consumption, total_cost = simulate_battery_size(size)
    costs.append(total_cost)
    simulations.append((size, battery_consumption, total_cost))

# Determine the best battery size based on the lowest total cost
best_size = suggested_battery_sizes[np.argmin(costs)]

# Show the simulations and how we arrived at the value of the best battery size
print("Simulations:")
for size, battery_consumption, total_cost in simulations:
    print("Battery Size:", size, "kWh")
    print("Battery Consumption:", battery_consumption, "kWh")
    print("Total Cost:", total_cost, "EUR")
    print()


# In[9]:


print("The best battery size is:", best_size, "kWh")

