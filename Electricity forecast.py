# Import necessary libraries
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np 
import pandas as pd
from pandas import DataFrame
from math import sqrt
from sklearn.metrics import mean_squared_error
import pmdarima as pmd
from statsmodels.tsa.arima.model import ARIMA


#ARIMA

# Read the CSV file containing the data
df = pd.read_csv(
    'C:/Users/ratan/Downloads/KwhConsumptionBlower78_2.csv',
    names=['id', 'date', 'time', 'consumption'],  # Specify column names
    parse_dates={'datetime': ['date', 'time']},   # Combine date and time columns into datetime
    index_col='id',                               # Set 'id' column as the index
    header=0,                                     # Specify that column names are in the first row
    date_parser=lambda x, y: pd.to_datetime(x + ' ' + y, format='%d %b %Y %H:%M:%S')  # Custom date parser
)

# Display the first few rows of the DataFrame
print(df.head())

# Print information about the date range of the data
print(f"min date: {df['datetime'].min()}, max date: {df['datetime'].max()}")
print(f"range: {df['datetime'].max() - df['datetime'].min()}")

# Sort the DataFrame by datetime
df_sorted = df.sort_values('datetime')

# Calculate daily consumption using rolling sum
daily_cons = df_sorted.rolling('1D', on='datetime').sum()

# Group by month and calculate total consumption, average consumption, and count
monthly_stats = daily_cons.groupby(daily_cons['datetime'].dt.month)['consumption'].agg(['sum', 'mean', 'count'])

# Display statistics for monthly consumption
print(monthly_stats)

# Plot daily energy consumption
daily_cons.plot(x='datetime', y='consumption', title="Daily energy consumption", figsize=(10, 6))
plt.show()

# Describe the daily consumption data
print(daily_cons.describe())

# Filter for day consumption (from 06:00 to 18:00)
day_consumption = df[(df['datetime'].dt.time >= pd.Timestamp('06:00').time()) &
                     (df['datetime'].dt.time < pd.Timestamp('18:00').time()) &
                     (df['consumption'] > 0.5)]

# Filter for night consumption (outside of 06:00 to 18:00)
night_consumption = df[~df.index.isin(day_consumption.index) &
                       (df['consumption'] > 0.5)]

# Sort night consumption by datetime
night_consumption = night_consumption.sort_values('datetime')

# Sort day consumption DataFrame by datetime
day_consumption = day_consumption.sort_values('datetime')

# Calculate hourly consumption using rolling sum
day_hourly_cons = day_consumption.rolling('1H', on='datetime').sum()
night_hourly_cons = night_consumption.rolling('1H', on='datetime').sum()

# Plot hourly energy consumption
fig, ax = plt.subplots(1, 1, figsize=(20, 6))
ax.plot(day_hourly_cons['datetime'], day_hourly_cons['consumption'], 'g-', label='Day Consumption')
ax.plot(night_hourly_cons['datetime'], night_hourly_cons['consumption'], 'r-', label='Night Consumption')
ax.legend()
ax.set_title("Hourly Energy Consumption")
ax.set_xlabel("Date")
ax.set_ylabel("Energy Consumption")
plt.show()

# Calculate daily consumption for day and night
daily_day_cons = day_consumption.rolling('1d', on='datetime').sum()
daily_night_cons = night_consumption.rolling('1d', on='datetime').sum()

# Plot daily energy consumption for day and night
fig, ax = plt.subplots(1, 1, figsize=(15, 7))
ax.plot(daily_day_cons['datetime'], daily_day_cons['consumption'], 'g-', label='Day Consumption')
ax.plot(daily_night_cons['datetime'], daily_night_cons['consumption'], 'r-', label='Night Consumption')
ax.legend()
ax.set_title("Daily Energy Consumption")
ax.set_xlabel("Date")
ax.set_ylabel("Energy Consumption")
plt.show()

# Ensure datetime is set as the index and sorted
df_copy = df.set_index('datetime').sort_index()

# Resample the DataFrame to daily frequency and calculate sum for each day
df_daily = df_copy.resample('D').sum()

# Perform seasonal decomposition
result = seasonal_decompose(df_daily, model='additive', period=5)  # Adjust the period accordingly

# Plot the decomposed components
plt.rc("figure", figsize=(15, 6))
result.plot()
plt.show()

# Perform multiplicative decomposition
result = seasonal_decompose(df_daily, model='multiplicative', period=5)  # Adjust the period accordingly

# Plot the decomposed components
plt.rc("figure", figsize=(15, 6))
result.plot()
plt.show()

# Automatically select ARIMA model using pmdarima
model = pmd.auto_arima(df_copy, start_p=1, start_q=1, test='adf', m=12, seasonal=True, trace=True)

# Fit ARIMA model
model = ARIMA(df_copy, order=(2, 0, 0))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Calculate residuals
residuals = pd.DataFrame(model_fit.resid)

# Plot residuals
residuals.plot()
plt.show()

# Plot density plot of residuals
residuals.plot(kind='kde')
plt.show()

# Plot Actual vs Fitted
plt.plot(df_copy,color='red',label='Actual')
plt.plot(model_fit.fittedvalues, color='green',label='Fitted')
plt.legend()
plt.show()

# Split data into train and test sets
X = df_copy.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]

# Walk-forward validation
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    model = ARIMA(history, order=(2, 0, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# Calculate RMSE
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
mse = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % mse)

# Plot actual vs predicted
plt.plot(test,color='red', label='Actual')
plt.plot(predictions, color='green', label='Predicted')
plt.legend()
plt.show()


#SARIMA



# Fit SARIMA model
sarima_model = SARIMAX(df_copy, order=(2, 0, 0), seasonal_order=(1, 1, 1, 12))
sarima_model_fit = sarima_model.fit()

# Print SARIMA model summary
print(sarima_model_fit.summary())

# Calculate residuals
sarima_residuals = pd.DataFrame(sarima_model_fit.resid)

# Plot residuals
sarima_residuals.plot()
plt.show()

# Plot density plot of residuals
sarima_residuals.plot(kind='kde')
plt.show()

# Plot Actual vs Fitted
plt.plot(df_copy, color='red', label='Actual')
plt.plot(sarima_model_fit.fittedvalues, color='green', label='Fitted')
plt.legend()
plt.show()

# Split data into train and test sets
X = df_copy.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]

# Walk-forward validation
history = [x for x in train]
sarima_predictions = list()

for t in range(len(test)):
    sarima_model = SARIMAX(history, order=(2, 0, 0), seasonal_order=(1, 1, 1, 12))
    sarima_model_fit = sarima_model.fit(disp=0)
    output = sarima_model_fit.forecast()
    yhat = output[0]
    sarima_predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# Calculate RMSE
sarima_rmse = sqrt(mean_squared_error(test, sarima_predictions))
print('Test RMSE: %.3f' % sarima_rmse)

# Calculate MSE
sarima_mse = mean_squared_error(test, sarima_predictions)
print('Test MSE: %.3f' % sarima_mse)

# Plot actual vs predicted
plt.plot(test, color='red', label='Actual')
plt.plot(sarima_predictions, color='green', label='Predicted')
plt.legend()
plt.show()



#ARCH model



from arch import arch_model

df = pd.read_csv('C:/Users/ratan/Downloads/KwhConsumptionBlower78_2.csv',
                 names=['id', 'date', 'time', 'consumption'],  # Specify column names
                 parse_dates={'datetime': ['date', 'time']},   # Combine date and time columns into datetime
                 header=0,                                    # Specify that column names are in the first row
                 date_parser=lambda x, y: pd.to_datetime(x + ' ' + y, format='%d %b %Y %H:%M:%S')  # Custom date parser
                 )

# Reset index to avoid duplicate labels
df.reset_index(drop=True, inplace=True)

# Calculate returns as percentage change in consumption
df['returns'] = df['consumption'].pct_change()

# Drop rows with NaN values
df.dropna(inplace=True)

# Calculate squared returns as proxy for volatility
df['returns_squared'] = df['returns'] ** 2

# Fit ARCH model
model = arch_model(df['returns_squared'], vol='ARCH', p=1)
arch_model_fit = model.fit()

# Print model summary
print(arch_model_fit.summary())

# Plot standardized residuals
arch_model_fit.plot()

# Calculate ARCH model predictions
arch_predictions = arch_model_fit.conditional_volatility

# Calculate MSE and RMSE
arch_mse = np.mean((df['returns_squared'] - arch_predictions) ** 2)
arch_rmse = np.sqrt(arch_mse)
print('RMSE: %.3f' % arch_rmse)
print('MSE: %.3f' % arch_mse)



#GARCH model



df = pd.read_csv(
    'C:/Users/ratan/Downloads/KwhConsumptionBlower78_2.csv',
    names=['id', 'date', 'time', 'consumption'],  # Specify column names
    parse_dates={'datetime': ['date', 'time']},   # Combine date and time columns into datetime
    header=0,                                     # Specify that column names are in the first row
    date_parser=lambda x, y: pd.to_datetime(x + ' ' + y, format='%d %b %Y %H:%M:%S')  # Custom date parser
)

# Reset index to avoid duplicate labels
df.reset_index(drop=True, inplace=True)

# Calculate returns as percentage change in consumption
df['returns'] = df['consumption'].pct_change()

# Drop rows with NaN values
df.dropna(inplace=True)

# Fit GARCH model
model = arch_model(df['returns'], vol='GARCH', p=1, q=1)
garch_model_fit = model.fit()

# Print model summary
print(garch_model_fit.summary())

# Plot standardized residuals
garch_model_fit.plot()

# Calculate GARCH model predictions
garch_predictions = garch_model_fit.conditional_volatility

# Calculate MSE and RMSE
garch_mse = np.mean((df['returns'] - garch_predictions) ** 2)
garch_rmse = np.sqrt(garch_mse)
print('RMSE: %.3f' % garch_rmse)
print('MSE: %.3f' % garch_mse)




#state-space model



import statsmodels.api as sm

# Read the CSV file containing the data
df = pd.read_csv(
    'C:/Users/ratan/Downloads/KwhConsumptionBlower78_2.csv',
    names=['id', 'date', 'time', 'consumption'],  # Specify column names
    parse_dates={'datetime': ['date', 'time']},   # Combine date and time columns into datetime
    header=0,                                     # Specify that column names are in the first row
    date_parser=lambda x, y: pd.to_datetime(x + ' ' + y, format='%d %b %Y %H:%M:%S')  # Custom date parser
)

# Reset index to avoid duplicate labels
df.reset_index(drop=True, inplace=True)

# Calculate returns as percentage change in consumption
df['returns'] = df['consumption'].pct_change()

# Drop rows with NaN values
df.dropna(inplace=True)

# Define the state-space model
mod = sm.tsa.statespace.SARIMAX(df['consumption'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

# Fit the model
res = mod.fit(disp=False)

# Print model summary
print(res.summary())

# Plot the observed and fitted values
fig, ax = plt.subplots(figsize=(15, 7))
df['consumption'].plot(ax=ax, label='Observed')
res.fittedvalues.plot(ax=ax, label='Fitted')
ax.set_title('Observed vs Fitted')
ax.legend()
plt.show()

# Plot the residuals
res.plot_diagnostics(figsize=(15, 10))
plt.show()
# Calculate SARIMA model predictions
sarima_predictions = res.fittedvalues

# Calculate MSE and RMSE
sarima_mse = np.mean((df['consumption'] - sarima_predictions) ** 2)
sarima_rmse = np.sqrt(sarima_mse)
print('RMSE: %.3f' % sarima_rmse)
print('MSE: %.3f' % sarima_mse)
