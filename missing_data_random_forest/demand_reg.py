import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from scipy import stats
import glob
import holidays
from calendar import monthrange
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.multioutput import RegressorChain
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from numpy import absolute, mean, std
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV

warnings.filterwarnings('ignore')

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 12)

dateparse = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M')
data = pd.read_csv("/data/primary_demand_combined_csv.csv",
                   parse_dates=['datetime'], date_parser=dateparse)

df = pd.DataFrame(data)
df.set_index('datetime', drop=True, inplace=True)
print('df size before dropping na:', df.shape, '\n')

# Plot skewness of data
# graph_avon = sns.distplot(df['avonmouth'], color="dodgerblue",
#                          label="Skewness : %.2f" % (df['avonmouth'].skew()))
# graph_avon = graph_avon.legend(loc="best")


# Remove NA values
na_drop_df = df.dropna(axis=0)
print('df size after dropping na:', na_drop_df.shape,
      f'a total of {df.shape[0] - na_drop_df.shape[0]} rows have been lost')


# Remove outliers either using the z-score or the IQR
def remove_outliers(df, method='LOF & IQR', threshold=3):
    if method == 'Z Score':  # This is where all data points lying outside 3 standard deviations are removed
        df = df[(np.abs(stats.zscore(df)) < threshold).all(axis=1)]
    elif method == 'IQR':
        cols = df.columns
        Q1 = df[cols].quantile(0.20)
        Q3 = df[cols].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[cols] < (Q1 - 1.5 * IQR)) | (df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    elif method == 'IsoF':
        # identify outliers in the dataset
        iso = IsolationForest(n_estimators=100, contamination=0.20, max_features=df.shape[1])
        outliers_predicted = iso.fit_predict(df)
        df['outlier'] = outliers_predicted
        df = df[df['outlier'] == 1]
        df.drop('outlier', axis=1, inplace=True)
    elif method == 'LOF & IQR':
        cols = df.columns
        Q1 = df[cols].quantile(0.15)
        Q3 = df[cols].quantile(0.90)
        IQR = Q3 - Q1
        df = df[~((df[cols] < (Q1 - 1.5 * IQR)) | (df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
        lof = LocalOutlierFactor(contamination=0.4)
        outliers_predicted = lof.fit_predict(df)
        df['outlier'] = outliers_predicted
        df = df[df['outlier'] == 1]
        df.drop('outlier', axis=1, inplace=True)
    else:
        df = df

    return df, method


fig, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
df.plot(legend=False, cmap='viridis', ax=ax[0], title='Before NA Removal')
na_drop_df.plot(legend=False, cmap='viridis', ax=ax[1], title='Before Outlier Removal')
cleaned_df, method = remove_outliers(df=na_drop_df)
cleaned_df.plot(legend=False, cmap='viridis', ax=ax[2], title=f'After Outlier Removal using the {method} method')
plt.xlim(df.index[0], df.index[-1])

print('df size after removing outliers:', cleaned_df.shape,
      f'a total of {na_drop_df.shape[0] - cleaned_df.shape[0]} rows have been lost')


# Time to import the weather data near the feeder
def add_weather_data(path=r'weather_data/'):
    # get data file names
    filenames = glob.glob(path + "/*.csv")

    dfs = []
    for filename in filenames:
        weather_df = pd.read_csv(filename, header=0)
        dfs.append(weather_df)

    # Concatenate all data into one DataFrame
    big_frame = pd.concat(dfs, axis=0, ignore_index=True)
    avg_weather = big_frame.groupby(by=['datetime']).mean().reset_index()

    # Convert the date/time columns to datetime
    avg_weather['datetime'] = avg_weather['datetime'].apply(pd.to_datetime, format='%Y/%m/%d %H:%M:%S')
    avg_weather.set_index('datetime', drop=True, inplace=True)

    # Resample the hourly data to half-hourly and interpolate between the hours
    avg_weather = avg_weather.resample('30T').interpolate().reset_index()

    return avg_weather


avg_weather = add_weather_data()

# Merge the weather and the demand dataframes on the datetime column for both the original and cleaned df
final_df = pd.merge(cleaned_df, avg_weather, how='inner', left_on='datetime', right_on='datetime')
original_df = pd.merge(df, avg_weather, how='inner', left_index=True, right_on='datetime')


# Convert the date column to useful features for the model
def expand_datetime(df):
    data = df.copy()
    data['year'] = data['datetime'].dt.year
    data['season'] = data['datetime'].dt.month % 12 // 3 + 1
    data['is_winter'] = [1 if val == 4 in data['season'] else 0 for val in data['season']]
    data['month'] = data['datetime'].dt.month
    data['week_of_year'] = data['datetime'].dt.isocalendar().week.astype('int')
    data['day'] = data['datetime'].dt.day
    data['dayofweek'] = data['datetime'].dt.dayofweek
    data['hour'] = data['datetime'].dt.hour
    data['minute'] = data['datetime'].dt.minute
    data["is_wknd"] = np.where(data['datetime'].dt.weekday >= 5, 1, 0)
    data["is_working_hr"] = np.where(((data['hour'] >= 8) & (data['hour'] <= 19)), 1, 0)
    data["is_lunch_hr"] = np.where(data['hour'] == 13, 1, 0)

    uk_holidays = holidays.UnitedKingdom()
    data['Holiday'] = [1 if str(val).split()[0] in uk_holidays else 0 for val in data['datetime']]

    # 2D time conversion
    data['days_in_month'] = monthrange(data['year'].all(), data['month'].all())[1]
    data['hourmin'] = data['hour'] + (data['minute'] / 60)
    data['hour_x'] = np.sin((360 / 24) * data['hourmin'])
    data['hour_y'] = np.cos((360 / 24) * data['hourmin'])
    data['day_x'] = np.sin((360 / data['days_in_month']) * data['day'])
    data['day_y'] = np.cos((360 / data['days_in_month']) * data['day'])
    data['month_x'] = np.sin((360 / 12) * data['month'])
    data['month_y'] = np.cos((360 / 12) * data['month'])

    data.drop(['days_in_month'], axis=1, inplace=True)

    return data


def expand_weather(df):
    data = df.copy()
    # # Get the temperature 24 hour into the future as another feature
    data['temperature_+24h'] = data['temperature'].shift(periods=-48)
    # Get the temperature 12 hour into the future as another feature
    data['temperature_+12h'] = data['temperature'].shift(periods=-24)
    # Get the temperature 1 hour into the future as another feature
    data['temperature_+1h'] = data['temperature'].shift(periods=-2)
    # Get the temperature 30 mins into the future as another feature
    data['temperature_+30m'] = data['temperature'].shift(periods=-1)

    # Get the average temperature over the past day
    data['daily_avg_temp-24h'] = data.rolling(48, min_periods=48)['temperature'].mean()

    # Get the average temperature over the next day
    data['daily_avg_temp+24h'] = data['temperature'].shift(periods=-48).rolling(48, 48).mean()

    # Get the max temperature over the next day
    data['daily_max_temp+24h'] = data['temperature'].shift(periods=-48).rolling(48, 48).max()

    # Get the min temperature over the next day
    data['daily_min_temp+24h'] = data['temperature'].shift(periods=-48).rolling(48, 48).min()

    temp_diff = np.diff(data["temperature"], n=1)
    data["temperature_diff1"] = np.append([0], temp_diff)

    data.dropna(subset=['temperature_+1h', 'temperature_+30m', 'temperature_+12h', 'temperature_+24h',
                        'daily_avg_temp-24h',
                        'daily_avg_temp+24h', 'daily_max_temp+24h', 'temperature_diff1',
                        'daily_min_temp+24h'], axis=0, inplace=True)

    return data

final_df = expand_datetime(final_df)
original_df = expand_datetime(original_df)
final_df = expand_weather(final_df)
original_df = expand_weather(original_df)

print(final_df.columns)

X_cols = ['temperature', 'solar_irradiance', 'windspeed_north', 'windspeed_east',
          'season', 'is_winter', 'month', 'week_of_year', 'day', 'dayofweek', 'hour', 'is_wknd',
          'is_working_hr', 'hourmin', 'month_x', 'month_y', 'temperature_+1h', 'temperature_+30m',
          'daily_avg_temp-24h', 'daily_avg_temp+24h','daily_max_temp+24h', 'daily_min_temp+24h', 'avonmouth',
          'eastville', 'clifton', 'bedminster']
y_cols = ['bishopsworth', 'cairns_road', 'lockleaze', 'stock_bishop', 'st_pauls_j', 'filton_j']

X = final_df[X_cols]
y = final_df[y_cols]

# Split dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=42, shuffle=False)

model = ExtraTreesRegressor(n_estimators=50, random_state=42)

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
multi_output_model = MultiOutputRegressor(estimator=model)
min_features_to_select = 1  # Minimum number of features to consider
selector = RFECV(model, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1,
                 min_features_to_select=min_features_to_select)
selector = selector.fit(X_train, y_train)
print(selector.support_)

print("Optimal number of features : %d" % selector.n_features_)
print("Ranking of features : ", selector.ranking_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (RMSE)")
plt.plot(
    range(min_features_to_select, len(selector.grid_scores_) + min_features_to_select),
    abs(selector.grid_scores_))

n_scores = cross_val_score(multi_output_model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=cv,
                           n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)

# summarize performance
print('Average RMSE: %.3f with a standard deviation of (%.3f)' % (mean(n_scores), std(n_scores)))

# Fit model to training data
multi_output_model.fit(X_train, y_train)

# make predictions for the missing 24 hr period and a couple days after
pred_range = original_df.loc[
    (original_df['datetime'] >= '2020-03-03 00:30:00') & (original_df['datetime'] <= '2020-03-06 00:00:00')]
pred_range = pred_range.drop('datetime', axis=1)
X_rows = pred_range[X_cols]

yhat = multi_output_model.predict(X_rows)
pred = yhat[:, 5]

# Forecast for 3 days
forecast_window = 48 * 3
forecast_period_dates = pd.date_range(start=pd.to_datetime('2020-03-01 00:00:00'), periods=forecast_window,
                                      freq='30T').tolist()

# # Obtain the values for before the missing data period
# before = original_df.loc[
#     (original_df['datetime'] < '2021-01-11 00:00:00') & (original_df['datetime'] > '2021-01-07 00:00:00')]
# before_y = before[['datetime', y_cols[0]]]

# Actual values compared to predictions for the missing 24 hr period and a couple days after
after = original_df.loc[
    (original_df['datetime'] < '2020-03-06 00:00:00') & (original_df['datetime'] >= '2020-03-03 00:30:00')]
after_y = after[['datetime', y_cols[0]]]

fig3, ax = plt.subplots(1, 1, figsize=(16, 10))
sns.lineplot(x=forecast_period_dates, y=pred, ax=ax, label='Prediction', color='gold')
sns.lineplot(x=after_y['datetime'], y=after_y[y_cols[0]], ax=ax, label='Actual', color='black', alpha=0.4)
ax.set_title("Scenario 1")


# Scenario Predictions
# Scenario Date Ranges
scenario_dates = {"Scenario": ["Scenario 1"],
                  "start_date": ['2020-01-01 00:00:00'],
                  "end_date": ['2020-06-10 00:00:00']}

scenarios = []
for (scenario, start, end) in zip(scenario_dates["Scenario"], scenario_dates["start_date"], scenario_dates["end_date"]):
    scenario_range = original_df.loc[(original_df['datetime'] >= start) & (original_df['datetime'] <= end)]
    scenario_range = scenario_range.drop('datetime', axis=1)
    scenario_X = scenario_range[X_cols]
    print(scenario, f"length = {scenario_X.shape[0]} half-hours")
    scenario_y = multi_output_model.predict(scenario_X)
    scenarios.append(scenario_y)

scenarios_df = pd.DataFrame(np.concatenate(scenarios), columns=y_cols)
scenarios_df['datetime'] = pd.date_range(start='2020-01-01 00:00:00',
                                         end='2020-06-10 00:00:00',
                                         freq='30T')

scenarios_df.to_csv(path_or_buf='model_results.csv')




