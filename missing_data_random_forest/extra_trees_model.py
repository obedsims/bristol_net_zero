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

dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data = pd.read_csv("C:/Users/osims/PycharmProjects/SolarProject/WPD_Round_3/Data/phase_1_training.csv",
                   parse_dates=['datetime'], date_parser=dateparse)

df = pd.DataFrame(data)
df.set_index('datetime', drop=True, inplace=True)
print('df size before dropping na:', df.shape, '\n')

# Plot skewness of data
graph_GSP = sns.distplot(df['Iron Acton (Agg incl Gen)_GSP'], color="dodgerblue",
                         label="Skewness : %.2f" % (df['Iron Acton (Agg incl Gen)_GSP'].skew()))
graph_GSP = graph_GSP.legend(loc="best")

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

# Plot the GSP before and after cleaning
fig2, ax = plt.subplots(2, 1, figsize=(16, 10))
sns.boxplot(x=na_drop_df['Iron Acton (Agg incl Gen)_GSP'], ax=ax[0])
sns.boxplot(x=cleaned_df['Iron Acton (Agg incl Gen)_GSP'], ax=ax[1])


# Time to import the weather data near the feeder
def add_weather_data(path=r'C:/Users/osims/PycharmProjects/SolarProject/WPD_Round_3/Data/weather_data_ch3'):
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


def add_energy_features(df):
    data = df.copy()

    # GSP
    # Add variable based on the demand 24 hours before current time
    data['Iron Acton (Agg incl Gen)_GSP_-24h'] = data['Iron Acton (Agg incl Gen)_GSP']. \
        shift(periods=48).interpolate(method='linear')

    # Add variable based on the demand 24 hours after current time
    data['Iron Acton (Agg incl Gen)_GSP_+24h'] = data['Iron Acton (Agg incl Gen)_GSP']. \
        shift(periods=-48).interpolate(method='linear')

    # Max demand over the previous 26h
    data['GSP_max_-26h'] = data.rolling(52, min_periods=1)['Iron Acton (Agg incl Gen)_GSP'].max()

    # Min demand over the previous 26h
    data['GSP_min_-26h'] = data.rolling(52, min_periods=1)['Iron Acton (Agg incl Gen)_GSP'].min()

    # Mean demand over the previous 26h
    data['GSP_avg_-26h'] = data.rolling(52, min_periods=1)['Iron Acton (Agg incl Gen)_GSP'].mean()

    data.dropna(subset=['GSP_max_-26h', 'Iron Acton (Agg incl Gen)_GSP_-24h',
                        'Iron Acton (Agg incl Gen)_GSP_+24h'], axis=0, inplace=True)

    return data


final_df = expand_datetime(final_df)
original_df = expand_datetime(original_df)
final_df = expand_weather(final_df)
original_df = expand_weather(original_df)
final_df = add_energy_features(final_df)
original_df = add_energy_features(original_df)

# Correlation Insights
corr_matrix = final_df.corr()
print(corr_matrix['Iron Acton (Agg incl Gen)_GSP'].sort_values(ascending=False).to_string())

X_cols = ['Filton 11kV K Bar (Agg incl Gen)_Prim_0', 'Filton 11kV J Bar (Agg incl Gen)_Prim_0',
          'Seabank 33kV (Agg incl Gen)_BSP_2', 'temperature', 'solar_irradiance', 'windspeed_north', 'windspeed_east',
          'pressure', 'spec_humidity', 'year', 'season', 'dayofweek', 'week_of_year', 'daily_max_temp+24h',
          'is_wknd', 'is_working_hr', 'month_x', 'month_y', 'temperature_+1h', 'temperature_+30m', 'temperature_+24h',
          'daily_avg_temp-24h', 'daily_avg_temp+24h', 'is_winter', 'Iron Acton (Agg incl Gen)_GSP_-24h',
          'GSP_max_-26h', 'Iron Acton (Agg incl Gen)_GSP_-24h', 'day_x', 'day_y', 'hour_x', 'hour_y',
          'day', 'hour', 'hourmin', 'GSP_min_-26h', 'GSP_avg_-26h', 'daily_min_temp+24h', 'temperature_+24h']
# X columns after feature importance optimisation
# X_cols = ['Filton 11kV K Bar (Agg incl Gen)_Prim_0', 'Filton 11kV J Bar (Agg incl Gen)_Prim_0',
#           'temperature', 'solar_irradiance', 'year', 'season', 'dayofweek',
#           'week_of_year', 'is_wknd', 'is_working_hr', 'month_x', 'month_y', 'temperature_+1h', 'daily_avg_temp-24h',
#           'daily_avg_temp+24h', 'is_winter', 'GSP_max_-26h', 'day', 'hour', 'hourmin', 'GSP_min_-26h',
#           'GSP_avg_-26h', 'daily_max_temp+24h', 'daily_min_temp+24h', 'is_lunch_hr', 'Holiday', 'month_x', 'month_y']
y_cols = ['Iron Acton (Agg incl Gen)_GSP', 'Lockleaze 33kV J Bar (Agg incl Gen)_BSP_0',
          'Clifton 11kV (Agg incl Gen)_Prim_0', 'Eastville 11kV (Agg incl Gen)_Prim_0',
          'Patchway 11kV (Agg incl Gen)_Prim_0', 'Cotham Primary (Agg incl Gen)_Prim_0',
          "Feeder Road 33kV 'J' Bar (Agg incl Gen)_BSP_1",
          'Bishopsworth 11kV (Agg incl Gen)_Prim_1', 'Bedminster 11kV (Agg incl Gen)_Prim_1',
          'WOODLAND WAY 11kV J BAR (Agg incl Gen)_Prim_1', 'Bower Ashton 11kV (Agg incl Gen)_Prim_1',
          'WOODLAND WAY 11kV K BAR (Agg incl Gen)_Prim_1', 'Keynsham West 11kV (Agg incl Gen)_Prim_1',
          'Western Approach 11kV (Agg incl Gen)_Prim_2', 'Rolls Royce Filton 132kV (Agg incl Gen)_BSP_3',
          'Chipping Sodbury 33Kv (Agg incl Gen)_BSP_4', 'Chipping Sodbury 11kV (Agg incl Gen)_Prim_4',
          "Feeder Road 33kV 'K' Bar (Agg incl Gen)_BSP_5", 'Feeder Road A (Agg incl Gen)_Prim_5',
          'Feeder Road B (Agg incl Gen)_Prim_5', 'Broadweir (Agg incl Gen)_Prim_5', 'Whitchurch (Agg incl Gen)_Prim_5',
          'St Pauls 132kV (Agg incl Gen)_BSP_6', 'St Pauls 11kV K Bar (Agg incl Gen)_Prim_6',
          'St Pauls 11kV J Bar (Agg incl Gen)_Prim_6', 'Bradley Stoke 33kV (Agg incl Gen)_BSP_7',
          'Almondsbury 11kV (Agg incl Gen)_Prim_7', 'Cribbs Causeway 11kV (Agg incl Gen)_Prim_7',
          'Abbeywood 11kV K Bar (Agg incl Gen)_Prim_7',
          'Bradley Stoke 11kV (Agg incl Gen)_Prim_7', 'Lockleaze 33kV K Bar (Agg incl Gen)_BSP_8',
          'Hewlett Packard 11kV (Agg incl Gen)_Prim_8', 'Lockleaze 11kV (Agg incl Gen)_Prim_8',
          'Winterbourne 11kV (Agg incl Gen)_Prim_8', 'Mangotsfield 11kV (Agg incl Gen)_Prim_8',
          'Emersons Green (Agg incl Gen)_Prim_8', 'Abbeywood 11kV J Bar (Agg incl Gen)_Prim_8']

X = final_df[X_cols]
y = final_df[y_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=42, shuffle=False)

# Loop through a number of
# models = {"DecisionTreeRegressor": DecisionTreeRegressor(),
#           "Ridge": Ridge(),
#           "Lasso": Lasso(),
#           "KNeighbours": KNeighborsRegressor(),
#           "XGB Regressor": XGBRegressor(),
#           "Bagging Regressor": BaggingRegressor(n_estimators=8)}
#
# for name, model in models.items():

model = ExtraTreesRegressor(n_estimators=50, random_state=42)
# model = ExtraTreesRegressor(n_estimators=170, min_samples_leaf=1, min_samples_split=2, max_features='auto',
#                             bootstrap=True, max_depth=30, random_state=42)

# A regressor chain takes the X inputs and predicts each y output, one at a time. Once y[i] has been predicted it
# will be used as an input for the next prediction
# order = [i for i in range(0, len(y_cols))]
# order = sorted(order, reverse=False)
# multi_output_model = RegressorChain(model, order=order)

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start=30, stop=200, num=10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 30, num=5)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 3, 4]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split}

# multi_output_model = MultiOutputRegressor(estimator=model)
# define the evaluation procedure
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

# rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=cv, verbose=4,
#                                random_state=42, n_jobs=-1, scoring='neg_root_mean_squared_error')
# evaluate the model and collect the scores
n_scores = cross_val_score(multi_output_model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=cv,
                           n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)
# summarize performance

print('Average RMSE: %.3f with a standard deviation of (%.3f)' % (mean(n_scores), std(n_scores)))

# rf_random.fit(X, y)
#
# print(rf_random.best_params_)
# print(rf_random.best_score_)

multi_output_model.fit(X_train, y_train)

# Feature Importances
model.fit(X_train, y_train.iloc[:, 0])

# # lgb.plot_importance(model, figsize=(16, 10))
# importances = model.feature_importances_
# std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
# forest_importances = pd.Series(importances, index=X_cols)

# fig4, ax = plt.subplots(figsize=(16, 10))
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    rmse = mean_squared_error(test_labels, predictions, squared=False)
    print('Model Performance')
    print('RMSE = {:0.3f}'.format(rmse), '\n')

    return rmse


# base_model = ExtraTreesRegressor(n_estimators=30, random_state=42)
# base_model.fit(X_train, y_train.iloc[:, 0])
# base_rmse = evaluate(base_model, X_test, y_test.iloc[:, 0])
base_rmse = 4.3847
print(f"\nBase Extra Trees Model RMSE = {base_rmse}")
current_rmse = evaluate(model, X_test, y_test.iloc[:, 0])
print('Improvement of {:0.2f} %.'.format(100 * -(current_rmse - base_rmse) / base_rmse))

# make predictions for the missing 24 hr period and a couple days after
pred_range = original_df.loc[
    (original_df['datetime'] >= '2021-01-11 00:00:00') & (original_df['datetime'] <= '2021-01-14 23:30:00')]
pred_range = pred_range.drop('datetime', axis=1)
X_rows = pred_range[X_cols]

yhat = multi_output_model.predict(X_rows)
pred = yhat[:, 0]

# Forecast for 4 days
forecast_window = 96 * 2
forecast_period_dates = pd.date_range(start=pd.to_datetime('2021-01-11 00:00:00'), periods=forecast_window,
                                      freq='30T').tolist()

# Obtain the values for the GSP before the missing data period
before = original_df.loc[
    (original_df['datetime'] < '2021-01-11 00:00:00') & (original_df['datetime'] > '2021-01-07 00:00:00')]
before_y = before[['datetime', y_cols[0]]]

# Actual values compared to predictions for the missing 24 hr period and a couple days after
after = original_df.loc[
    (original_df['datetime'] < '2021-01-15 00:00:00') & (original_df['datetime'] >= '2021-01-12 00:00:00')]
after_y = after[['datetime', y_cols[0]]]

fig3, ax = plt.subplots(1, 1, figsize=(16, 10))
sns.lineplot(x=before_y['datetime'], y=before_y[y_cols[0]], ax=ax, label='GSP Before 24 hour Period', color='indianred')
sns.lineplot(x=forecast_period_dates, y=pred, ax=ax, label='GSP Prediction', color='gold')
sns.lineplot(x=after_y['datetime'], y=after_y[y_cols[0]], ax=ax, label='GSP Actual', color='black', alpha=0.4)
ax.set_title("Scenario 1")

# Actual values compared to predictions for after the missing 24 hr period
actual = after_y.iloc[:, 1]
preds_sliced = pred[48:]

# sns.set_style('darkgrid')
# fig4, ax = plt.subplots(1, 1, figsize=(16, 10))
# sns.scatterplot(x=preds_sliced, y=actual, color='dodgerblue', ax=ax)
# ax.set_title("Predictions vs Actual")
# ax.set_ylabel("Actual")
# ax.set_xlabel("Predictions")


# Scenario Predictions
# Scenario Date Ranges
scenario_dates = {"Scenario": ["Scenario 1", "Scenario 2a", "Scenario 2b", "Scenario 3a", "Scenario 3b", "Scenario 3c",
                               "Scenario 4a", "Scenario 4b", "Scenario 5a", "Scenario 5b", "Scenario 6a", "Scenario 6b",
                               "Scenario 7", "Scenario 8"],
                  "start_date": ['2021-01-11 00:00:00', '2020-04-01 00:00:00', '2020-01-12 00:00:00',
                                 '2020-01-06 00:00:00', '2020-09-09 00:00:00', '2021-02-22 00:00:00',
                                 '2020-06-08 00:00:00', '2021-07-20 00:00:00', '2020-07-09 00:00:00',
                                 '2021-06-20 00:00:00', '2021-05-13 00:00:00', '2021-11-11 00:00:00',
                                 '2020-02-11 00:00:00', '2021-03-14 00:00:00'],
                  "end_date": ['2021-01-11 23:30:00', '2020-04-01 23:30:00', '2020-01-12 23:30:00',
                               '2020-01-06 23:30:00', '2020-09-09 23:30:00', '2021-02-22 23:30:00',
                               '2020-06-08 23:30:00', '2021-07-20 23:30:00', '2020-07-09 23:30:00',
                               '2021-06-20 23:30:00', '2021-05-13 23:30:00', '2021-11-11 23:30:00',
                               '2020-02-11 23:30:00', '2021-03-14 23:30:00']}

scenarios = []
for (scenario, start, end) in zip(scenario_dates["Scenario"], scenario_dates["start_date"], scenario_dates["end_date"]):
    scenario_range = original_df.loc[(original_df['datetime'] >= start) & (original_df['datetime'] <= end)]
    scenario_range = scenario_range.drop('datetime', axis=1)
    scenario_X = scenario_range[X_cols]
    print(scenario, f"length = {scenario_X.shape[0]} half-hours")
    scenario_y = multi_output_model.predict(scenario_X)
    scenarios.append(scenario_y)

scenarios_df = pd.DataFrame(np.concatenate(scenarios), columns=y_cols)
scenarios_df['Scenario'] = np.where(np.arange(len(scenarios_df)) % 48, 'xx', 'New Scenario')

count = 1
name_count = 0
for i, row in scenarios_df.iterrows():
    row['Scenario'] = scenario_dates["Scenario"][name_count]
    count += 1
    if count == 49:
        count = 1
        name_count += 1
    scenarios_df.at[i, 'Scenario'] = row['Scenario']

# scenarios_df.to_csv(path_or_buf='C:/Users/osims/PycharmProjects/SolarProject/WPD_Round_3/phase_1_model_results.csv')


plt.show()
