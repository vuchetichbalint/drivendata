%matplotlib inline

# mute warnings for this blog post
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 40)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = Path('../data/final/public')

# for training our model
train_values = pd.read_csv(DATA_DIR / 'train_values.csv',
                           index_col=0,
                           parse_dates=['timestamp'])

train_labels = pd.read_csv(DATA_DIR / 'train_labels.csv',
                           index_col=0)


train_values.head()


# subset to final rinse phase observations 
final_phases = train_values[(train_values.target_time_period)]

# let's look at just one process
final_phase = final_phases[final_phases.process_id == 20017]


 calculate target variable
final_phase = final_phase.assign(target=np.maximum(final_phase.return_flow, 0) * final_phase.return_turbidity)


# plot flow, turbidity, and target 
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

ax[0].plot(final_phase.return_flow)
ax[0].set_title('Return flow in final phase')

ax[1].plot(final_phase.return_turbidity, c='orange')
ax[1].set_title('Return turbidity in final phase')

ax[2].plot(final_phase.target, c='green')
ax[2].set_title('Turbidity in final phase in NTU.L');


# sum to get target
final_phase.target.sum()


# confirm that value matches the target label for this process_id
train_labels.loc[20017]


train_values = train_values[train_values.phase != 'final_rinse']

train_values.groupby('process_id').phase.nunique().value_counts().sort_index().plot.bar()
plt.title("Number of Processes with $N$ Phases");



# create a unique phase identifier by joining process_id and phase
train_values['process_phase'] = train_values.process_id.astype(str) + '_' + train_values.phase.astype(str)
process_phases = train_values.process_phase.unique()

# randomly select 80% of phases to keep
rng = np.random.RandomState(2019)
to_keep = rng.choice(
                process_phases,
                size=np.int(len(process_phases) * 0.8),
                replace=False)

train_limited = train_values[train_values.process_phase.isin(to_keep)]

# subset labels to match our training data
train_labels = train_labels.loc[train_limited.process_id.unique()]


train_limited.groupby('process_id').phase.nunique().value_counts().sort_index().plot.bar()
plt.title("Number of Processes with $N$ Phases (Subset for Training)");



def prep_metadata(df):
    # select process_id and pipeline
    meta = df[['process_id', 'pipeline']].drop_duplicates().set_index('process_id') 
    
    # convert categorical pipeline data to dummy variables
    meta = pd.get_dummies(meta)
    
    # pipeline L12 not in test data
    if 'L12' not in meta.columns:
        meta['pipeline_L12'] = 0
    
    # calculate number of phases for each process_object
    meta['num_phases'] = df.groupby('process_id')['phase'].apply(lambda x: x.nunique())
    
    return meta

# show example for first 5,000 observations
prep_metadata(train_limited.head(5000))




# variables we'll use to create our time series features
ts_cols = [
    'process_id',
    'supply_flow',
    'supply_pressure',
    'return_temperature',
    'return_conductivity',
    'return_turbidity',
    'return_flow',
    'tank_level_pre_rinse',
    'tank_level_caustic',
    'tank_level_acid',
    'tank_level_clean_water',
    'tank_temperature_pre_rinse',
    'tank_temperature_caustic',
    'tank_temperature_acid',
    'tank_concentration_caustic',
    'tank_concentration_acid',
]



def prep_time_series_features(df, columns=None):
    if columns is None:
        columns = df.columns
    
    ts_df = df[ts_cols].set_index('process_id')
    
    # create features: min, max, mean, standard deviation, and mean of the last five observations
    ts_features = ts_df.groupby('process_id').agg(['min', 'max', 'mean', 'std', lambda x: x.tail(5).mean()])
    
    return ts_features

# show example for first 5,000 observations
prep_time_series_features(train_limited.head(5000), columns=ts_cols)


def create_feature_matrix(df):
    metadata = prep_metadata(df)
    time_series = prep_time_series_features(df)
    
    # join metadata and time series features into a single dataframe
    feature_matrix = pd.concat([metadata, time_series], axis=1)
    
    return feature_matrix


train_features = create_feature_matrix(train_limited)



train_features.head()


%%time
rf = RandomForestRegressor(n_estimators=1000, random_state=2019)
rf.fit(train_features, np.ravel(train_labels))


# load the test data
test_values = pd.read_csv(DATA_DIR / 'test_values.csv',
                         index_col=0,
                         parse_dates=['timestamp'])

# create metadata and time series features
test_features = create_feature_matrix(test_values)


test_features.head()


preds = rf.predict(test_features)

submission_format = pd.read_csv(DATA_DIR / 'submission_format.csv', index_col=0)



# confirm everything is in the right order
assert np.all(test_features.index == submission_format.index)


my_submission = pd.DataFrame(data=preds,
                             columns=submission_format.columns,
                             index=submission_format.index)


my_submission.head()


my_submission.to_csv('submission.csv')