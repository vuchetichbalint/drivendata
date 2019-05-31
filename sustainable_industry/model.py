import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split


DATA_DIR = Path('data')

def mape(y_true, y_pred):
    return (
        np.abs(y_pred - y_true) / np.where(np.abs(y_true) < 290_000, 290_000, np.abs(y_true))
    ).mean()

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


def prep_time_series_features(df, columns=None):
    if columns is None:
        columns = df.columns

    ts_df = df[ts_cols].set_index('process_id')

    # create features: min, max, mean, standard deviation, and mean of the last five observations
    ts_features = ts_df.groupby('process_id').agg(['min', 'max', 'mean', 'std', lambda x: x.tail(5).mean()])

    return ts_features

def create_feature_matrix(df):
    metadata = prep_metadata(df)
    time_series = prep_time_series_features(df)
    
    # join metadata and time series features into a single dataframe
    feature_matrix = pd.concat([metadata, time_series], axis=1)
    
    return feature_matrix


def read_preprocess_data():
    train_values = pd.read_csv(DATA_DIR / 'train_values.csv',
                           index_col=0,
                           parse_dates=['timestamp'])

    train_labels = pd.read_csv(DATA_DIR / 'train_labels.csv',
                           index_col=0)

    test_values = pd.read_csv(DATA_DIR / 'test_values.csv',
                         index_col=0,
                         parse_dates=['timestamp'])


    train_values = train_values[train_values.phase != 'final_rinse']

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

    train_features = create_feature_matrix(train_limited)
    test_features = create_feature_matrix(test_values)

    # assume it returns pure numpy arrays!

    # important to shuffle the data!!!!


    return x_train, y_train, x_validation, y_validation, x_production


def get_best_model(model, hyperparams, x, y):
    grid = GridSearchCV(
        estimator=model,
        param_grid=hyperparams,
        cv=5,
        scoring=make_scorer(score_func=mape, greater_is_better=False),
    )

    print('Fitting model:')
    print(model)
    grid.fit(x, y)
    print(f'Best params: {grid.best_params_}')
    print(f'Test score: {grid.best_score_}')


    return grid.best_estimator_


def one_layer_modelling(model, hyperparams, x, y):
    # searching for best params
    best_model = get_best_model(model, hyperparams, x, y)

    preds = []
    #cv by hand
    kf = KFold(n_splits=10, shuffle=False, random_state=2019)
    for train_index, dev_index in kf.split(x):
        x_train, x_dev, y_train = x[train_index], x[dev_index], y[train_index]

        partial_model = clone(best_model)
        partial_model.train(x_train, y_train)
        y_hat = partial_model.predict(x_dev)

        preds.append(y_hat)

    # concat preds into a single array
    preds = np.concatenate(preds)

    return best_model, preds


def model_layers(layer1_models, layer2_model, x, y):
    layer1_best_models = []
    layer2_features = []
    for model in layer1_models:
        best_model, preds = one_layer_modelling(model['estimator'], model['hyperparams'], x, y)
        layer1_best_models.append(best_model)
        # assert preds.shape == (x,)
        # required preds.shape == (x,1)
        layer2_features.append(preds.reshape([-1,1]))
    # flatten layer2_features
    layer2_x = np.concatenate(layer2_features).T
    # assert layer2_x.shape == (n_samples, n_features)

    best_layer2_model = get_best_model(layer2_model['estimator'], layer2_model['hyperparams'], layer2_x, y)

    return layer1_best_models, best_layer2_model


def apply_models(layer1_models, layer2_model, x):
    layer1_preds = []
    for model in layer1_models:
        y_hat = model.predict(x)
        # assert y_hat.shape == (x,)
        # required y_hat.shape == (x,1)
        layer1_preds.append(y_hat.reshape([-1,1]))
    layer2_features = np.concatenate(layer1_preds)
    # assert layer2_features.shape == (n_samples, n_features)

    layer2_preds = layer2_model.predict(layer2_features)

    return layer2_preds


def validate(layer1_models, layer2_model, x, y):
    y_hat = apply_models(layer1_models, layer2_model, x)
    validation_mape = mape(y, y_hat)
    return validation_mape


def production(layer1_models, layer2_model, x):
    y_hat = apply_models(layer1_models, layer2_model, x)
    return y_hat


def store_experiment(layer1_models, layer2_model, validation_mape, y_hat):
    filename = str(datetime.datetime.now())[:19].replace(' ', '').replace('-', '').replace(':', '')

    submission_format = pd.read_csv(DATA_DIR / 'submission_format.csv', index_col=0)
    my_submission = pd.DataFrame(data=y_hat,
                             columns=submission_format.columns,
                             index=submission_format.index)

    my_submission.to_csv(DATA_DIR / f'submission_{filename}.csv')

    t = datetime.datetime.now()
    # possible not gonna work ¯\_(ツ)_/¯
    log = f"""Time: {t}, MAPE: {validation_mape},
            layer1_models
            {layer1_models},
            layer2_model
            {layer2_model}
        """

    with open(f'logs/{filename}.log', 'a') as f:
    f.write(log)


def run(layer1_models, layer2_model):
    x_train, y_train, x_validation, y_validation, x_production = read_preprocess_data()

    layer1_models, layer2_model = model_layers(layer1_models, layer2_model, x_train, y_train)

    validation_mape = validate(layer1_models, layer2_model, x_validation, y_validation)
    y_hat = production(layer1_models, layer2_model, x_production)

    print(f'--- MAPE: {validation_mape} ----')

    store_experiment(layer1_models, layer2_model, validation_mape, y_hat)

    print('Everything is awesome!')


if __name__ == '__main__':

    layer1_models = [
        {
            'estimator': RandomForestRegressor(),
            'hyperparams': {'asd': [1, 2, 3]}
        },
        {
            'estimator': RandomForestRegressor(),
            'hyperparams': {'asd': [1, 2, 3]}
        },
    ]

    layer2_model = {
        'estimator': RandomForestRegressor(),
        'hyperparams': {'asd': [1, 2, 3]}
    }

    run(layer1_models, layer2_model)
