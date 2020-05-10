import pandas as pd

from prediction_models import (
    LastValueModel,
)

from loss_functions import MAE, RMSE

# Ivan opens the app, and is asked to insert some inputs, such as:
# - the target series:
# - the date that splits train and validation
#
# He is amazed that the output is a series of blocks, one for each model,
# composed by
# - a titled plot that shows train, validation and prediction series
# - a table with a performance value for each known loss function.


DATA_LOCATION = 'data/daily-min-temperatures.csv'
PLOT_SAVE_LOCATION = '/home/boats/Desktop'


def importDataFrame(data_location: str) -> pd.DataFrame:
    data = pd.read_csv(data_location)
    data = data.rename(columns={
        data.columns[0]: 'dt',
        data.columns[1]: 'val',
    })
    data['dt'] = pd.to_datetime(data['dt'])
    data['val'] = data['val'].astype('float')
    return data


def makeTrainDF(df: pd.DataFrame, split_date: str) -> pd.DataFrame:
    split_date = pd.to_datetime(split_date)
    train_df = df[df['dt'] <= split_date].copy(deep=True)
    return train_df


def makeValidationDF(df: pd.DataFrame, split_date: str) -> pd.DataFrame:
    split_date = pd.to_datetime(split_date)
    valid_df = df[df['dt'] > split_date].copy(deep=True)
    return valid_df


def makeEmptyPredictionDF(df: pd.DataFrame, split_date: str) -> pd.DataFrame:
    split_date = pd.to_datetime(split_date)
    empty_pred_df = df[df['dt'] > split_date].copy(deep=True)
    empty_pred_df['val'] = None
    return empty_pred_df


if __name__ == '__main__':
    df = importDataFrame(DATA_LOCATION)
    split_date = '1987-01-01'
    train = makeTrainDF(df, split_date)
    valid = makeValidationDF(df, split_date)
    empty_pred = makeEmptyPredictionDF(df, split_date)
    pred = LastValueModel.create_prediction(train, empty_pred)
    LastValueModel.present_results(train, valid, pred, PLOT_SAVE_LOCATION)
    MAE_val = MAE.calculate_performance(valid, pred)
    RMSE_val = RMSE.calculate_performance(valid, pred)
    print('MAE --> ', MAE_val)
    print('RMSE --> ', RMSE_val)
