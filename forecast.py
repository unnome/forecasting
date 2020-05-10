import pandas as pd


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
