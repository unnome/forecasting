from matplotlib import pyplot as plt
import pandas as pd

from prediction_models import (
    LastValueModel,
    PredictionModel,
)


# Ivan opens the app, and is asked to insert some inputs, such as:
# - the target series:
# - the date that splits train and validation
#
# He is amazed that the output is a series of blocks, one for each model,
# composed by
# - a titled plot that shows train, validation and prediction series
# - a table with a performance value for each known loss function.


DATA_LOCATION = 'data/daily-min-temperatures.csv'


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
    train_df = df[df['dt'] <= split_date]
    return train_df


def makeEmptyPredictionDF(df: pd.DataFrame, split_date: str) -> pd.DataFrame:
    split_date = pd.to_datetime(split_date)
    empty_pred_df = df[df['dt'] > split_date].copy(deep=True)
    empty_pred_df['val'] = None
    return empty_pred_df


def makePlotDF(target_df: pd.DataFrame,
               prediction_df: pd.DataFrame) -> pd.DataFrame:
    target_df['label'] = 'truth'
    prediction_df['label'] = 'prediction'
    plotting_df = pd.concat([target_df, prediction_df])
    return plotting_df


def plot_truth_and_pred(plotting_df: pd.DataFrame,
                        PredModel: PredictionModel) -> plt.plot:
    plotting_df = plotting_df.pivot(index='dt', columns='label', values='val')
    plotting_df.plot(figsize=[20, 10], alpha=0.5)
    plt.title(PredModel.name)
    plt.savefig('/home/boats/Desktop/plot_test.png')
    print('plot saved')
    return


if __name__ == '__main__':
    df = importDataFrame(DATA_LOCATION)
    split_date = '1987-01-01'
    train = makeTrainDF(df, split_date)
    empty_pred = makeEmptyPredictionDF(df, split_date)
    pred = LastValueModel.create_prediction(train, empty_pred)
    plot_df = makePlotDF(df, pred)
    plot_truth_and_pred(plot_df, LastValueModel)
