from fbprophet import Prophet
from matplotlib import pyplot as plt
import pandas as pd


class PredictionModel():

    def __init__(self, name):
        self.name = name

    def create_prediction(self, train_df: pd.DataFrame,
                          end_date: str) -> pd.DataFrame:
        return 'prediction interface method, please add relevant method'

    def present_results(self, train_df: pd.DataFrame,
                        valid_df: pd.DataFrame,
                        pred_df: pd.DataFrame,
                        save_location: str) -> plt.plot:
        train_df['label'] = 'train'
        valid_df['label'] = 'validation'
        pred_df['label'] = 'prediction'
        self.plot_df = pd.concat([train_df, valid_df, pred_df])
        self.plot_pivot = self.plot_df.pivot_table(
            index='dt', columns='label', values='val'
        )
        self.plot_pivot.plot(figsize=[20, 10], alpha=0.7)
        plt.title(self.name)
        plt.savefig(save_location + self.name + '.png')
        return self.plot_df


def last_value_prediction(train_df: pd.DataFrame,
                          empty_pred_df: pd.DataFrame) -> pd.DataFrame:
    last_value = (
        train_df['val'][train_df['dt'] == train_df['dt'].max()].values[0]
    )
    pred_df = empty_pred_df.copy(deep=True)
    pred_df['val'] = last_value
    return pred_df


LastValueModel = PredictionModel(name='Last value')
LastValueModel.create_prediction = last_value_prediction


def prophet_prediction(train_df: pd.DataFrame,
                       empty_pred_df: pd.DataFrame) -> pd.DataFrame:

    # rename train_df columns
    df = train_df.rename(columns={
        'dt': 'ds',
        'val': 'y',
    })

    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=empty_pred_df.shape[0])
    pred_df = m.predict(future)

    # change column names again
    pred_df.rename(columns={
        'ds': 'dt',
        'yhat': 'val',
    }, inplace=True)
    pred_df = (
        pred_df[pred_df['dt'] >= empty_pred_df['dt'].min()][['dt', 'val']]
    )
    return pred_df


ProphetModel = PredictionModel(name='Facebook Prophet')
ProphetModel.create_prediction = prophet_prediction
