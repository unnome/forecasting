import pandas as pd


class PredictionModel():

    def __init__(self, name):
        self.name = name

    def create_prediction(self, train_df: pd.DataFrame,
                          end_date: str) -> pd.DataFrame:
        predict_df = train_df
        return predict_df


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
