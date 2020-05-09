import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math


class LossFunction():

    def __init__(self, name):
        self.name = name

    def calculate_performance(pred_df: pd.DataFrame) -> float:
        return 999


def calculate_MAPE(valid_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    y_true = valid_df['val']
    y_pred = pred_df['val']
    MAE = mean_absolute_error(y_true, y_pred)
    return MAE


def calculate_RMSE(valid_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    y_true = valid_df['val']
    y_pred = pred_df['val']
    RMSE = math.sqrt(mean_squared_error(y_true, y_pred))
    return RMSE


MAE = LossFunction(name='MAE')
MAE.calculate_performance = calculate_MAPE
RMSE = LossFunction(name='RMSE')
RMSE.calculate_performance = calculate_RMSE
