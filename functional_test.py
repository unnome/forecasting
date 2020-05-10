import unittest

from prediction_models import (
    LastValueModel,
    ProphetModel,
)

from forecast import (
    importDataFrame,
    makeTrainDF,
    makeValidationDF,
    makeEmptyPredictionDF,
)

from loss_functions import MAE, RMSE


class NewTest(unittest.TestCase):

    def setUp(self):
        self.DATA_LOCATION = 'data/daily-min-temperatures.csv'
        self.PLOT_SAVE_LOCATION = '/home/boats/Desktop/'

    def test_user_journey(self):
        # Ivan just downloaded some data and wants to play with it
        # He makes a dataframe from the csv
        self.df = importDataFrame(self.DATA_LOCATION)
        self.assertIsNotNone(self.df)

        # He determines the split date and creates train and validation dfs
        self.split_date = '1987-01-01'
        self.train = makeTrainDF(self.df, self.split_date)
        self.valid = makeValidationDF(self.df, self.split_date)
        self.empty_pred = makeEmptyPredictionDF(self.df, self.split_date)

        # He decides to apply the Naive prediction model
        self.LV_pred = LastValueModel.create_prediction(
            self.train, self.empty_pred)

        # He then wants to admire the results of his hard work in a png file
        LastValueModel.present_results(
            self.train, self.valid, self.LV_pred, self.PLOT_SAVE_LOCATION)

        # He also wants to see the performance of the model on MAE and RMSE
        self.MAE_val = MAE.calculate_performance(self.valid, self.LV_pred)
        self.RMSE_val = RMSE.calculate_performance(self.valid, self.LV_pred)
        self.assertGreater(self.MAE_val, 0.0)
        self.assertGreater(self.RMSE_val, 0.0)

        # Satisfied, he now moves on to Prophet.
        # He creates the prediction
        self.PH_pred = ProphetModel.create_prediction(
            self.train, self.empty_pred)

        # He plots the result in a png file
        ProphetModel.present_results(
            self.train, self.valid, self.PH_pred, self.PLOT_SAVE_LOCATION)

        # He measures the performance of the model


if __name__ == '__main__':
    unittest.main(warnings='ignore')
