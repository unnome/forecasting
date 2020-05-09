import unittest
import pandas as pd

from forecast import (
    importDataFrame,
    makeEmptyPredictionDF,
    makeTrainDF,
    makePlotDF,
)
from prediction_models import PredictionModel


class Target(unittest.TestCase):

    def setUp(self):
        self.my_test_ts = importDataFrame()
        self.column_names = self.my_test_ts.columns

    def test_df_contains_only_two_columns(self):
        """
        the dataframe contains only two columns
        """
        self.assertEqual(self.my_test_ts.shape[1], 2)

    def test_timeseries_contains_at_leats_12_entries(self):
        self.assertGreaterEqual(self.my_test_ts.shape[0], 12)

    def test_df_columns_have_correct_names(self):
        # we want to ensure that the first column is named "dt"
        # and the second column is named "val"
        self.assertEqual(self.column_names[0], 'dt')
        self.assertEqual(self.column_names[1], 'val')

    def test_df_only_has_two_columns(self):
        self.assertEqual(len(self.column_names), 2)

    def test_columns_contain_expected_data_types(self):
        self.assertEqual(self.my_test_ts['dt'].dtype, 'datetime64[ns]')
        self.assertEqual(self.my_test_ts['val'].dtype, 'float')


class TrainDF(unittest.TestCase):

    def setUp(self):
        self.target_df = importDataFrame()
        self.split_date = pd.to_datetime('1986-01-01 00:00:00')
        self.split_df = makeTrainDF(self.target_df, self.split_date)

    def test_df_contains_only_data_before_split(self):
        self.assertTrue(self.split_df['dt'].max() == self.split_date)


class EmptyPredictionDF(unittest.TestCase):

    def setUp(self):
        self.target_df = importDataFrame()
        self.split_date = pd.to_datetime('1986-01-01 00:00:00')
        self.empty_pred_df = makeEmptyPredictionDF(self.target_df,
                                                   self.split_date)

    def test_df_columns_have_correct_names(self):
        # the df contains a 'dt' column and a 'prediction'
        self.column_names = self.empty_pred_df.columns
        self.assertEqual(self.column_names[0], 'dt')
        self.assertEqual(self.column_names[1], 'val')

    def test_empty_pred_df_only_contains_dates_after_split(self):
        self.assertGreater(self.empty_pred_df['dt'].min(), self.split_date)

    def test_empty_pred_df_has_null_values_in_val_column(self):
        for value in self.empty_pred_df['val']:
            self.assertIsNone(value)


class PredictionModelTest(unittest.TestCase):

    def setUp(self):
        self.target_df = importDataFrame()
        self.split_date = pd.to_datetime('1986-01-01 00:00:00')
        self.train_df = makeTrainDF(self.target_df, self.split_date)
        self.empty_pred_df = makeEmptyPredictionDF(self.target_df,
                                                   self.split_date)

        self.test_model = PredictionModel('test name')

        def dummy_pred_fn(train_df, empty_pred_df):
            last_value = (train_df[train_df['dt'] == train_df['dt']
                                   .max()]['val'])
            pred_df = empty_pred_df.copy(deep=True)
            pred_df['val'] = last_value
            return pred_df

        self.test_model.predict = dummy_pred_fn
        self.pred_df_from_model = self.test_model.predict(self.train_df,
                                                          self.empty_pred_df)

    def test_model_has_predict_method(self):
        self.assertTrue(hasattr(self.test_model, 'create_prediction'))

    def test_model_predict_output_columns_are_named_as_expected(self):
        self.column_names = self.pred_df_from_model.columns
        self.assertEqual(self.column_names[0], 'dt')
        self.assertEqual(self.column_names[1], 'val')

    def test_model_predict_output_columns_have_values(self):
        for value in self.pred_df_from_model['val']:
            self.assertIsNotNone(value)


class PlottingDF(unittest.TestCase):

    def setUp(self):
        self.target_df = importDataFrame()
        self.split_date = pd.to_datetime('1986-01-01 00:00:00')
        self.train_df = makeTrainDF(self.target_df, self.split_date)
        self.empty_pred_df = makeEmptyPredictionDF(self.target_df,
                                                   self.split_date)

        self.test_model = PredictionModel('test name')

        def dummy_pred_fn(train_df, empty_pred_df):
            last_value = (train_df[train_df['dt'] == train_df['dt']
                                   .max()]['val'])
            pred_df = empty_pred_df.copy(deep=True)
            pred_df['val'] = last_value
            return pred_df

        self.test_model.predict = dummy_pred_fn
        self.pred_df_from_model = self.test_model.predict(self.train_df,
                                                          self.empty_pred_df)
        self.plotting_df = makePlotDF(self.target_df,
                                      self.pred_df_from_model)

    def test_df_has_the_three_expected_columns(self):
        # the plotting df should have dt, val and label
        self.assertEqual(len(self.plotting_df.columns), 3)

    def test_model_predict_output_columns_have_values(self):
        for value in self.plotting_df['val']:
            self.assertIsNotNone(value)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
