import numpy as np
import os.path
import pandas as pd
import unittest

from forecast import (
    importDataFrame,
    makeEmptyPredictionDF,
    makeTrainDF,
    makeValidationDF,
)
from prediction_models import (
    PredictionModel,
    LastValueModel,
    ProphetModel,
)

from loss_functions import LossFunction, MAE, RMSE


class Target(unittest.TestCase):

    def setUp(self):
        self.data_location = 'data/daily-min-temperatures.csv'
        self.my_test_ts = importDataFrame(self.data_location)
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
        self.data_location = 'data/daily-min-temperatures.csv'
        self.target_df = importDataFrame(self.data_location)
        self.split_date = pd.to_datetime('1986-01-01 00:00:00')
        self.train_df = makeTrainDF(self.target_df, self.split_date)

    def test_df_contains_only_data_before_split(self):
        self.assertTrue(self.train_df['dt'].max() == self.split_date)

    def test_df_has_min_date_like_target_df(self):
        self.assertTrue(
            self.train_df['dt'].min() == self.target_df['dt'].min())

    def test_train_df_contains_column_dt_and_val(self):
        self.assertEqual(self.train_df.columns[0], 'dt', self.train_df.columns)
        self.assertEqual(self.train_df.columns[1], 'val')


class EmptyPredictionDF(unittest.TestCase):

    def setUp(self):
        self.data_location = 'data/daily-min-temperatures.csv'
        self.target_df = importDataFrame(self.data_location)
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


class TestPredictionModelClass(unittest.TestCase):

    def setUp(self):
        self.data_location = 'data/daily-min-temperatures.csv'
        self.plot_save_location = '/home/boats/Desktop/'
        self.target_df = importDataFrame(self.data_location)
        self.split_date = pd.to_datetime('1986-01-01 00:00:00')
        self.train_df = makeTrainDF(self.target_df, self.split_date)
        self.valid_df = makeValidationDF(self.target_df, self.split_date)
        self.empty_pred_df = makeEmptyPredictionDF(self.target_df,
                                                   self.split_date)
        self.pred_df = LastValueModel.create_prediction(
            self.train_df, self.empty_pred_df)

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

        self.plot_df = self.test_model.present_results(
            self.train_df, self.valid_df, self.pred_df, self.plot_save_location
        )

    def test_model_has_predict_method(self):
        self.assertTrue(hasattr(self.test_model, 'create_prediction'))

    def test_model_predict_output_columns_are_named_as_expected(self):
        self.column_names = self.pred_df_from_model.columns
        self.assertEqual(self.column_names[0], 'dt')
        self.assertEqual(self.column_names[1], 'val')

    def test_model_predict_output_columns_have_values(self):
        for value in self.pred_df_from_model['val']:
            self.assertIsNotNone(value)


class TestLossFunctionClass(unittest.TestCase):

    def setUp(self):
        self.data_location = 'data/daily-min-temperatures.csv'
        self.target_df = importDataFrame(self.data_location)
        self.split_date = pd.to_datetime('1987-01-01 00:00:00')
        self.train_df = makeTrainDF(self.target_df, self.split_date)
        self.valid_df = makeValidationDF(self.target_df, self.split_date)
        self.empty_pred_df = makeEmptyPredictionDF(self.target_df,
                                                   self.split_date)
        self.pred_df = LastValueModel.create_prediction(
            self.train_df, self.empty_pred_df)
        self.MAPE = LossFunction(name='MAPE')

        def calc_MAPE(valid_df, pred_df):
            test = valid_df['val'].sub(pred_df['val'])
            test = test.sum()
            return test

        self.MAPE.calculate_performance = calc_MAPE

    def test_loss_fn_class_has_calculate_performance_method(self):
        self.assertTrue(hasattr(self.MAPE, 'calculate_performance'))

    def test_loss_fn_work_with_rest_of_funnel(self):
        self.MAPE_val = self.MAPE.calculate_performance(
            self.valid_df, self.pred_df)
        self.my_np_float = np.float64(0.99)
        self.assertEqual(type(self.MAPE_val), type(self.my_np_float))


class ProphetModelTest(unittest.TestCase):

    def setUp(self):
        self.data_location = 'data/daily-min-temperatures.csv'
        self.plot_name = 'plots/plot.png'
        self.target_df = importDataFrame(self.data_location)
        self.split_date = pd.to_datetime('1987-01-01 00:00:00')
        self.train_df = makeTrainDF(self.target_df, self.split_date)
        self.valid_df = makeValidationDF(self.target_df, self.split_date)
        self.empty_pred_df = makeEmptyPredictionDF(self.target_df,
                                                   self.split_date)
        self.ph_pred = ProphetModel.create_prediction(
            self.train_df, self.empty_pred_df)

    def tearDown(self):
        try:
            os.remove(self.plot_name)
        except OSError:
            pass

    def test_prediction_df_has_the_same_shape_of_target_df(self):
        self.assertEqual(self.empty_pred_df.shape, self.ph_pred.shape)

    def test_MAE_loss_functions_outputs_a_number(self):
        self.ph_MAE = MAE.calculate_performance(self.valid_df, self.ph_pred)
        self.float_nr = np.float64('0.9')
        self.assertEquals(type(self.float_nr), type(self.ph_MAE),
                          type(self.ph_MAE))

    def test_RMSE_loss_functions_outputs_a_number(self):
        self.ph_RMSE = RMSE.calculate_performance(self.valid_df, self.ph_pred)
        self.float_nr = float('0.9')
        self.assertEquals(type(self.float_nr), type(self.ph_RMSE),
                          type(self.ph_RMSE))

    def test_prophet_outputs_exactly_the_columns_we_expect(self):
        self.assertEqual(self.ph_pred.columns[0], 'dt')
        self.assertEqual(self.ph_pred.columns[1], 'val')

    def test_prophet_df_has_unique_values_in_index(self):
        self.assertListEqual(self.valid_df['dt'].to_list(),
                             self.ph_pred['dt'].to_list())


class PlottingProphetDataFrame(unittest.TestCase):

    def setUp(self):
        self.data_location = 'data/daily-min-temperatures.csv'
        self.plot_name = 'plots/plot.png'
        self.target_df = importDataFrame(self.data_location)
        self.split_date = pd.to_datetime('1987-01-01 00:00:00')
        self.train_df = makeTrainDF(self.target_df, self.split_date)
        self.valid_df = makeValidationDF(self.target_df, self.split_date)
        self.empty_pred_df = makeEmptyPredictionDF(self.target_df,
                                                   self.split_date)
        self.ph_pred = ProphetModel.create_prediction(
            self.train_df, self.empty_pred_df)

        self.train_df['label'] = 'train'
        self.valid_df['label'] = 'validation'
        self.ph_pred['label'] = 'prediction'

        self.merged_df = pd.concat([self.train_df,
                                    self.ph_pred,
                                    self.valid_df])
        self.merged_pivot = self.merged_df.pivot_table(index='dt',
                                                       columns='label',
                                                       values='val')

    def test_prophet_plotting_df_ends_at_same_date_of_target(self):
        self.assertEqual(self.merged_df.dt.max(),
                         self.target_df.dt.max())

    def test_train_df_has_dt_column(self):
        self.assertEqual(self.train_df.columns[0], 'dt')

    def test_prophet_train_df_starts__at_same_date_of_target(self):
        self.assertEqual(self.target_df.dt.min(),
                         self.train_df.dt.min())

    def test_prophet_plotting_df_starts_at_same_date_of_target(self):
        self.assertEqual(self.target_df['dt'].min(),
                         self.merged_df['dt'].min())


if __name__ == '__main__':
    unittest.main(warnings='ignore')
