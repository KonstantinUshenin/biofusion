import numpy as np
import pandas as pd


class ShiftEliminator:

    def __init__(self):
        self.data_source = self.ds = DataSource(self)
        self.integration = self.int = self.fusion = self.fuse = Fusion(self)

    def result(self):
        return pd.concat(self.ds.data_series, ignore_index=True)

class Fusion:

    def __init__(self, pipeline):
        self.pipe = pipeline

    def mean_substraction(self, strategy='substraction_to_zero_mean'):

        data_col = self.pipe.ds._get_data_col()
        mean_series = np.zeros((len(self.pipe.ds.data_series), len(data_col)))

        for idx,ds in enumerate(self.pipe.ds.data_series):
            mean = ds[data_col].mean().values
            mean_series[idx,:] = mean

        if strategy == 'substraction_to_zero_mean':
            for idx, ds in enumerate(self.pipe.ds.data_series):
                ds[data_col] = ds[data_col] - mean_series[idx]
        elif strategy == 'division_to_one_mean':
            for idx, ds in enumerate(self.pipe.ds.data_series):
                ds[data_col] = ds[data_col]/mean_series[idx]
        elif strategy == 'substraction_of_average_mean':
            average_mean = mean_series.mean(axis=0)
            for idx, ds in enumerate(self.pipe.ds.data_series):
                ds[data_col] = ds[data_col] - average_mean
        elif strategy == 'substraction_to_average_mean':
            average_mean = mean_series.mean(axis=0)
            for idx, ds in enumerate(self.pipe.ds.data_series):
                ds[data_col] = ds[data_col] - (mean_series[idx] - average_mean)
        else:
            raise NotImplementedError()

        self.mean_series = mean_series

class DataSource:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.name_series = []
        self.data_series = []
        self.data_col_series = []
        self.markup_col_series = []
        self.adv_col_series = []

    def add(self, data,
                    name=None,
                    data_col=None,
                    markup_col=None,
                    adv_col=None,
                    deep_copy=True):

        data_ = data.copy(deep=True) if deep_copy else data

        self.data_series.append(data_)

        if name is None:
            name="_default_{}".format(len(self.name_series))
        self.name_series.append(name)

        if data_col is None:
            data_col  = data_.columns
        self.data_col_series.append(data_col)

        if markup_col is None:
            markup_col = data_col[0]
        self.markup_col_series.append(markup_col)

        if adv_col is None:
            adv_col = []
        self.adv_col_series.append(adv_col)

    def _get_data_col(self):
        return self.data_col_series[0]