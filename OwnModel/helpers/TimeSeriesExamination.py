import numpy as np
import numbers
# import ultraimport
from numbers import Number
from sklearn.metrics import r2_score
from .MTimeSeries import MTimeSeries
from .TimeSeries import TimeSeries
from .StandardStorage import StandardStorage
from .StandardStorage import read_csv_database_file
from .TableWriter import TablePrinter
# ultraimport('__dir__/../helpers/MTimeSeries.py', 'MTimeSeries', globals=globals())
# ultraimport('__dir__/../helpers/TimeSeries.py', 'TimeSeries', globals=globals())
# ultraimport('__dir__/../helpers/StandardStorage.py', 'StandardStorage', globals=globals())
# ultraimport('__dir__/../helpers/StandardStorage.py', 'read_csv_database_file', globals=globals())
# ultraimport('__dir__/../helpers/TableWriter.py', 'TablePrinter', globals=globals())
from datetime import datetime
from math import floor
import time
import os

class MetricIntegralError:
    def __init__(self):
        pass

    def __call__(self, reference_ts, computed_ts, upper=None, lower=None):
        r = reference_ts.data
        c = computed_ts.data
        return 100 * (r.sum() - c.sum()) / r.sum() # %

    @property
    def unit(self):
        return "%"

    def __str__(self):
        return "IE"

class MetricMAE:
    def __init__(self):
        pass

    def __call__(self, reference_ts, computed_ts, upper=None, lower=None):
        e = (reference_ts - computed_ts).abs()
        return np.nanmean(e.data)  # MAE

    @property
    def unit(self):
        return "MAE"

    def __str__(self):
        return "MAE"


class MetricRMSE:
    def __init__(self):
        pass

    def __call__(self, reference_ts, computed_ts, upper=None, lower=None):
        e = (computed_ts - reference_ts)
        ee = e * e
        return np.sqrt(np.nanmean(ee.data))  # RMSE

    @property
    def unit(self):
        return "RMSE"

    def __str__(self):
        return "RMSE"

class MetricRRMSE:
    def __init__(self):
        pass

    def __call__(self, reference_ts, computed_ts, upper=None, lower=None):
        r = reference_ts.data
        c = computed_ts.data
        return np.sqrt(np.nanmean((c - r) ** 2)) / np.nanmean(r)

    @property
    def unit(self):
        return "rRMSE"

    def __str__(self):
        return "rRMSE"


class MetricMAPE:
    def __init__(self):
        pass

    def __call__(self, reference_ts, computed_ts, upper=None, lower=None):
        r = reference_ts.data
        c = computed_ts.data
        e = np.nansum(abs(2*(c-r) / (r+c))) * 100 / len(r)
        return e # RMSE

    @property
    def unit(self):
        return "MAPE"

    def __str__(self):
        return "MAPE"

class MetricMBE:
    def __init__(self):
        pass

    def __call__(self, reference_ts, computed_ts, upper=None, lower=None):
        r = reference_ts.data
        c = computed_ts.data
        e = sum(c-r) / len(r)
        return e

    @property
    def unit(self):
        return "MBE"

    def __str__(self):
        return "MBE"

class MetricRMBE:
    def __init__(self):
        pass

    def __call__(self, reference_ts, computed_ts, upper=None, lower=None):
        r = reference_ts.data
        c = computed_ts.data
        e = 100 * sum(c-r) / sum(r)
        return e

    @property
    def unit(self):
        return "rMBE"

    def __str__(self):
        return "rMBE"


class MetricNMAE:
    def __init__(self):
        pass

    def __call__(self, reference_ts, computed_ts, upper=None, lower=None):
        r = reference_ts.data
        c = computed_ts.data
        e = (reference_ts - computed_ts).abs().data / np.nanmean(r)
        return np.nanmean(e)  # MAE

    @property
    def unit(self):
        return "NMAE"

    def __str__(self):
        return "NMAE"

class MetricR2:
    def __init__(self):
        pass

    def __call__(self, reference_ts, computed_ts, upper=None, lower=None):
        r = reference_ts.data
        c = computed_ts.data

        if len(r) == 0 or len(c) == 0:
            return 1

        return r2_score(r, c)

    @property
    def unit(self):
        return "R2"

    def __str__(self):
        return "R2"

class TimeSeriesExamination:
    """
    TimeSeriesExaminator class provides functions and control flow useful during models examination. This class provides
    three main components
        1. it makes a window test/train loop over a dataset
        2. it prepares MTS and statistics as a response form main function
        3. It stores results in predefined file system
            (main directory - has to be provided in constructor if not provided then no storage available)
               -> <date>_<time>_<models count> (ex "231027_0728_1)
                    -> timeseries.xls and statistics.csv
    """
    def __init__(self, directory=None, verbose=0, development_mode=False):
        """
        Constructor
        :param: development_mode = in this mode only one prediction line is prepared. rest is skipped. It allows to make
        prediction very fast but it doesn't produce reliable results.
        """
        self._time_series_sampling_converter = None
        self._enable_cache = False # enable or disable loading model from storage
        self._errors = None
        self._directory = directory
        self._verbose = verbose
        self._x = None
        self._development_mode = development_mode
        self._models = []
        self._mts = None
        self._metrics = []
        self._exogenous_mts = None
        self._cache_file_headers = ["Timestamp", "Dataset", "ModelHash", "ModelStr", "PredictionFile",
                              "window_size", "batch", "predict_horizon",  "predict_window"]

    @property
    def models(self):
        return self._models

    @property
    def dataset(self):
        return self._x

    def register_metric(self, metric_as_callable_obj):
        self.verbose_print(f"Registered metric: {metric_as_callable_obj}")
        self._metrics.append(metric_as_callable_obj)

    def check_inited(self):
        if self._x is None or len(self._models) == 0:
            raise RuntimeError("TimeSeriesExamination. Object not inited." +
            "Use register_x_timeseries and register_model functions before!")

    def check_test_train_done(self):
        if self._mts is None:
            raise RuntimeError("TimeSeriesExamination. Object tested given model yet." +
                               "Use train_test to prepare a report!")

    def verbose_print(self, *args, **kwargs):
        if self._verbose > 0:
            print(f"TimeSeriesExamination:", *args, **kwargs)

    def register_x_timeseries(self, ts, time_series_sampling_converter, exogenous_mts=None):
        self._x = ts
        self._exogenous_mts = exogenous_mts
        self._time_series_sampling_converter = time_series_sampling_converter
        self.verbose_print("Dataset (X) Timeseries registered\n", self._x)

    def register_model(self, model, window_size=None, predict_window=None, batch=None, predict_horizon=None):
        """
        Function appends model to list of models under examination.
        :param: model - instance of object which implements fit(window) and predict(test_window[:,1]) functions accoring
        to sklearn scheme.
        """

        #check if first object has full configuration
        if len(self._models) == 0:
            if window_size is None:
                raise RuntimeError("TimeSeriesExamination.register_model. You have to specify window_size and batch for first registered model.")
            if batch is None:
                batch = 1 # batch has its default value = 1
            if predict_window is None:
                predict_window = 1  # batch has its default value = 1
            if predict_horizon is None:
                predict_horizon = 1  # learn_horizon has its default value = 1
        else:
            # use previously defined element if configuration not provided
            window_size = window_size if window_size is not None else self._models[-1]["window_size"]
            batch = batch if batch is not None else self._models[-1]["batch"]
            predict_window = predict_window if predict_window is not None else self._models[-1]["predict_window"]
            predict_horizon = predict_horizon if predict_horizon is not None else self._models[-1]["predict_horizon"]

        self._models.append({
            "model": model,
            "window_size": window_size,
            "batch": batch, #learn once per batch (decrease learning count to make model faster)
            "predict_horizon": predict_horizon, # defines the prediction period samples_ahead
            "predict_window": predict_window, # defines the prediction period samples_ahead
            # "cachedPrefix": "Mx" -> field defines to column name prefix
            # "cached": mts -> field added while load_directory if cache file was found. obtained results will be used
            # instead of computing new ones
        })
        self.verbose_print(f"Model {model} registered as M{len(self._models)}. Total model counts is: {len(self._models)}")

    def train_test_trajectory(self):
        self.check_inited()
        self.load_directory()
        self._mts = MTimeSeries()
        self._mts[self._x.name] = self._x

        self._errors = MTimeSeries()

        if self._development_mode:
            self.verbose_print(f"===================================")
            self.verbose_print(f" Warning: DEVELOPMENT MODE ENABLED ")
            self.verbose_print(f"===================================")

        # compute max window_size to adjust begining of all prediction
        max_ws = np.max([m["window_size"] for m in self._models])

        for i, m in enumerate(self._models):
            self.create_metric_mts(self._errors, i)

            lh = int(m["predict_horizon"]+1)
            for j in range(lh):
                self._mts[f"M{i}_{j}"] = TimeSeries.zeros(self._x, name=f"{m['model']}_{j}", unit=self._x.unit, value=0)
                self._mts[f"M{i}_{j}_top"] = TimeSeries.zeros(self._x, name=f"{m['model']}_{j}_top", unit=self._x.unit, value=0)
                self._mts[f"M{i}_{j}_bottom"] = TimeSeries.zeros(self._x, name=f"{m['model']}_{j}_bottom", unit=self._x.unit, value=0)
            self.verbose_print(f"Starting interation over (M{i+1}): {m['model']}")

            model = m["model"] #store model object "regressor"
            model.set_path_forecasting() #set data validation to path forecasting

            t = time.time()
            z = 0
            if "cached" in m:
                prefix = m["cachePrefix"]
                for j in range(m["predict_horizon"]):
                    ts = m["cached"][f"{prefix}_{j}"]
                    self._mts[f"M{i}_{j}"] = ts

                for k in range(m["window_size"], len(self._x) -  m["predict_horizon"]):
                    current_prediction_line = z%lh
                    self.compute_metrics(self._errors, i, k, self._x[k:k + m["predict_horizon"]],
                                         self._mts[f"M{i}_{current_prediction_line}"][k:k + m["predict_horizon"]])
                    z = z + 1
            else:
                for window, k, test_window in self._x.model_test_iteration(window_size=m["window_size"],
                                                                           predict_horizon=m["predict_horizon"],
                                                                           verbose=self._verbose,
                                                                           skip=max_ws-m["window_size"],
                                                                           step=1):
                    current_prediction_line = z % lh
                    if self._development_mode and current_prediction_line != 0:
                        z += 1
                        continue

                    if z%m["batch"] == 0:
                        model.fit(window)

                    data = np.zeros((1, 1+m["predict_window"]))
                    # prepare X dataset for model
                    data[0, 0] = test_window.timestamps[0]
                    data[0, 1:] = window.data[-m["predict_window"]:]

                    pred = model.predict(data, predict_horizon=m["predict_horizon"],
                                         time_series_sampling_converter=self._time_series_sampling_converter)
                    #print(pred[0].prediction, "S", pred.shape)
                    # print("k", k, m["window_size"], len(window), len(test_window), len(pred))
                    # print(k, k+m["learn_horizon"])

                    self._mts[f"M{i}_{current_prediction_line}"][k:k+m["predict_horizon"]] = pred[0].prediction
                    self._mts[f"M{i}_{current_prediction_line}_top"][k:k+m["predict_horizon"]] = pred[0].upper
                    self._mts[f"M{i}_{current_prediction_line}_bottom"][k:k+m["predict_horizon"]] = pred[0].lower
                    self._mts[f"M{i}_{current_prediction_line}"][k+m["predict_horizon"]] = None

                    self.compute_metrics(self._errors, i, k, test_window, self._mts[f"M{i}_{current_prediction_line}"][k:k+m["predict_horizon"]])

                    #self._mts[f"M{i}_{current_prediction_line}"][k:k + m["predict_window"]] = current_prediction_line
                    z += 1

            m["execution_time"] = time.time() - t # store model execution time

        self.save_cache_file()
        return self._mts, self._errors

    def train_test_sample_ahead(self):
        self.check_inited()
        self._mts = MTimeSeries()
        self._mts[self._x.name] = self._x
        self._errors = MTimeSeries()

        if self._development_mode:
            self.verbose_print(f"===================================")
            self.verbose_print(f" Warning: DEVELOPMENT MODE ENABLED ")
            self.verbose_print(f"===================================")

        # compute max window_size to adjust beginning of all prediction
        max_ws = np.max([m["window_size"] for m in self._models])
        for i, m in enumerate(self._models):
            self.create_metric_mts(self._errors, i)
            self._mts[f"M{i}"] = TimeSeries.zeros(self._x, name=f"{m['model']}", unit=self._x.unit, value=0)
            self._mts[f"M{i}_top"] = TimeSeries.zeros(self._x, name=f"{m['model']}_top", unit=self._x.unit,value=0)
            self._mts[f"M{i}_bottom"] = TimeSeries.zeros(self._x, name=f"{m['model']}_bottom", unit=self._x.unit, value=0)
            self.verbose_print(f"Starting interation over (M{i + 1}): {m['model']}")
            model = m["model"]  # store model object "regressor"
            model.set_step_ahead_forecasting() # set data validation to sample ahead forecasting
            t = time.time()

            ph = m["predict_horizon"]
            for windows, k, test_windows in self._x.model_test_iteration(window_size=m["window_size"],
                                                           predict_horizon=ph,
                                                           verbose=100,
                                                           skip=max_ws - m["window_size"],
                                                           predict_window= m["predict_window"],
                                                           batch_size=m["batch"]):
                # Check if exogenous params are available. If so, then select aproppiate period and pass to fit and
                # predict functions.
                train_exogenous, test_exogenous = None, None
                if self._exogenous_mts is not None:
                    train_exogenous = self._exogenous_mts[k + ph:k + ph + len(windows)]
                    test_exogenous = self._exogenous_mts[k + ph:k + ph + len(windows)]

                model.fit(windows[0], predict_horizon=ph) #fit data the longest window
                # model should return 2D array shape: (len(windows), 1)
                # print(f"windows: ", len(windows), [len(x) for x in windows[:4]])
                # print(f"test_windows: ", len(test_windows), [len(x) for x in test_windows[:4]])
                # give model prediction windows according to the description below
                #  >----------------> time >-------------------------->
                #                                   ->-->-->--
                #                                   |        |
                # __________________################$~~~~~~~~?_________
                #                   pred_window     |    |   |
                #                    current_position    |   |
                # predict_horizon (period between $ and ?)   |
                #         prediction(value generated by model)
                # prediction window and predict horizon are given into the model. Model generate one prediction for each
                # one prediction window.

                pred = model.predict([window[-m["predict_window"]:] for window in windows],
                                     predict_horizon=ph,
                                     time_series_sampling_converter=self._time_series_sampling_converter)

                self._mts[f"M{i}"][k + ph:k + ph + len(windows)] = pred.prediction.reshape(1,-1)

            m["execution_time"] = time.time() - t  # store model execution time

        y_pred = self._x[max_ws:]
        for i, m in enumerate(self._models):
            prediction = self._mts[f"M{i}"][max_ws:]
            # upper = self._mts[f"M{i}_top"][max_ws:]
            # lower = self._mts[f"M{i}_bottom"][max_ws:]
            for metric in self._metrics:
                n = f"M{i}_{str(metric)}"
                self._errors[n].data[:] = metric(y_pred, prediction)

        return self._mts, self._errors


    def create_metric_mts(self, mts, model):
        for metric in self._metrics:
            n = f"M{model}_{str(metric)}"
            mts[n] = TimeSeries.zeros(self._x, name=n, unit=metric.unit, value=0)

    def compute_metrics(self, mts, model, k, reference, prediction):
        for metric in self._metrics:
            n = f"M{model}_{str(metric)}"
            mts[n][k] = metric(reference, prediction)

    def load_directory(self):
        self.verbose_print(f"Directory loading process.")
        if self._directory is not None:
            self.verbose_print(f" - Looking for {self._directory}")
            if not os.path.exists(self._directory):
                self.verbose_print(f" - Directory not found. Creating...")
                os.makedirs(self._directory, exist_ok=True)
                if not os.path.exists(self._directory):
                    raise RuntimeError(f"TimeSeriesExamination.load_directory: Cannot create directory {self._directory}")
                self.verbose_print(f" - Directory created.")

            cache_file = os.path.join(self._directory, "cache.xlsx")
            if not os.path.exists(cache_file):
                self.verbose_print(f" - Cache file '{cache_file}' not exists. Creating...")
                ss = StandardStorage()
                ss.headers = self._cache_file_headers
                ss.save_excel_format(cache_file)
            cache = StandardStorage()
            cache.load_excel_format(cache_file)
            self.verbose_print(f" - Cache file '{cache_file}' loaded.")

            if self._enable_cache:
                for i, m in enumerate(self._models):
                    self.verbose_print(f" - Searching for cached {str(m['model'])}")
                    h = str(m['model'].hash())
                    self.verbose_print(f" - hash {h}")
                    hash_col = cache.get_column_index("ModelHash")
                    pred_col = cache.get_column_index("PredictionFile")
                    ws_col = cache.get_column_index("window_size")
                    batch_col = cache.get_column_index("batch")
                    ph_col = cache.get_column_index("predict_horizon")
                    pw_col = cache.get_column_index("predict_window")
                    dataset_col = cache.get_column_index("Dataset")
                    for row in cache:
                        if row[hash_col] == h and int(row[ws_col]) == m["window_size"] and int(row[batch_col]) == m["batch"] and \
                                int(row[ph_col]) == m["predict_horizon"] and int(row[pw_col]) == m["predict_window"] and \
                                os.path.isfile(row[pred_col]) and row[dataset_col] == self._x.name:

                            self.verbose_print(f" - found. Cached results will be used.")
                            mts = MTimeSeries()
                            mts.load_excel_format(row[pred_col])
                            if len(mts) > 0:
                                m["cached"] = mts
                                n = mts.names[0].split("_") #"M0_125" -> "M0" "125"
                                m["cachePrefix"] = n[0] # "M0"
                            break
                            # print(mts)
            else:
                self.verbose_print(f" - Cache loading disabled ")
        else:
            self.verbose_print(f"Directory searching disabled.")

    def save_cache_file(self):
        self.verbose_print(f"Directory saving process.")
        if self._directory is not None:
            cache_file = os.path.join(self._directory, "cache.xlsx")
            cache = StandardStorage()
            cache.load_excel_format(cache_file)
            timestamp = time.time()
            ts = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H:%M:%S')
            #print("cache", cache.data, cache.headers)

            for i, m in enumerate(self._models):
                if not "cached" in m:
                    f = os.path.join(self._directory, f"{ts}_{i}.xlsx")
                    cache.append([ts, self._x.name, m["model"].hash(), str(m["model"]), f, m["window_size"], m["batch"],
                                  m["predict_horizon"],  m["predict_window"]])
                    names = []
                    for name in self._mts.names:
                        if name.startswith(f"M{i}_"):
                            names.append(name)

                    if self._enable_cache:
                        self._mts[names].save_excel_format(f)

            #print(cache)
            cache.save_excel_format(path=cache_file)

    def statistics(self, format="txt"):
        """
        Function computes statistics of computed forecasts.
        :param: format - format of TablePrinter "txt" or "latex"
        """
        self.check_test_train_done()
        headers = ["Database", "Model", "Batch", "Window", "Predict_horizon", "Predict_window"]

        for metric in self._metrics:
            headers.append(str(metric))

        headers.append("Ex_time")
        table = TablePrinter(*headers, precision=[0,0,0,0,0,*[3 for _ in self._metrics],0], format=format)

        for i, m in enumerate(self._models):
            row = [str(self._x.name), str(m["model"]), m["batch"], m["window_size"], m["predict_horizon"], m["predict_window"]]

            for metric in self._metrics:
                row.append(np.nanmean(self._errors[f"M{i}_{str(metric)}"].data))

            row.append(m["execution_time"])
            table.append(row)

        return table


    
