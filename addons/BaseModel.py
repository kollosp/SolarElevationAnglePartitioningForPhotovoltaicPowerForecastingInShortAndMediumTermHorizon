import sys, ultraimport
import traceback
from datetime import datetime
import numpy as np
from hashlib import md5
import os
ultraimport('__dir__/../../helpers/SolarInsulation.py', 'SolarInsulation', globals=globals())
ultraimport('__dir__/../../helpers/MTimeSeries.py', 'MTimeSeries', globals=globals())
ultraimport('__dir__/../../helpers/TimeSeries.py', 'TimeSeries', globals=globals())
ultraimport('__dir__/../../helpers/StandardStorage.py', 'StandardStorage', globals=globals())
ultraimport('__dir__/../../helpers/StandardStorage.py', 'read_csv_database_file', globals=globals())

class PredictionResult:
    """
    A product of one prediction
    """
    def __init__(self, prediction=None, upper=None, lower=None):
        self._prediction = prediction #np. Array or TimeSeries
        self._upper = upper #np. Array or TimeSeries Confidence interval upper boundary
        self._lower = lower #np. Array or TimeSeries Confidence interval lower boundary

    def check_consistence(self):
        """
        Function checks correctness of data inserted to the object
        """
        # if upper is None and lower is None:
        #     return
        if len(self._prediction) != len(self._upper) or \
            len(self._lower) != len(self._upper):
            raise RuntimeError("Predict Result is not consistent. Check correctness of provided data: "
                               f"len(self._prediction)=={len(self._prediction)}, len(self._upper)=={len(self._upper)} "
                               f"len(self._lower)=={len(self._lower)}")

    @property
    def prediction(self):
        return self._prediction
    @property
    def upper(self):
        return self._upper
    @property
    def lower(self):
        return self._lower

    @prediction.setter
    def prediction(self, a):
        self._prediction = a
        self.check_consistence()

    @upper.setter
    def upper(self, a):
        self._upper = a
        self.check_consistence()

    @lower.setter
    def lower(self, a):
        self._lower = a
        self.check_consistence()

    def __len__(self):
        return len(self._prediction)

    def __str__(self):
        if self._lower is None or self._upper is None:
            return str(self._prediction) #"[" + ','.join([f"{p}" for p in self._prediction]) + f"\nLength: {len(self)}]"
        return '\n'.join([f"{l} < {p} < {u}" for l,p,u in zip(self._lower, self._prediction, self._upper)]) + f" Length: {len(self)}"

class PredictionResults:
    """
    Class contains a list of predict results. Each predict result is a product of one prediction.
    """
    def __init__(self, results=None, upper=None, lower=None):
        self._results = None
        if isinstance(results, np.ndarray):
            self.from_2d_array(results, upper, lower)
        else:
            self._results = results if results is not None else []

    def from_2d_array(self, data, upper=None, lower=None):
        self._results = []

        if upper is not None and len(upper) == len(data) and lower is not None and len(lower) == len(data):
            for i, d in enumerate(data):
                self._results.append(PredictionResult(data[i], upper[i], lower[i]))
        else:
            for i,d in enumerate(data):
                self._results.append(PredictionResult(d))

    @property
    def shape(self):
        if len(self._results) > 0:
            return len(self._results), len(self._results[0])
        return 0,0

    @property
    def prediction(self):
        return np.array([d.prediction for i,d in enumerate(self._results)])

    @property
    def upper(self):
        return np.array([d.upper for i,d in enumerate(self._results)])

    @property
    def lower(self):
        return np.array([d.lower for i,d in enumerate(self._results)])

    def append(self, predict_result):
        self._results.append(predict_result)

    def __getitem__(self, item):
        return self._results[item]

    def __str__(self):
        return "{" + "\n".join([f"[{pr}]" for pr in self._results]) + f"Shape: {self.shape}" + "}"

class ModelBaseClass:
    def __init__(self, name=None, version=__file__):
        self._name = name if name is not None else type(self).__name__
        self._params = {
            "v": version if version is not None else self.version()
        }
        self._predict_horizon = None
        self._time_series_sampling_converter = None
        self._prediction_mode = 0 # 0 - path forecasting
                                  # 1 - step ahead forecasting

    def describe(self):
        raise RuntimeError("Model does not implement describe function.")

    @property
    def is_path_forecasting(self):
        return self._prediction_mode == 0

    @property
    def is_step_ahead_forecasting(self):
        return self._prediction_mode == 1

    def set_path_forecasting(self):
        self._prediction_mode = 0

    @property
    def effective_predict_horizon(self):
        if self.is_path_forecasting:
            return self._predict_horizon
        elif self.is_step_ahead_forecasting:
            return 1

    def set_step_ahead_forecasting(self):
        self._prediction_mode = 1

    @staticmethod
    def version(path):
        path = os.path.realpath(path)
        m_time = os.path.getmtime(path)
        return datetime.fromtimestamp(m_time).strftime('%Y%m%d%H%M%S')

    def fit(self, window, predict_horizon=None, time_series_sampling_converter=None):
        if predict_horizon is not None:
            self._predict_horizon = predict_horizon
        self._time_series_sampling_converter = time_series_sampling_converter
        return self._fit(window)

    def predict(self, windows, predict_horizon=None, time_series_sampling_converter=None):
        """
        windows is list of windows (timeseries) where each window inside has size bigger then predict_horizon
        """
        if predict_horizon is not None:
            self._predict_horizon = predict_horizon
        self._time_series_sampling_converter = time_series_sampling_converter

        #ret value should be PredictionResults
        ret_value = self._predict(windows)

        ph = self.predict_horizon if self.is_path_forecasting else 1

        try:
            #chech correctness of produced response
            if len(ret_value.shape) == 2 and ret_value.shape[0] == len(windows) and ret_value.shape[1] == ph:
                return ret_value

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            raise RuntimeError(f"{self.param_string()}.predict - incorrect return value from overwritten _predict function! "
                               f"returned type is {type(ret_value)} should be PredictionResults, len(ret_value.shape) == {len(ret_value.shape)} shoud be 2 "
                               f"ret_value.shape[0] == {ret_value.shape[0]} shoud be {len(windows)}, ret_value.shape[1] == {ret_value.shape[1]} should be {ph}")


        raise RuntimeError(
            f"{self.param_string()}.predict - incorrect returned array structure form overwritten _predict function. "
            "Condition not met: len(ret_value.shape) == 2 and ret_value.shape[0] == len(windows) and ret_value.shape[1] == self.predict_horizon "
            "for 'path forecasting' or 1 if 'step ahead forecasting'. "
            f"Returned: len(ret_value.shape) == {len(ret_value.shape)}, ret_value.shape == {ret_value.shape} should be {(len(windows), ph)} ")

    def predict_param(self, data, time_series_sampling_converter=None):
        # ret value should be np array
        ret_value = self._predict_param(data)
        self._time_series_sampling_converter = time_series_sampling_converter

        try:
            # check correctness of produced response
            if len(ret_value.shape) == 2 and ret_value.shape[0] == len(data):
                return ret_value

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            raise RuntimeError(
                f"{self.param_string()}.predict - incorrect return type form overwritten _predict function! returned type is"
                f" {type(ret_value)} should be np.array.")

        raise RuntimeError(
            f"{self.param_string()}.predict - incorrect returned array structure form overwritten _predict function. "
            "Condition not met: len(ret_value.shape) == 2 and ret_value.shape[0] == len(data) "
            f"Returned: len(ret_value.shape) == {len(ret_value.shape)}, ret_value.shape == {ret_value.shape} should be {len(data)} ")

    def _predict_param(self, data):
        """
        Function returns parameter used during prediction data has the same format as in predict function, however the
        response is different. Response is 2d array. len(array)  == len(data) array.shape[1] depends on model and
        contains parameters produced for data provided.
        """
        return np.array([[0] for _ in range(data.shape[0])])

    def _fit(self, window):
        pass

    def _predict(self, data):
        """
        function predicts future values.
        data = [[timestamp, ts[i-1], ts[i-2], ..., ts[i-n]],[timestamp, ...], ... , [..]]
            data[j] contains an list of arguments for model. The first element is timestamp. Following elements are past
            data starting from the newest ending with the oldest (in this case i-index define nuber of past data).
        :return: numpy 2d array at size of len(data) x self.predict_horizon where in each row future values are stored. The
                 higher index the more distant data
        """

        # default Model is identity (AR(1))
        return np.array([[data[i, -1] for _ in range(self.predict_horizon)] for i in range(data.shape[0])])

    @property
    def predict_horizon(self):
        return self._predict_horizon

    @property
    def tssc(self):
        return self._time_series_sampling_converter

    @property
    def name(self):
        return self._name

    def register_param(self, parameter, value, default_value=None):
        if default_value is not None:
            self._params[parameter] = value if value is not None else default_value
        else:
            self._params[parameter] = value

        return self._params[parameter]

    def param_string(self):
        """
        Default model's string representation is ModelName(p1=v1,p2=v2,...,pn=vn). It can be change by overwriting
        describe_str funtion.
        """
        return self.name + "(" + ", ".join([f"{value}:{self._params[value]}" for parameter, value in enumerate(self._params)]) + ")"

    def describe_str(self):
        return self.param_string()

    def __str__(self):
        return self.name + "(" + ",".join([f"{value}:{self._params[value]}" for parameter, value in enumerate(self._params)]) + ")"
        #return self.name + "(" + ",".join([f"{self._params[value]}" for parameter, value in enumerate(self._params)]) + ")"

    def hash(self):
        return md5(str(self).encode()).hexdigest()