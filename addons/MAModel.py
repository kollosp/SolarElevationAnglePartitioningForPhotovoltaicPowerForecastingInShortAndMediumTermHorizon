
import sys, ultraimport
import traceback
from datetime import datetime
import numpy as np
ultraimport('__dir__/../../helpers/SolarInsulation.py', 'SolarInsulation', globals=globals())
ultraimport('__dir__/../../helpers/MTimeSeries.py', 'MTimeSeries', globals=globals())
ultraimport('__dir__/../../helpers/TimeSeries.py', 'TimeSeries', globals=globals())
ultraimport('__dir__/../../helpers/StandardStorage.py', 'StandardStorage', globals=globals())
ultraimport('__dir__/../../helpers/StandardStorage.py', 'read_csv_database_file', globals=globals())
ultraimport('__dir__/BaseModel.py', 'ModelBaseClass', globals=globals())
ultraimport('__dir__/BaseModel.py', 'PredictionResults', globals=globals())

class MAModel(ModelBaseClass):
    def __init__(self, ma_window_size=1, version=ModelBaseClass.version(__file__)):
        super(MAModel, self).__init__(version=version)
        self._ma_window_size = self.register_param("ma_window_size", ma_window_size)

    def _predict(self, windows):
        # self._predict_horizon if path forecasting. Else 1
        ret = np.zeros((len(windows), self.effective_predict_horizon))

        # for each dataset sample
        for i in range(len(windows)):
            # compute MA using already made predictions
            for j in range(ret.shape[1]):
                w = np.zeros(self._ma_window_size)
                if j > self._ma_window_size:
                    w = ret[i,j-self._ma_window_size:j]
                elif self._ma_window_size > j > 0:
                    w[-j:] = ret[i, :j]
                    w[:-j] = windows[i].data[-self._ma_window_size+j:]
                else:
                    w = windows[i].data[-self._ma_window_size:]

                ret[i,j] = np.mean(w)
        return PredictionResults(ret, upper=ret, lower=ret)

    def _fit(self, window):
        pass