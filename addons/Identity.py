import sys, ultraimport
import traceback
from datetime import datetime
import numpy as np
ultraimport('__dir__/../../helpers/SolarInsulation.py', 'SolarInsulation', globals=globals())
ultraimport('__dir__/../../helpers/MTimeSeries.py', 'MTimeSeries', globals=globals())
ultraimport('__dir__/../../helpers/TimeSeries.py', 'TimeSeries', globals=globals())
ultraimport('__dir__/../../helpers/StandardStorage.py', 'StandardStorage', globals=globals())
ultraimport('__dir__/../../helpers/StandardStorage.py', 'read_csv_database_file', globals=globals())

ultraimport('__dir__/../../helpers/TimeSeries.py', 'TimeSeriesSamplingConverter', globals=globals())
ultraimport('__dir__/BaseModel.py', 'ModelBaseClass', globals=globals())
ultraimport('__dir__/BaseModel.py', 'PredictionResults', globals=globals())

class Identity(ModelBaseClass):
    def __init__(self, delay_in_samples=TimeSeriesSamplingConverter().day, version=ModelBaseClass.version(__file__)):
        super().__init__(version=version)
        self._delay_in_samples = self.register_param("delay_in_samples", delay_in_samples)

    def _predict(self, data):
        # ret = np.zeros((len(data), predict_horizon))
        ret = np.array([[data[i].data[-self._delay_in_samples] for _ in range(self.effective_predict_horizon)]
                        for i in range(len(data))])
        return PredictionResults(ret, upper=ret, lower=ret)

    def _fit(self, window):
        pass