
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
import tensorflow as tf


class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    # if self.label_index is None:
    #   return inputs
    # result = inputs[:, :, self.label_index]
    return inputs[:, -1, 0]

class BaselineTF(ModelBaseClass):
    def __init__(self, version=ModelBaseClass.version(__file__)):
        super(BaselineTF, self).__init__(version=version)
        #self._ma_window_size = self.register_param("ma_window_size", ma_window_size)

    def _predict(self, data):

        ret = np.zeros((len(data), self.predict_horizon))
        # for each dataset sample

        return PredictionResults(ret, upper=ret, lower=ret)

    def _fit(self, window):
        pass