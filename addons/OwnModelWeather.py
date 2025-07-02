import sys, ultraimport
import traceback
from datetime import datetime
import numpy as np
ultraimport('__dir__/../../helpers/SolarInsulation.py', 'SolarInsulation', globals=globals())
ultraimport('__dir__/../../helpers/TimestampsProcessing.py', 'TimestampsProcessing', globals=globals())
ultraimport('__dir__/../../helpers/MTimeSeries.py', 'MTimeSeries', globals=globals())
ultraimport('__dir__/../../helpers/TimeSeries.py', 'TimeSeries', globals=globals())
ultraimport('__dir__/BaseModel.py', 'ModelBaseClass', globals=globals())
ultraimport('__dir__/BaseModel.py', 'PredictionResults', globals=globals())
ultraimport('__dir__/../../helpers/TimeSeriesOverlay.py', 'TimeSeriesOverlay', globals=globals())
ultraimport('__dir__/../../helpers/TimeSeriesExamination.py', 'TimeSeriesExamination', globals=globals())

import matplotlib.pyplot as plt


class OwnModelWeather(ModelBaseClass):
    DISPATCH_SOLAR_ELEVATION = 0
    DISPATCH_TIME = 1

    def __init__(self, latitude_degrees, elevation_angle_bins, power_bins,
                 moving_avg_window, interpolation=False,
                 distribution_method=0, version=ModelBaseClass.version(__file__)):
        super().__init__(version=version)
        self._latitude_degrees = latitude_degrees #self.register_param("lat", )
        self._elevation_angle_bins = self.register_param("elev", elevation_angle_bins)
        self._power_bins = self.register_param("pow", power_bins)
        self._moving_avg_window = self.register_param("ma_window", moving_avg_window)
        self._interpolation = self.register_param("int", interpolation)
        self._distribution_method = self.register_param("dm", distribution_method)  # 0-solar elevation angle, 1-time dispatch

        self._is_learnt = False


    def _fit(self, window):
        self._is_learnt = True

    def _predict(self, windows):
        tssc = self.tssc
        self.check_fitted()

        # self._predict_horizon if path forecasting. Else 1
        timestampses = np.zeros((len(windows), self.effective_predict_horizon))

        timestampses[:, 0] = [w.timestamps[-1] for w in windows]

        for i in range(len(windows)):
            for j in range(self.effective_predict_horizon):
                timestampses[i, self.effective_predict_horizon - j - 1] = \
                    (self._predict_horizon - j)*tssc.sampling_interval_seconds + timestampses[i, 0]

        timestampses = np.array([timestampses]).reshape((len(windows), self.effective_predict_horizon))

        ret = np.zeros((len(windows), self.effective_predict_horizon))
        #ret = np.array([[0.0] * self.predict_horizon]).reshape(-1,1)
        #
        # # store parameters during processing
        # previous_ts, previous_bin = 0,0
        # next_ts, next_bin = 0,0
        #
        # for i, xx in enumerate(timestampses):
        #     try:
        #         for j,x in enumerate(xx):
        #             current_ts = x
        #             if self._distribution_method == OwnModelWeather.DISPATCH_SOLAR_ELEVATION:
        #                 tmp = SolarInsulation.elevation(np.array([current_ts]),
        #                                                  latitude_degrees=self._latitude_degrees,
        #                                                  positive_only=self._positive_angles_only,
        #                                                  bins=self._elevation_angle_bins).astype(int)
        #                 current_bin = tmp[0]
        #             else:
        #                 tmp = TimestampsProcessing.day_bins(np.array([current_ts]),bins=self._elevation_angle_bins).astype(int)
        #                 current_bin = tmp[0]
        #
        #             #make linear interpolation
        #             if self._interpolation:
        #                 if next_ts <= current_ts:
        #                     next_ts, next_bin = SolarInsulation.when_elevation_bin(current_bin,
        #                                                          tssc.create_range(current_ts, time_delta=24*3600),
        #                                                          latitude_degrees=self._latitude_degrees,
        #                                                          positive_only=self._positive_angles_only,
        #                                                          bins=self._elevation_angle_bins)
        #
        #                     previous_ts, previous_bin = SolarInsulation.when_elevation_bin(current_bin,
        #                                                          tssc.create_range(current_ts, time_delta=24*3600, reversed=True),
        #                                                          latitude_degrees=self._latitude_degrees,
        #                                                          positive_only=self._positive_angles_only,
        #                                                          bins=self._elevation_angle_bins)
        #
        #                 percent = (current_ts - previous_ts) / (next_ts - previous_ts)
        #                 #print("xx", previous_ts, "<", current_ts,"<", next_ts, percent*100, "%")
        #
        #                 previous_extreme = self._phi["extremes"][f"{str(current_bin)}"][0]
        #                 next_extreme = self._phi["extremes"][f"{str(next_bin)}"][0]
        #                 # according to extremes
        #
        #                 ret[i, j] = previous_extreme + percent * (next_extreme - previous_extreme)
        #             else:
        #                 ret[i, j] = self._phi["extremes"][f"{str(current_bin)}"][0]
        #
        #         # adjust scale
        #         if self._scale_adjustment:
        #             scale_adjust_point = np.mean(windows[i].data[1:])
        #             s = scale_adjust_point / ret[i,0] if ret[i,0] > 0 else 1
        #             # bias = ret[i,0]
        #             # ret[i] = (ret[i] - bias) * s + bias
        #             ret[i] = ret[i] * s
        #
        #         # adjust beginning to last sample
        #         if self._translation_adjustment:
        #             translation_adjust_point = windows[i].data[-1]
        #             ret[i] = ret[i] - (ret[i, 0] - translation_adjust_point)
        #         ret[i, ret[i] < 0] = 0
        #         ret[i, ret[i] > self._max] = self._max
        #     except KeyError as error:
        #         ret[i, 0] = 0
        #
        #     except IndexError as error:
        #         # print("Model predict")
        #         # print("fit", self._phi["kde"], f"{x}")
        #         # print(traceback.format_exc())
        #         # if column not found then none data are available (maybe learning data too short).
        #         ret[i, 0] = 0

        return PredictionResults(ret, lower=None, upper=None)

    def check_fitted(self):
        if not self._is_learnt:
            raise RuntimeError("Model. Use fit function before you use predict function!")

    def _predict_param(self, data):
        """Predict param works like in parent object"""
        tssc = self.tssc
        self.check_fitted()

        X = data[:, 0]
        timestampses = X

        # compute categories - ints are textual categories, due to transformation2D policy
        timestampses = SolarInsulation.elevation(timestampses, latitude_degrees=self._latitude_degrees,
                                                 positive_only=self._positive_angles_only, bins=self._elevation_angle_bins).astype(int)

        ret = np.zeros((X.shape[0], 3))
        for i, x in enumerate(timestampses):
            try:
                ret[i] = self._phi["extremes"][f"{str(x)}"]
            except:
                ret[i] = [0,0,0]

        return ret
