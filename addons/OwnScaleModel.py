import sys, ultraimport
import traceback
from datetime import datetime
import numpy as np
ultraimport('__dir__/../../helpers/SolarInsulation.py', 'SolarInsulation', globals=globals())
ultraimport('__dir__/../../helpers/TimestampsProcessing.py', 'TimestampsProcessing', globals=globals())
ultraimport('__dir__/../../helpers/MTimeSeries.py', 'MTimeSeries', globals=globals())
ultraimport('__dir__/../../helpers/TimeSeries.py', 'TimeSeries', globals=globals())
ultraimport('__dir__/../../helpers/StandardStorage.py', 'StandardStorage', globals=globals())
ultraimport('__dir__/../../helpers/StandardStorage.py', 'read_csv_database_file', globals=globals())
ultraimport('__dir__/BaseModel.py', 'ModelBaseClass', globals=globals())
ultraimport('__dir__/BaseModel.py', 'PredictionResults', globals=globals())

import matplotlib.pyplot as plt


class OwnScaleModel(ModelBaseClass):
    DISPATCH_SOLAR_ELEVATION = 0
    DISPATCH_TIME = 1
    POLICY_PESSIMISTIC = 0
    POLICY_NEUTRAL = 1
    POLICY_OPTIMISTIC = 2


    def __init__(self, latitude_degrees, elevation_angle_bins, power_bins,
                 moving_avg_window, translation_adjustment=False, scale_adjustment=False,
                 extremes_neighbourhood_adjustment=False, force_policy=None, distribution_method=1, version=ModelBaseClass.version(__file__)):
        super().__init__(version=version)
        self._latitude_degrees = self.register_param("lat", latitude_degrees)
        self._elevation_angle_bins = self.register_param("elev", elevation_angle_bins)
        self._power_bins = self.register_param("pow", power_bins)
        self._moving_avg_window = self.register_param("ma_window", moving_avg_window)
        self._translation_adjustment = self.register_param("t_aj", translation_adjustment)
        self._scale_adjustment = self.register_param("s_aj", scale_adjustment)
        self._extremes_neighbourhood_adjustment = self.register_param("en_aj", extremes_neighbourhood_adjustment)
        self._force_policy = self.register_param("policy", force_policy)
        self._positive_angles_only = True
        self._distribution_method = self.register_param("dm", distribution_method)  # 0-solar elevation angle, 1-time dispatch
        self._fit_counter = 0
        self._phi = {
            "histograms": [],
            "kde": []
        }
        self._is_learnt = False
        self._max = 0
        self._fig = None

    @property
    def histograms(self):
        return self._phi["histograms"]

    def transformation(self, ts, latitude_degrees, elevation_angle_bins, power_bins):
        mts2 = MTimeSeries()
        mts2["PV"] = ts

        # on this step flat pv signal is being grouped according to sun elevation angle and divided by the noon
        # (after and before noon) are seperated. elevation_angle_bins define how many groups will be created. Angles are
        # from range -1..1 if positive_only == False otherwise range is 0..1.
        if self._distribution_method == OwnScaleModel.DISPATCH_SOLAR_ELEVATION:
            mts2 = mts2.transformation2D(transformer="array_bins",
                                         array=SolarInsulation.elevation(mts2["PV"].timestamps,
                                                                              latitude_degrees=latitude_degrees,
                                                                              positive_only=self._positive_angles_only,
                                                                              bins=elevation_angle_bins).astype(int).astype(str))
        elif self._distribution_method == OwnScaleModel.DISPATCH_TIME:
            mts2 = mts2.transformation2D(transformer="array_bins",
                                         array=TimestampsProcessing.day_bins(mts2["PV"].timestamps, bins=elevation_angle_bins).astype(int).astype(str))

        # on this step power is quantificated. The number of bins is power_bins. Now mts2 is a set of histograms. Which
        # presents the distribution of pv power for various elevation angles
        # presentation: mts2.plot_histogram_grid(rows=3, cols=10)
        mts2 = mts2.perform_func(TimeSeries.histogram, bins=power_bins, max=ts.max, min=0, normalize_0_1=True, fixed_names=True)

        return mts2

    def _fit(self, window):
        self._max = window.max*1.1
        w = window.moving_avg(window_size=self._moving_avg_window)
        w = w.roll(shift=-int(self._moving_avg_window / 2))  # rolling is available only on past data!!! AVOID looking ahead!
        mts = self.transformation(w,
                                  latitude_degrees=self._latitude_degrees,
                                  elevation_angle_bins=self._elevation_angle_bins,
                                  power_bins=self._power_bins)
        self._fit_counter += 1
        if self._fig is None and self._fit_counter > 5:
            self._fig, _ = mts.plot_histogram_grid(rows=int(self._elevation_angle_bins / 10) + 1, cols=10)
            plt.show()

        self._phi["kde"] = mts.perform_func(TimeSeries.kernel_density_estimation, bandwidth=0.1)
        self._phi["histograms"] = mts
        self._is_learnt = True
        self._phi["max"] = None
        self._phi["extremes"] = {}

        for name in self._phi["kde"].names:
            if self._phi["histograms"][name].most_common_value() == 0:
                policy = [
                    0 # for pv only during the night there is only one value
                ]
            else:
                policy = [
                    self._phi["kde"][name].weighted_mean, # neutral
                ]


            self._phi["extremes"][name] = policy
            # find maximum possible value to scaling
            if self._phi["max"] is None:
                self._phi["max"] = policy[0]
            else:
                self._phi["max"] = policy[0] if policy[0] > self._phi["max"] else self._phi["max"]

    @staticmethod
    def future_timestamps(windows, effective_predict_horizon, predict_horizon, tssc):
        timestampses = np.zeros((len(windows), effective_predict_horizon))
        timestampses[:, 0] = [w.timestamps[-1] for w in windows]

        for i in range(len(windows)):
            for j in range(effective_predict_horizon):
                timestampses[i, effective_predict_horizon - j - 1] = \
                    (predict_horizon - j) * tssc.sampling_interval_seconds + timestampses[i, 0]

        return timestampses

    def timestamps_to_categories(self, timestamps):
        shape = timestamps.shape

        # compute categories - ints are textual categories, due to transformation2D policy
        categories = SolarInsulation.elevation(timestamps.flatten(), latitude_degrees=self._latitude_degrees,
                                                 positive_only=self._positive_angles_only,
                                                 bins=self._elevation_angle_bins).astype(int)
        return np.array([categories]).reshape(shape)

    def _predict(self, windows):
        tssc = self.tssc
        self.check_fitted()

        shape = (len(windows), self.effective_predict_horizon)
        timestamps = self.future_timestamps(windows, self.effective_predict_horizon, self._predict_horizon, self.tssc)
        time_categories = self.timestamps_to_categories(timestamps)
        # self._predict_horizon if path forecasting. Else 1

        past_timestamps = np.array([w.timestamps for w in windows])
        #print("past_timestamps", past_timestamps)
        past_time_categories = self.timestamps_to_categories(past_timestamps)

        ret = np.zeros(shape)
        #ret = np.array([[0.0] * self.predict_horizon]).reshape(-1,1)

        for i in range(len(windows)):

            try:


                # compute in past prediction
                in_past_prediction = np.zeros(len(past_time_categories[i]))
                for j, x in enumerate(past_time_categories[i]):
                    extremes = self._phi["extremes"][f"{str(x)}"]
                    in_past_prediction[j] = extremes[0]

                past_prediction_integral = sum(in_past_prediction)
                past_reference_integral = sum(windows[i].data)

                scale = 1
                if past_reference_integral > 0 and past_prediction_integral > 0:
                    scale = past_reference_integral / past_prediction_integral

                m = self._phi["max"]
                if scale * m > 10:
                    scale = 10 / m
                # print(past_time_categories[i, -1], past_prediction_integral, past_reference_integral,
                #       past_reference_integral / past_prediction_integral, "scale:", scale)
                # compute in future prediction

                for j, x in enumerate(time_categories[i]):
                    extremes = self._phi["extremes"][f"{str(x)}"]
                    ret[i, j] = extremes[0] * scale



            except IndexError as error:
                # print("Model predict")
                # print("fit", self._phi["kde"], f"{x}")
                # print(traceback.format_exc())
                # if column not found then none data are available (maybe learning data too short).
                ret[i, 0] = 0

        return PredictionResults(ret, lower=ret, upper=ret)

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
