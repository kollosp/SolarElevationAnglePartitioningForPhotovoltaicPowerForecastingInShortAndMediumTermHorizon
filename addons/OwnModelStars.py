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

"""
Model preapares stairs like forecast
"""
class OwnModelStars(ModelBaseClass):
    DISPATCH_SOLAR_ELEVATION = 0
    DISPATCH_TIME = 1
    POLICY_PESSIMISTIC = 0
    POLICY_NEUTRAL = 1
    POLICY_OPTIMISTIC = 2
    POLICY_QUANTITIVE = 3

    @staticmethod
    def separation_search(mx_extremes, mi_extremes):
        try:
            sep = mx_extremes[int(len(mx_extremes) / 2), 0]
        except:
            # print(sep)
            sep = 0
        # even parity. Search for minimum
        if len(mx_extremes) % 2 == 0 and len(mx_extremes) > 1:
            mx_0 = mx_extremes[int((len(mx_extremes) - 1) / 2), 0]
            mx_1 = mx_extremes[int(len(mx_extremes) / 2), 0]

            try:
                mi = mi_extremes[0, 0]
            except:
                mi = 0

            for i in range(len(mi_extremes)):
                if mx_0 <= mi_extremes[i, 0] <= mx_1:
                    mi = mi_extremes[i, 0]
            sep = mi

        # compute expected values for groups
        # print("OwnModel.separation_search", sep)
        return int(sep)

    def __init__(self, latitude_degrees, elevation_angle_bins, power_bins,
                 moving_avg_window, translation_adjustment=False, scale_adjustment=False,
                 extremes_neighbourhood_adjustment=False, force_policy=None, positive_only=False,
                 distribution_method=0, version=ModelBaseClass.version(__file__)):
        super().__init__(version=version)
        self._latitude_degrees = latitude_degrees #self.register_param("lat", )
        self._elevation_angle_bins = self.register_param("elev", elevation_angle_bins)
        self._power_bins = self.register_param("pow", power_bins)
        self._moving_avg_window = self.register_param("ma_window", moving_avg_window)
        self._translation_adjustment = self.register_param("t_aj", translation_adjustment)
        self._scale_adjustment = self.register_param("s_aj", scale_adjustment)
        self._extremes_neighbourhood_adjustment = self.register_param("en_aj", extremes_neighbourhood_adjustment)
        self._force_policy = self.register_param("policy", force_policy)
        self._positive_angles_only = self.register_param("po", positive_only)
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
        if self._distribution_method == OwnModel.DISPATCH_SOLAR_ELEVATION:
            mts2 = mts2.transformation2D(transformer="array_bins",
                                         array=SolarInsulation.elevation(mts2["PV"].timestamps,
                                                                              latitude_degrees=latitude_degrees,
                                                                              positive_only=self._positive_angles_only,
                                                                              bins=elevation_angle_bins).astype(int).astype(str))
        elif self._distribution_method == OwnModel.DISPATCH_TIME:
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
        # self._fit_counter += 1
        # if self._fig is None and self._fit_counter > 5:
        #     self._fig, _ = mts.plot_histogram_grid(rows=int(self._elevation_angle_bins / 10) + 1, cols=10)
        #     plt.show()

        self._phi["kde"] = mts.perform_func(TimeSeries.kernel_density_estimation, bandwidth=0.1)
        self._phi["histograms"] = mts
        self._is_learnt = True

        self._phi["extremes"] = {}

        for name in self._phi["kde"].names:

            # max of interest (if odd parity then the separation point. in even parity find minimum just after this max)

            if self._phi["histograms"][name].most_common_value() == 0:
                policy = [
                    0 # for pv only during the night there is only one value
                ]
            else:
                kde = self._phi["kde"][name]
                if self._force_policy == OwnModel.POLICY_PESSIMISTIC:
                    mx_extremes = self._phi["kde"][name].extreme_maximum_indexes
                    mi_extremes = self._phi["kde"][name].extreme_minimum_indexes
                    # defining separation points of policies
                    sep = OwnModel.separation_search(mx_extremes, mi_extremes)
                    policy = [kde[:sep].weighted_mean]
                elif self._force_policy == OwnModel.POLICY_NEUTRAL:
                    policy = [kde.weighted_mean]
                elif self._force_policy == OwnModel.POLICY_OPTIMISTIC:
                    mx_extremes = self._phi["kde"][name].extreme_maximum_indexes
                    mi_extremes = self._phi["kde"][name].extreme_minimum_indexes
                    # defining separation points of policies
                    sep = OwnModel.separation_search(mx_extremes, mi_extremes)
                    policy = [kde[sep:].weighted_mean]
                elif self._force_policy == OwnModel.POLICY_QUANTITIVE:
                    policy = [np.argmax(kde.data) * max(kde.timestamps) / len(kde.timestamps) ]
                else:
                    raise ValueError(f"OwnModel incorrect force_policy value {self._force_policy}. Pass on of correct"
                                     f" values")

            self._phi["extremes"][name] = policy

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

        # compute categories - ints are textual categories, due to transformation2D policy
        if self._distribution_method == OwnModel.DISPATCH_SOLAR_ELEVATION:
            timestampses = SolarInsulation.elevation(timestampses.flatten(), latitude_degrees=self._latitude_degrees,
                                                    positive_only=self._positive_angles_only, bins=self._elevation_angle_bins).astype(int)
        elif self._distribution_method == OwnModel.DISPATCH_TIME:
            timestampses = TimestampsProcessing.day_bins(timestampses.flatten(),bins=self._elevation_angle_bins).astype(int)

        timestampses = np.array([timestampses]).reshape((len(windows), self.effective_predict_horizon))

        ret = np.zeros((len(windows), self.effective_predict_horizon))
        #ret = np.array([[0.0] * self.predict_horizon]).reshape(-1,1)

        for i, xx in enumerate(timestampses):
            try:
                #
                for j,x in enumerate(xx):
                    # if self._distribution_method == OwnModel.DISPATCH_SOLAR_ELEVATION:
                    #     timestampses = SolarInsulation.elevation(timestampses.flatten(),
                    #                                              latitude_degrees=self._latitude_degrees,
                    #                                              positive_only=self._positive_angles_only,
                    #                                              bins=self._elevation_angle_bins).astype(int)
                    # else:
                    #     timestampses = TimestampsProcessing.day_bins(timestampses.flatten(),
                    #                                                  bins=self._elevation_angle_bins).astype(int)
                    #
                    extremes = self._phi["extremes"][f"{str(x)}"]
                    # according to extremes
                    # ret[i, j] = extremes[self._force_policy]
                    ret[i, j] = extremes[0] # -> only one value inside since fit function takes into account policy.

                # adjust scale
                if self._scale_adjustment:
                    scale_adjust_point = np.mean(windows[i].data[1:])
                    s = scale_adjust_point / ret[i,0] if ret[i,0] > 0 else 1
                    # bias = ret[i,0]
                    # ret[i] = (ret[i] - bias) * s + bias
                    ret[i] = ret[i] * s

                # adjust beginning to last sample
                if self._translation_adjustment:
                    translation_adjust_point = windows[i].data[-1]
                    ret[i] = ret[i] - (ret[i, 0] - translation_adjust_point)
                ret[i, ret[i] < 0] = 0
                ret[i, ret[i] > self._max] = self._max
            except KeyError as error:
                ret[i, 0] = 0

            except IndexError as error:
                # print("Model predict")
                # print("fit", self._phi["kde"], f"{x}")
                # print(traceback.format_exc())
                # if column not found then none data are available (maybe learning data too short).
                ret[i, 0] = 0

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
