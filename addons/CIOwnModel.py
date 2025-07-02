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


class CIOwnModel(ModelBaseClass):
    """
        CIOwnModel abb. Confidence interval own model. Model prepares prediction and a confidence interval
    """
    def __init__(self, latitude_degrees, elevation_angle_bins, power_bins,
                 moving_avg_window, translation_adjustment=False, scale_adjustment=False,
                 version=ModelBaseClass.version(__file__)):
        super().__init__(version=version)
        self._latitude_degrees = self.register_param("latitude", latitude_degrees)
        self._elevation_angle_bins = self.register_param("elevation", elevation_angle_bins)
        self._power_bins = self.register_param("power", power_bins)
        self._moving_avg_window = self.register_param("ma_window", moving_avg_window)
        self._translation_adjustment = self.register_param("t_aj", translation_adjustment)
        self._scale_adjustment = self.register_param("s_aj", scale_adjustment)

        self._phi = {
            "histograms": [],
            "kde": []
        }
        self._is_learnt = False
        self._max = 0

    @staticmethod
    def transformation(ts, latitude_degrees, elevation_angle_bins, power_bins):
        mts2 = MTimeSeries()
        mts2["PV"] = ts

        # on this step flat pv signal is being grouped according to sun elevation angle and divided by the noon
        # (after and before noon) are seperated. elevation_angle_bins define how many groups will be created. Angles are
        # from range -1..1 if positive_only == False otherwise range is 0..1.
        mts2 = mts2.transformation2D(transformer="array_bins",
                                     array=SolarInsulation.elevation(mts2["PV"].timestamps,
                                                                          latitude_degrees=latitude_degrees,
                                                                          positive_only=True,
                                                                          bins=elevation_angle_bins).astype(int).astype(str))

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

        self._phi["kde"] = mts.perform_func(TimeSeries.kernel_density_estimation, bandwidth=1)
        self._phi["histograms"] = mts
        self._is_learnt = True

        self._phi["extremes"] = {}

        for name in self._phi["kde"].names:
            # defining separation points of policies
            mx_extremes = self._phi["kde"][name].extreme_maximum_indexes
            # max of interest (if odd parity then the separation point. in even parity find minimum just after this max)
            sep = mx_extremes[int(len(mx_extremes) / 2), 0]
            # even parity. Search for minimum
            if len(mx_extremes) % 2 == 0 and len(mx_extremes) > 1:
                mx_0 = mx_extremes[int((len(mx_extremes)-1) / 2), 0]
                mx_1 = mx_extremes[int(len(mx_extremes) / 2), 0]
                mi_extremes = self._phi["kde"][name].extreme_minimum_indexes
                mi = mi_extremes[0, 0]

                for i in range(len(mi_extremes)):
                    if mx_0 <= mi_extremes[i, 0] <= mx_1:
                        mi = mi_extremes[i, 0]
                sep = mi

            # compute expected values for groups
            sep = int(sep)
            #print("sep", sep, mx_extremes, mi_extremes)

            if self._phi["histograms"][name].most_common_value() == 0:
                policy = [
                    0,0,0 # for pv only during the night there is only one value
                ]
            else:
                policy = [
                    self._phi["kde"][name][:sep].weighted_mean, # pessimistic
                    self._phi["kde"][name].weighted_mean, # neutral
                    self._phi["kde"][name][sep:].weighted_mean, # optimistic
                ]



            self._phi["extremes"][name] = policy
            # self._phi["max"][name] = self._phi["kde"][name].max

    def _predict(self, data):

        tssc = self.tssc
        self.check_fitted()
        X = data[:, 0]
        predict_window = data[:, 1:]
        predict_window_len = predict_window.shape[1]
        timestampses = np.zeros((X.shape[0], predict_window_len + self.predict_horizon))

        timestampses[:, predict_window_len] = X

        for i in range(X.shape[0]):
            for j in range(predict_window_len):
                timestampses[i, j] = timestampses[i, predict_window_len] - (predict_window_len - j)*tssc.sampling_interval_seconds

            for j in range(self.predict_horizon):
                timestampses[i, j+predict_window_len] = j*tssc.sampling_interval_seconds + timestampses[i, predict_window_len]

        # compute categories - ints are textual categories, due to transformation2D policy
        timestampses = SolarInsulation.elevation(timestampses.flatten(), latitude_degrees=self._latitude_degrees,
                                                positive_only=True, bins=self._elevation_angle_bins).astype(int)
        timestampses = np.array([timestampses]).reshape((X.shape[0], predict_window_len + self.predict_horizon))

        ret = np.zeros((X.shape[0], self.predict_horizon))
        upper = np.zeros((X.shape[0], self.predict_horizon))
        lower = np.zeros((X.shape[0], self.predict_horizon))
        #ret = np.array([[0.0] * self.predict_horizon]).reshape(-1,1)

        for i, xx in enumerate(timestampses):
            # power generated in window
            window_power = predict_window[i, :].sum()
            history_timestamps = xx[:predict_window_len]
            future_timestamps = xx[predict_window_len:]
            try:
                residual_prediction = np.zeros(predict_window_len)
                for j, x in enumerate(history_timestamps):
                    extremes = self._phi["extremes"][f"{str(x)}"]
                    residual_prediction[j] = (extremes[-1] - extremes[0]) / 2 + extremes[0]

                #print("residual_prediction",xx,"\n", history_timestamps,"\n",residual_prediction)
                # print("residual_prediction",window_power.sum(),residual_prediction.sum())

                for j,x in enumerate(future_timestamps):
                    # kde = self._phi["kde"][f"{str(x)}"]
                    extremes = self._phi["extremes"][f"{str(x)}"]
                    #print("print", x, np.argmax(kde.data)* max(kde.timestamps) / len(kde.timestamps))
                    # according to extremes
                    ret[i, j] = ((extremes[-1] - extremes[0]) / 2 + extremes[0])
                    upper[i, j] = extremes[-1]
                    lower[i, j] = extremes[0]

                rs = residual_prediction.sum()
                wp = window_power.sum()

                if rs != 0 and wp != 0:
                    factor =  wp / rs
                    bin = int(self._elevation_angle_bins/2)
                    midday_extremes = self._phi["extremes"][f"{str(bin)}"]



                    for j, x in enumerate(future_timestamps):
                        ret[i, j] =  ret[i, j] * factor
                        upper[i, j] = upper[i, j] * factor
                        lower[i, j] = lower[i, j] * factor

            except IndexError as error:
                print("Model predict")
                print("fit", self._phi["kde"], f"{x}")
                print(traceback.format_exc())
                # if column not found then none data are available (maybe learning data too short).
                ret[i, 0] = 0
        return PredictionResults(ret, upper=upper, lower=lower)

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
                                                 positive_only=True, bins=self._elevation_angle_bins).astype(int)

        ret = np.zeros((X.shape[0], 3))
        for i, x in enumerate(timestampses):
            ret[i] = self._phi["extremes"][f"{str(x)}"]

        return ret
