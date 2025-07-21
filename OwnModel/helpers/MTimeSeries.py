import numpy as np
# import ultraimport
import sys
# ultraimport('__dir__/../helpers/TimeSeries.py', 'TimeSeries', globals=globals())
# ultraimport('__dir__/../helpers/StandardStorage.py', 'StandardStorage', globals=globals())
from .TimeSeries import TimeSeries
from .StandardStorage import StandardStorage
import matplotlib.pyplot as plt



def period_bins_transformer(timeSeries, period=10, bins=10):
    """
    Function performs 2d transformation. It looping over TimeSeries (timeSeries). It devides time series into periods, each
    period is divided into bins. For each bing mean is calculated. Then computed means in the same bin in each period are used
    to create new TimeSeries. It is repeated for all bins.
    :param timeSeries: Time series used to perform operation
    :param period: The length of period in observable TimeSeries
    :param bins: number of bins to be used in each period
    :return: MTimeSeries
    """
    #np.set_printoptions(threshold=sys.maxsize)
    #np.set_printoptions(precision=1,suppress=False)

    # cannot specify more
    if period < bins:
        raise ValueError("Period which is {0} cannot be lower than bins which is {1}".format(period,bins))

    new_timedelta = timeSeries.timedelta() * period / bins
    ts = timeSeries.resampling(new_timedelta)
    tss = [
        ts[i::bins]
        for i in range(0, bins, 1)
    ]

    return tss

class MTimeSeries():
    """
        MTimeSeries stands for M(ultiple)TimeSeries. It is a wrapper class for working with multiple TimeSeries
        or with multi dimentional data.

        - Examples

        Creation
            pv_prod = TimeSeries.from_standard_storage(ss2, name="X1", timestamps_column="X0", unit="kW")
            cons = TimeSeries.from_standard_storage(ss2, name="X2", timestamps_column="X0", unit="kW")
            sell = TimeSeries.from_standard_storage(ss2, name="X3", timestamps_column="X0", unit="zl/MWh")
            buy = TimeSeries.zeros(pv_prod, name="BPrice", unit="zl/MWh", value=650)

            mts = MTimeSeries([pv_prod,cons,sell,buy])

        Filtering by column name
            mts["X1"]       # access only one column
            mts[["X1","X2"]] # access many columns
            mts[64:1452+46] # access selected rows range from all columns

            mts.plot() # plot all columns

            Notice: mts["X1"] returns TimeSeries instead of MTimeSeries. If you need one-column MTimeSeries use mts[["X1"]] !
                 mts["X1"] = mts[["X2"]]

        Performing function on all stored TimeSeries
            mts.perform_func(TimeSeries.moving_avg, window_size=40)

        Updating columns
            mts["X1"] = mts["X1"].moving_avg(window_size=40)
            mts[mts.names[0]] = mts["X2"] #access to X1.MA(40) due to moving_avg name changing policy

        Adding columns
            mts["XN"] = mts["X1"].moving_avg(window_size=40) # if column XN is not defined inside mts then it will be
                                                             # created instead of raising exception

        Transformation2D
            # in case you have a function to which change timestamp into group use this:
            mts = mts[["X1"]].transformation2D(transformer="function_bins", callable_obj_or_func=SolarInsulation.elevation, latitude_degrees=16.884)
            # in case you want to use predefined time division use this:
            mts = mts[["X1"]].transformation2D(transformer="period_bins", period=288, bins=24)
            # in case you have an array use this:
            array = SolarInsulation.elevation(mts["X1"].timestamps,
                                 latitude_degrees=16.884, positive_only=True)
            mts = mts[["X1"]].transformation2D(transformer="array_bins", array=array)

            # after transforming you still have timeseries ! use histogram to compute density !
            mts = mts.perform_func(TimeSeries.histogram,bins=24)

            # after you have histograms plot them in the form you prefer
            mts.plot()
                or
            mts2.plot_histogram_grid(rows=2,cols=5) # if you want a set of histograms use this f, to display them
    """
    def __init__(self, time_series_objects=None):
        """
        :param time_series_objects: list of or one instance of TimeSeries class, which object should handle
        """
        self._time_series = []
        self._init_timeseries(time_series_objects)

    @staticmethod
    def _validate_timeseries(time_series_objects):
        """
        Function checks if time_series_objects is valid timeseries array. It also checks if timeseries require agreement
        This helper function is used by MTimeSeries.plot and MTimeSeries._init_timeseries to ensure correctness of
        inputted data
        """
        ts_len = None
        ts_0 = None
        ts_m1 = None
        agree_timestamps = False
        for i, ts in enumerate(time_series_objects):
            # not working for derived classes
            # if ts.__class__.__name__ != TimeSeries.__name__:
            #     raise ValueError(
            #         "Cannot assign not TimeSeries object into time series array. Incorrect object at position {0}".format(
            #             i))
            # else:
            if ts_len is None:
                ts_len = len(ts.timestamps)
                ts_0 = ts.timestamps[0]
                ts_m1 = ts.timestamps[-1]
            else:
                if ts_len != len(ts.timestamps) or ts_0 != ts.timestamps[0] or ts_m1 != ts.timestamps[-1]:
                    agree_timestamps = True

        return agree_timestamps

    def _init_timeseries(self, time_series_objects):

        if time_series_objects is None:
            return

        if isinstance(time_series_objects, list):
            pass
        elif isinstance(time_series_objects, tuple):
            time_series_objects = list(time_series_objects)
        else:
            #neither list nor tuple
            time_series_objects = [time_series_objects]

        #if histogram or other case when x axis is not an time axis
        is_x_time = all([ts.is_x_time for ts in time_series_objects])
        if not is_x_time:
            self._time_series = time_series_objects
            return


        agree_timestamps = MTimeSeries._validate_timeseries(time_series_objects)

        if agree_timestamps:
            print("Need to agree timestamps")
            self._time_series = TimeSeries.agree_timestamps(time_series_objects)
        else:
            self._time_series = time_series_objects

    def agree_timestamps(self):
        self._time_series = TimeSeries.agree_timestamps(self._time_series)

    def perform_func(self, func, *arg, **kwargs):
          return MTimeSeries([func(ts, *arg, **kwargs) for ts in self.tss])

    @property
    def tss(self):
        """
        Access to the time series list
        :return:
        """
        return self._time_series

    @property
    def tss_count(self):
        return len(self.tss)

    @property
    def ts1(self):
        """
        A shortcut function for accessing to the first stored timeseries
        :return: TimeSeries
        """
        self._check_emptiness()
        return self._time_series[0]

    @property
    def timestamps(self):
        return self.ts1.timestamps

    @property
    def is_empty(self):
        return len(self._time_series) == 0

    @property
    def names(self):
        return [ts.name for ts in self._time_series]

    @property
    def units(self):
        return [ts.unit for ts in self._time_series]

    def __iter__(self):
        pass

    def __getitem__(self, item):

        if isinstance(item, str):
            #print("getitem", item)
            t = [ts for ts in self.tss if ts.name == item]
            if len(t) > 0:
                return t[0]
            else:
                raise IndexError("Not {0} column found".format(item) + "Available: ", str([ts.name for ts in self.tss]))

        if isinstance(item, (list, tuple)):
            if all(isinstance(i, bool) for i in item):
                return self._get_by_boolean_array(item)
            else:
                return self._get_by_names(item)


        if isinstance(item, (float, int)):
            #print("len", len(self.tss), item)
            return self.tss[int(item)]

        mts = MTimeSeries([ts[item] for ts in self._time_series])

        #reverse name changing
        for i, ts in enumerate(self.tss):
            mts.tss[i].name = ts.name
            #print(mts.tss[i].name)
        return mts

    def __setitem__(self, key, value):
        #one column assignment
        if isinstance(key, str):
            if key not in self.names:
                # append new time series
                index = len(self.tss)
                self._time_series.append(None)
            else:
                # edit existing once
                index = self.names.index(key)
            #print(index)
            if value.__class__.__name__ == MTimeSeries.__name__:
                if len(value.tss) > 1 or len(value.tss) == 0:
                    raise ValueError("Cannot assign MTimeSeries multicolumn or empty into TimeSeries['key']. Specify one column MTimeSeries r-value")
                # assign only first column (it is one column MTS)
                self.tss[index] = value.ts1
            elif value.__class__.__name__ == TimeSeries.__name__:
                # assign only column
                value.name = key  # update name to the keyname
                self.tss[index] = value
        else:
            raise ValueError("Key must be a string type. Given {0}".format(type(key)))

    def plot(self, **kwargs):
        self._check_emptiness()
        self._check_adjustment()
        self._time_series[0].plot(self._time_series[1:], **kwargs)

    def plot_static(self, one_colors=False, alpha=1, fig=None, ax=None, mtss=[], vlines=[], x_title=""):
        if fig is None and ax is None:
            fig, ax = plt.subplots(1,1)
        ax.set_xlabel(x_title)
        ax.set_ylabel("Normalized power")

        mtss = [self, *mtss]

        for vline in vlines:
            ax.axvline(x=vline["x"], color=vline["color"])

        for i,mts in enumerate(mtss):
            color = None
            if one_colors is not None:
                if isinstance(one_colors, (tuple, list)):
                    color = one_colors[i]
                else:
                    color = one_colors

            for ts in mts.tss:
                ax.plot(ts.timestamps, ts.data, alpha=alpha, color=color)


        return fig, ax


    def plot_histogram_grid(self, rows = 6, cols = 4, bandwidth=0.1, kde_factor=0.1, fig=None, ax=None, x_limit=None):
        """
        Function is used to displaying a grid of histograms after a set of processing:

        # change singular mts into a vector
        mts2 = mts2.transformation2D(transformer="function_bins", callable_obj_or_func=SolarInsulation.elevation,
                                     latitude_degrees=16.884, positive_only=True)
        # transform vector of filtered data into histograms
        mts2 = mts2.perform_func(TimeSeries.histogram, bins=10)
        # display computed histograms
        mts2.plot_histogram_grid(rows=2,cols=5)
        """
        if fig is None and ax is None:
            fig, ax = plt.subplots(rows, cols)

        names = [t.name for t in self.tss]
        #try sort if not possible then assign natural order
        try:
            srt = list([y for y, x in sorted(zip(self.tss, names), key=lambda pair: int(pair[1]))])
        except:
            srt = self.tss
        #print([s.name for s in srt])
        y_limit = None
        if y_limit is None:
            y_limit = max([ts.max for ts in self.tss])
        if x_limit is None:
            x_limit = max([ts.timestamps[-1] for ts in self.tss])

        for r in range(rows):
            for c in range(cols):
                index = r * cols + c
                if index < len(self.tss):
                    if rows == 1 and cols == 1:
                        _ax = ax
                    elif rows == 1:
                        _ax = ax[c]
                    elif cols == 1:
                        _ax = ax[r]
                    else:
                        _ax = ax[r, c]

                    ts = srt[r * cols + c]
                    # print(x_plot, np.exp(score))
                    _ax.set_xlim(0, max(ts.timestamps))
                    # print(f"r={r}, c={c}")
                    # print(np.array([[ts,d, p, s] for ts, d, p, s in zip(ts.timestamps, ts.data, x_plot, np.exp(score))]))
                    kde_ts = ts.kernel_density_estimation(bandwidth) * kde_factor
                    # extreme_max = kde_ts.extreme_maximum
                    # extreme_min = kde_ts.extreme_minimum
                    #print("extreme", extreme_max, extreme_min)
                    _ax.fill_between(kde_ts.timestamps, kde_ts.data, alpha=0.2, color="black")

                    # _ax.scatter(x=extreme_max[:, 0], y=extreme_max[:, 1], s=8, color="red")
                    # _ax.scatter(x=extreme_min[:, 0], y=extreme_min[:, 1], s=8, color="green")
                    w = max(ts.timestamps) / (len(ts.timestamps)+1)
                    _ax.bar(x=ts.timestamps, height=ts.data, alpha=0.8, width=w, align='edge')
                    _ax.set_title(ts.name, fontsize=9)
                    _ax.set_ylim([0,y_limit])
                    _ax.set_xlim([0,x_limit])
                    if r < rows-1:
                        _ax.set_xticks([])
                    if c > 0:
                        _ax.set_yticks([])

        return fig, ax


    def _get_by_names(self, names):
        tss = []
        for n in names:
            index = self.names.index(n)
            tss.append(self.tss[index])
        return MTimeSeries(tss)

    def _get_by_boolean_array(self, boolean_array):
        for index, _ in enumerate(self.tss):
            tss.append(self.tss[boolean_array])
        return MTimeSeries(tss)

    def _check_emptiness(self):
        if self.is_empty:
            raise ValueError("MTimeSeries is empty, cannot perform selected method.")

    def _check_adjustment(self):
        if MTimeSeries._validate_timeseries(self.tss):
            raise ValueError("MTimeSeries contains not adjusted TimeSeries. It means that some function like 'plot' cannot be done.\n"
                             "Function of this object is limited to storing an array of TimeSeries. If you want to perform\n"
                             "more functions use MTimeSeries.agree_timestamps(self) function. MTimeSeries: \n" + str(self))

    def _check_singularity(self):
        if self.is_empty or len(self.names) > 1:
            raise ValueError("MTimeSeries is not singular, cannot perform selected method.")

    def transformation2D(self, transformer, **kwargs):
        """
        Function transforms singular MTimeSeries (containing only one TimeSeries) into many TimeSeries by dividing data
        into separeted subgroups according to assigment function
        :param transformer:
        :param kwargs:
        :return:
        """
        self._check_singularity()

        names = []

        if transformer == "period_bins":
            period = kwargs["period"] if kwargs["period"] is not None else 10
            bins = kwargs["bins"] if kwargs["bins"] is not None else 10
            tss = period_bins_transformer(self.ts1, period=period, bins=bins)
            names = [ts.name for ts in tss]

        if transformer == "function_bins" or transformer == "array_bins":
            if transformer == "function_bins" :
                callable_obj_or_func = kwargs["callable_obj_or_func"]
                del kwargs["callable_obj_or_func"]
                if callable_obj_or_func is None:
                    raise ValueError("transformation2D for transformation function_bins needs callable_obj_or_func argument specified")
                #  transform timestamps into nts which contains a finite number of group indicies.
                #  len(nts) == len(self.tt1.timestamps), len(nts_unique) << len(nts). nts is not rising or falling !
                #  It contains information: which group sample[i] belongs to.
                nts = callable_obj_or_func(self.ts1.timestamps, **kwargs)
            else:
                if kwargs["array"] is None:
                    raise ValueError(
                        "transformation2D for transformation array_bins needs array argument specified")
                nts = kwargs["array"]
                if len(nts) != len(self.ts1):
                    raise ValueError("transformation2D for transformation array_bins needs array same length as timestamps lenght")

            nts_unique = np.unique(np.array(nts))
            # print("Tranformation2D", nts_unique, nts)
            tss = []
            # this function can be performed only on mts which is made of only ONE ts (singular mts)!
            for group in nts_unique:
                names.append("{0}".format(group))
                tss.append(self.ts1[nts == group])

        # equalization based on cutting longer series
        if "equalize" not in kwargs or kwargs["equalize"] == True:
            # compute length of the shortest timeseries
            min_lengths = min(len(ts) for ts in tss)
            # equalize length if flag is enable
            tss = [ts[0:min_lengths] for ts in tss]

        #equalization based on removing first and last elements
        else:
            tss = [tss[i] for i in range(1, len(tss)-1)]

        # agree timestamps (shift all timestamps left to the beginning of period)
        ts = list(range(len(tss[0].timestamps)))
        for i in range(0, len(tss)):
            #tss[i].timestamps = tss[0].timestamps
            tss[i].timestamps = ts
        for name, ts in zip(names, tss):
            ts.name = name

        return MTimeSeries(tss)

    def __len__(self):
        if self.is_empty:
            return 0
        else:
            return len(self.tss[0])

    def __str__(self):
        txt = "--- Multiple Time Series --- \n"

        if self._check_emptiness() == False:
            txt += " - empty"
            return txt
        txt += " Signals:\n{0}\n".format("\n".join(["  {0} [{1}] - {2}".format(ts.name, ts.unit, len(ts)) for ts in self.tss]))

        txt += " Size: {0}\n".format(len(self))
        txt += " TS count: {0}\n".format(len(self.tss))
        if len(self) > 1:
            txt += " Time range: {0} - {1}\n".format(self.ts1.timestamps[0], self.ts1.timestamps[-1])
        elif len(self) == 1:
            txt += "Time range: {0} - {0}\n".format(self.ts1.timestamps[0])
        return txt


    def flatten(self):
        """
        Function makes MTimeSeries flat. It joins all TSs into numpy array according to the order of tss in tss array
        """
        return np.array([ts.data for ts in self.tss]).flatten('C')

    def to_numpy(self):
        """
        Function returns MTimeSeries as 2D numpy array
        """
        return np.array([ts.data for ts in self.tss])

    def from_numpy(self, numpy_array):
        """
        Function reinitialize ts.data for each timeseries inside the object - be careful it doesn't check data consistence
        """
        for i,ts  in enumerate(self.tss):

            ts.data = numpy_array[i].copy()

    def clone(self):
        tss = []
        for ts in self.tss:
            tss.append(ts.clone())

        return MTimeSeries(tss)

    def data(self):
        return np.array([ts.data for ts in self.tss])

    def save_excel_format(self, path):
        """Save MTS using StandardStorage Engine"""
        ss = StandardStorage()
        ss.headers = ["Timestamp", *self.names]
        ss.data = np.zeros((len(self.ts1), len(self.tss) + 1)) # + timestamps column
        ss.data[:, 0] = self.ts1.timestamps
        for i, ts in enumerate(self.tss):
            ss.data[:, i+1] = ts.data
        ss.save_excel_format(path)

    def save_csv_format(self, path):
        """Save MTS using StandardStorage Engine"""
        ss = StandardStorage()
        ss.headers = ["Timestamp", *self.names]
        ss.data = np.zeros((len(self.ts1), len(self.tss) + 1)) # + timestamps column
        ss.data[:, 0] = self.ts1.timestamps
        for i, ts in enumerate(self.tss):
            ss.data[:, i+1] = ts.data
        ss.save_csv_format(path)

    def load_excel_format(self, path):
        """Load MTS using StandardStorage Engine"""
        ss = StandardStorage()
        ss.load_excel_format(path)

        for i in range(1, len(ss.headers)):
            c = np.array(ss.get_column_by_index(i), dtype=np.float)
            ts = TimeSeries(values=c,
                                             timestamps=ss.get_column_by_index(0),
                                             name=ss.headers[i],
                                             unit="kW")
            self[ss.headers[i]] = ts

    def max(self, default=0):
        a = np.full(len(self.ts1), -np.inf)
        for i in range(len(self)):
            for j in range(len(self.tss)):
                if a[i] < self.tss[j].data[i]:
                    a[i] = self.tss[j].data[i]
        a[a == -np.inf] = default
        return TimeSeries(values=a, timestamps=self.timestamps, name="mts.max()",
                          unit="")

    def min(self, default=0):
        a = np.full(len(self.ts1), +np.inf)
        for i in range(len(self)):
            for j in range(len(self.tss)):
                if a[i] > self.tss[j].data[i]:
                    a[i] = self.tss[j].data[i]

        a[a == np.inf] = default
        return TimeSeries(values=a, timestamps=self.timestamps, name="mts.min()",
                          unit="")
    def mean(self):
        a = np.zeros(len(self.ts1))
        for i in range(len(self)):
            for j in range(len(self.tss)):
                if not np.isnan(self.tss[j].data[i]):
                    a[i] += self.tss[j].data[i]
            a[i] = a[i] / len(self.tss)


        return TimeSeries(values=a, timestamps=self.timestamps, name="mts.mean()",
                          unit="")

    def positive_mean(self):
        a = np.zeros(len(self.ts1))
        for i in range(len(self)):
            counter = 0
            for j in range(len(self.tss)):
                if not np.isnan(self.tss[j].data[i]) and self.tss[j].data[i] > 0:
                    a[i] += self.tss[j].data[i]
                    counter += 1
            if counter > 0:
                a[i] = a[i] / counter
        return TimeSeries(values=a, timestamps=self.timestamps, name="mts.positive_mean()",
                          unit="")
    @property
    def T(self):
        return self.transposition()

    def transposition(self):
        """Function transpose dataset"""
        count = len(self) # number of timeseries in transposed mts
        ts = []

        timestamps = list(range(len(self.tss)))
        for i in range(count):
            values = [self.tss[index].data[i] for index in range(len(self.tss))]
            ts.append(TimeSeries(values=values, timestamps = timestamps, name=f"T({i})"))
        return MTimeSeries(ts)