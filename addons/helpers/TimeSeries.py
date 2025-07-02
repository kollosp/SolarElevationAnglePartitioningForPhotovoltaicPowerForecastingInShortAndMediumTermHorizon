
import numpy as np
import numbers
import ultraimport
from numbers import Number
ultraimport('__dir__/../helpers/ArraySlider.py', 'ArraySlider', globals=globals())
ultraimport('__dir__/../helpers/StandardStorage.py', 'StandardStorage', globals=globals())
ultraimport('__dir__/../helpers/TimeSeriesDataset.py', 'TimeSeriesDataset', globals=globals())
ultraimport('__dir__/../helpers/TimeSeriesDataset.py', 'TimeSeriesDatasetXY', globals=globals())
import matplotlib.pyplot as plt
from statsmodels.tsa import stattools
from datetime import datetime
from sklearn.linear_model import LinearRegression
from math import floor
from sklearn.neighbors import KernelDensity
import time
from dtaidistance import dtw
from math import floor

class TimeSeriesSamplingConverter:
    def __init__(self, sampling_interval_seconds=300, begin_timestamp=0):
        self._sampling_interval_seconds = sampling_interval_seconds
        self._begin_timestamp = begin_timestamp

    @property
    def day(self):
        seconds_during_day = 60*60*24
        return int(seconds_during_day / self._sampling_interval_seconds)

    @property
    def sampling_interval_seconds(self):
        return self._sampling_interval_seconds

    def create_range(self, begin, end=None, time_delta=None, reversed=False):
        begin = int(begin)
        if end is not None:
            if not reversed:
                return list(range(begin, int(end), self._sampling_interval_seconds))
            else:
                l = list(range(int(end), begin, self._sampling_interval_seconds))
                l.reverse()
                return l
        elif time_delta is not None:
            if not reversed:
                return list(range(begin, begin+time_delta, self._sampling_interval_seconds))
            else:
                l = list(range(begin - time_delta, begin, self._sampling_interval_seconds))
                l.reverse()
                return l
        else:
            raise ValueError("TimeSeriesSamplingConverter.create_range: either 'end' or 'time_delta' have to be defined. "
                             "'End' is timestamp limiting range, time_delta is number of seconds to be added to 'begin'. ")

    @property
    def days30(self):
        return int(30 * self.day)

    @property
    def year(self):
        return int(365 * self.day)

    @property
    def hour(self):
        seconds_during_hour = 60 * 60
        return int(seconds_during_hour / self._sampling_interval_seconds)

class TimeSeriesIterator:
    """
        Class TimeSeriesIterator is an iterator class for looping over TimeSeries

        Time series window iterator. It allows to go over timeseries using selected window. It starts iteration form
        sample at position: window_size-th.

    """
    def __init__(self, time_series, window_size=1, verbose=0, skip=0, step=1):
        """
        :param time_series: TimeSeries to be iterated
        :param window_size: Window size for window sliding iteration
        :param skip: Number of samples at the beginning of TimeSeries to be skipped
        :param step: Number of samples to move at each iteration
        :param verbose:
        """
        self._is_tuple = type(time_series) is tuple
        self._len = 0
        self._step = step
        self._verbose = verbose
        if self._is_tuple:
            self._len = len(time_series[0])
        else:
            self._len = len(time_series)
        self._window_size = window_size
        self._time_series = time_series
        self._skip = skip
        self._current_index = self._window_size + skip
        self._loop_iterator = 0
        self._execution_time = time.time()
        self._execution_start = time.time()
    def __iter__(self):
        return self

    @property
    def count(self):
        return (self.end_iteration() - self._window_size - self._skip) / self.increment_by() + 1

    def verbose_stop(self):
        t = time.time()
        ts = t - self._execution_start
        print(f"It: {self._loop_iterator+1} / {self.count:.0f} ~ 100% | Done in {ts:.0f}s")

    def verbose(self):
        t = time.time()
        dt = t - self._execution_time
        ts = t - self._execution_start
        c = self.count
        li =self._loop_iterator+1
        p =li/c
        print(f"It: {li} / {c:.0f} ~ {100*p:.0f}% | T: {ts:.0f}s | LoopT: {dt:.3f}s | "
              f"ETA: {dt*(c-li)/self._verbose:.0f}s | EPT: {(c/li * ts):.0f}s")
        self._execution_time = t

    def end_iteration(self):
        """
        Function defines when to stop iteration
        """
        return self._len

    def increment_by(self):
        """
        Function define show to increment current index for each step
        """
        return self._step

    def increment_current_index(self):
        """
        Function increments current index using predefine function that can be override in nested classes
        """
        self._current_index += self.increment_by()
        self._loop_iterator += 1

    def __next__(self):

        if self._current_index < self.end_iteration():
            i = self._current_index

            if self._is_tuple:
                window = tuple([ts[i - self._window_size:i] for ts in self._time_series])
            else:
                window = self._time_series[i - self._window_size:i]
            if self._verbose > 0 and self._loop_iterator % self._verbose == 0:
                self.verbose()

            self.increment_current_index()
            return window, i
        else:
            if self._verbose > 0:
                self.verbose_stop()
            raise StopIteration


class TimeSeriesModelTestIterator(TimeSeriesIterator):
    def __init__(self, time_series, window_size=1, predict_horizon=1, verbose=0, skip=0, step=1, predict_window=None, batch_size=None,):
        super().__init__(time_series, window_size, verbose, skip, step)
        self._predict_window = predict_window if predict_window is not None else window_size
        self._predict_horizon = predict_horizon
        self._batch_size = batch_size

        if self._batch_size is not None:
            self._step = self._batch_size

    # @override - this method overrides method from parent object
    def end_iteration(self):
        return self._len - self._predict_horizon

    # @override - this method overrides method from parent object
    def increment_by(self):
        return self._step

    def next_no_batch(self, window, i):
        if self._is_tuple:
            test_window = tuple([ts[i:i + self._predict_horizon] for ts in self._time_series])
        else:
            test_window = self._time_series[i:i + self._predict_horizon]

        return window, i, test_window

    def next_batch(self, window, i):
        """
        Batch contains many single iterations.
        windows: is a list of windows where first element has size of windows_size and rest elements have predict_window
        size. The first element is used to fit model when rest of them are used only for prediction.

        test_windows: is a list of true data to compare with predictions made by models.

        """
        effective_batch_count = self._batch_size
        # effective on the last batch which is smaller.
        if effective_batch_count + i > self.end_iteration():
            effective_batch_count = self.end_iteration() - i
        windows = [window]
        test_windows = []

        # first batch is window! next batches has only predict_window size (may it be shorter than window)
        for j in range(i+1, i + effective_batch_count):
            if self._is_tuple:
                w = tuple([ts[j - self._predict_window:j] for ts in self._time_series])
            else:
                w = self._time_series[j - self._predict_window:j]
            windows.append(w)

        #normaly prepare test windows for all instances in batch
        for j in range(i, i + effective_batch_count):
            if self._is_tuple:
                w = tuple([ts[j:j + self._predict_horizon] for ts in self._time_series])
            else:
                w = self._time_series[j:j + self._predict_horizon]
            test_windows.append(w)

        return windows, i, test_windows

    def __next__(self):
        window, current_index = super().__next__()

        if self._batch_size is None:
            return self.next_no_batch(window, current_index)
        else:
            return self.next_batch(window, current_index)

class TimeSeries():
    """
    Examples:
        ts1 = TimeSeries.from_numpy(np.arange(0,1000,0.2), timestamps=None, name=None)
        # loading from excel via StandardStorage
        ss = StandardStorage()
        ss.load_excel_format("environment/tests/test.xlsx")
        ts2 = TimeSeries.from_standard_storage(ss,name="y_true",timestamps_column="timestamp")

        # iterate over timeseries using specified window
        for ts_w, i in ts2.window_iteration(50):
            pass

        # iterate over timeseries sample by sample
        for ts_w, i in ts2:
            pass

        # compute ts statistics
        ts2_moving_avg = ts2.moving_avg(window_size=30)
        ts2_diff = ts2.diff(order=1)
        ts2_diff2 = ts2.diff(order=2)

        # Arithmetical operations
        ts_ones = TimeSeries.from_numpy(np.ones(1000), name="ts_ones")
        ts_3ies = ts_ones + ts_ones + ts_ones
        ts_zeros = ts_3ies - ts_ones - ts_ones - ts_ones

        # iterating over multiple timeSeries with the same timestampses
        for (w_ones, w_3), i in TimeSeries.zip(ts_ones, ts_3ies, window_size=15):
            ts_zeros[i] = w_ones[0] * w_3[0]*2
            pass

        #plotting
        ts_ones.plot() # plot alone
        ts_ones.plot([ts_3ies]) # plot two curves on the same graph
        ts_ones.plot(conv_func=TimeSeries.acf) # plot auto correlation function
        ts_ones.plot(conv_func=TimeSeries.pacf) # plot partial auto correlation function
        ts_ones.plot(conv_func=TimeSeries.fft) # plot fast fourier transformation
        TimeSeries.static_plot([ts_ones, ts_3ies])

        Computing features of the TimeSeries:

        def is_production(ts_window, max_value=1, min_value=0):
            return max_value if ts_window.data[-1] > 1 else min_value
        is_prod = pv_prod.perform_window_function(function=is_production, window_size=25, max_value=5, min_value=1)
        R2 = pv_prod.perform_window_function(function=TimeSeries.LinearRegR2, window_size=140).moving_avg(12)

    """

    @staticmethod
    def first_january():
        dt = datetime(2020, 1, 1)
        return int(dt.timestamp())

    @staticmethod
    def agree_timestamps(tss):
        """
            Function agrees timestampses of ts1 and ts2. Two timeseries are expanded to have same timestamps. For example
                ts1: XXXX
                ts2:    YYYY
                will be changed into
                ts1: XXXX000
                ts2: 000YYYY
                empty places are filled with zeros.

            Function is usefull when you want to store multiple TimeSeries in MTimeSeries and you are not sure that they have
            same timestamps
        """
        #print("TimeSeries.agree_timestamps", tss[0], tss[0].timedelta())
        td = tss[0].timedelta()
        for ts in tss:
            if ts.timedelta() != td:
                raise ValueError("All tss time series have to have the same timedelta."
                                 " Use resampling method to fix this issue. First {0} found {1}".format(td, ts.timedelta()))

        c_timestamps = np.concatenate([ts.timestamps for ts in tss])
        min_ts = np.min(c_timestamps)
        max_ts = np.max(c_timestamps)
        #print("TimeSeries.agree_timestamps",min_ts, max_ts, td)
        timestamps = np.arange(min_ts, max_ts, td)

        #print("len", len(timestamps))

        ret = []
        for ts in tss:
            #end_l = np.argmax(timestamps>ts.timestamps[0]) -1 #
            end_l = np.argmin(timestamps < ts.timestamps[0]) #
            start_r = np.argmax(timestamps > ts.timestamps[-1]) # position of right zeros. If start_r == 0
            if start_r == 0:
                start_r = len(timestamps) # this is the longest ts. NO padding at the right
            if end_l == 0:
                #start_r = start_r - 1
                pass

            # zeros_l = TimeSeries(values=0, timestamps=timestamps[0:end_l])

            # create base:
            t = TimeSeries(values=0, timestamps=timestamps)
            # fill base with ts:
            print("ts len: ", len(t), end_l, start_r, len(ts.data))
            t.data[end_l:start_r] = ts.data


            # #skip in case of th
            # if start_r != 0:
            #     zeros_r = TimeSeries(values=0, timestamps=timestamps[start_r:])
            #     t = TimeSeries.concatenate([zeros_l, ts, zeros_r])
            # else:
            #     t = TimeSeries.concatenate([zeros_l, ts])

            t.name = ts.name
            t.unit = ts.unit
            ret.append(t)

        return ret

    @staticmethod
    def concatenate(tss):
        """
        Function concatenates all time series in tss into one long timeseries
        :param tss: a list of TimeSeries
        :return: Concatenated timeseries
        """
        timestamps = np.concatenate([ts.timestamps for ts in tss])
        data = np.concatenate([ts.data for ts in tss])
        return TimeSeries(values=data,timestamps=timestamps, name="c({0})".format("+".join([ts.name for ts in tss])), unit=tss[0].unit)

    @staticmethod
    def LinearRegR2(ts_window):
        """
            Function computes R^2 metric of linear regression to the given SimeSeries
        """
        _x, _y = ts_window.timestamps.reshape(-1, 1), ts_window.data.reshape(-1, 1)
        if np.count_nonzero(_y == 0) > 0:
            return 0

        lr = LinearRegression().fit(_x, _y)
        ret = 1 - lr.score(_x, _y)
        return ret

    def threshold(self, level, max_value, min_value):
        array = np.array([max_value if d > level else min_value for d in self.data])
        return TimeSeries(values=array, timestamps=self.timestamps, name=self.name + "THS()", unit=self.unit)
        #return [max_value if ts[i] > level else min_value for i in ts]

    @staticmethod
    def acf(window):
        acf_w = stattools.acf(window, nlags=1000)
        x = np.arange(1, len(acf_w) + 1, 1)
        return x, acf_w

    @staticmethod
    def fft(window):
        fft = np.fft.fft(window)
        freq = np.fft.fftfreq(window.shape[-1])
        # x = np.arange(len(fft))
        # take only positive half
        return freq[:int(fft.shape[0] / 2)], fft[:int(fft.shape[0] / 2)]

    @staticmethod
    def pacf(window):
        maxlags = 10000
        nlags = maxlags if len(window) > 2 * maxlags else int(len(window) / 2) - 1
        acf_w = stattools.pacf(window, nlags=nlags)
        x = np.arange(1, len(acf_w) + 1, 1)

        return x, acf_w

    @staticmethod
    def dtw_window(ts1,ts2,window_size, roll=False):
        ts = TimeSeries()
        ts.name = "DTW({0}, {1})".format(ts1.name, ts2.name)
        ts.unit = ""
        array = np.zeros(len(ts1))
        for (t1, t2), i in TimeSeries.zip(ts1, ts2, window_size=window_size):
            array[i] = dtw.distance_fast(t1.data, t2.data, use_pruning=True)
        if roll:
            array = np.roll(array, int(-window_size / 2))  # roll elements if flag is set True
        ts._set_series(array, ts1.timestamps.copy())
        return ts

    @staticmethod
    def subtraction_window(ts1, ts2, window_size, roll=False):
        """
        Function performs window subtraction
        """
        ts = TimeSeries()
        ts.name = "SUB({0}, {1})".format(ts1.name, ts2.name)
        ts.unit = ""
        array = np.zeros(len(ts1))
        for (t1, t2), i in TimeSeries.zip(ts1, ts2, window_size=window_size):
            array[i] = np.sum(np.abs(t1.data-t2.data))
        if roll:
            array = np.roll(array, int(-window_size / 2))  # roll elements if flag is set True
        ts._set_series(array, ts1.timestamps.copy())
        return ts

    @staticmethod
    def division_window(ts1, ts2, window_size):
        """
        Function performs window division. TS1 is devided by mean value in a window of ts2
        """
        ts = TimeSeries()
        ts.name = "DIVW({0}, {1})".format(ts1.name, ts2.name)
        ts.unit = ""
        ts2e = ts2.moving_avg(window_size=window_size, roll=True)

        ts._set_series(np.array([t1/t2 if t2 > 0 else t1 for t1,t2 in zip(ts1.data, ts2e.data)]), ts1.timestamps.copy())
        return ts


    @staticmethod
    def static_plot(tss, **kwargs):
        if len(tss) > 1:
            tss[0].plot(tss[1:], **kwargs)
        else:
            tss[0].plot(**kwargs)
    @staticmethod
    def datetime_from_timestamp(timestamp):
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def timestamp_from_datetime(datetime_str, str_format):
        if str_format == "date": str_format =  '%Y-%m-%d'
        if str_format == "time": str_format =  '%Y-%m-%d %H:%M:%S'
        if isinstance(datetime_str, list):
            return [int(datetime.timestamp(datetime.strptime(s, str_format))) for s in datetime_str]
        else:
             return int(datetime.timestamp(datetime.strptime(datetime_str, str_format)))

    @staticmethod
    def zip(*args, window_size=1, skip=0, step=1):
        #print(args, window_size)
        return TimeSeriesIterator(args,window_size=window_size, skip=skip, step=step)


    @staticmethod
    def zeros(base, name=None, unit="", value=0):
        if name is None:
            ts = TimeSeries(name="Zeros " + base.name, unit=unit)
        else:
            ts = TimeSeries(name=name, unit=unit)
        array = np.zeros(len(base))
        array[:] = value
        ts._set_series(values=array, timestamps=base.timestamps)
        return ts

    @staticmethod
    def from_pattern(pattern_array, timestamp_begin=0, timestamp_end=1000, time_delta=1, timestamp_per_pattern=4, name="", unit=""):
        timestamps = np.linspace(timestamp_begin, timestamp_end, int((timestamp_end - timestamp_begin) / time_delta))
        #print((timestamp_end - timestamp_begin) / time_delta , timestamps.shape)
        array = np.zeros(len(timestamps))

        for i, t in enumerate(timestamps):
            if pattern_array.__class__.__name__ == TimeSeries.__name__:
                array[i] = pattern_array.data[int((i/timestamp_per_pattern) % len(pattern_array))]
            else:
                array[i] = pattern_array[int((i/timestamp_per_pattern) % len(pattern_array))]

        return TimeSeries(values=array, timestamps=timestamps, name=name, unit=unit)

    @staticmethod
    def from_numpy(array, timestamps=None, name="", unit=""):
        ts = TimeSeries(name=name, unit=unit)
        if timestamps is None:
            ts._set_series(array, np.arange(0, array.shape[0], 1))
        else:
            ts._set_series(array, timestamps)

        return ts

    def to_standard_storage(self):
        ss = StandardStorage()
        ss.headers = ['timestamps', self.name]
        ss.init_zeros(len(self))
        ss.data[:, 0] = self.timestamps
        ss.data[:, 1] = self.data
        return ss

    @staticmethod
    def from_standard_storage(ss, name, timestamps_column="timestamp", unit="", check_dataset=True, timeseries_name=None):
        ts_column = ss.get_column(timestamps_column)
        ts_column = np.array([float(ts) for ts in ts_column])
        timestamp_begin = ts_column[0]
        timestamp_end = ts_column[-1]
        time_delta = ts_column[1] - ts_column[0]
        timestamps = np.linspace(timestamp_begin, timestamp_end, int((timestamp_end - timestamp_begin) / time_delta))
        ts = TimeSeries(name = name if timeseries_name is None else timeseries_name, unit=unit)

        if len(timestamps) != len(ts_column) and check_dataset:
            print("===== Dataset is BROKEN! Repairing it! =====")
            array = np.zeros(len(timestamps))
            dataset = ss.get_column(name)
            #iterate over dataset. Coping previous data in case of missing values. Removing reverses
            index = 0
            for i, t in enumerate(timestamps):
                # go over dataset
                if index != 0:
                    while not (ts_column[index - 1] <= t <= ts_column[index]) and index < len(dataset) -1:
                        #print(i, ts_column[index - 1] , t , ts_column[index], ts_column[index - 1] <= t,t <= ts_column[index])
                        index += 1
                array[i] = dataset[index]

                # the first element in ts_column and timestamps are equal due to np.arange. So only rewrite it
                if index == 0:
                    index += 1
            ts._set_series(array, timestamps)

        else:
            ts._set_series(ss.get_column(name), ss.get_column(timestamps_column))
        return ts

    @staticmethod
    def from_timeseries(ts, values=None, timestamps=None, name=None, unit=None):
        values = values if values is not None else ts.data
        timestamps = timestamps if timestamps is not None else ts.timestamps
        name = name if name is not None else ts.name
        unit = unit if unit is not None else ts.unit
        return TimeSeries(values,timestamps,name,unit)

    @staticmethod
    def from_timestamps_str(begin_str, end_str, timedelta, value=0, name="", unit="", include_last_element=True):
        """
        provide begin and end str in the following format: %Y-%m-%d %H:%M:%S
        """
        begin, end = TimeSeries.timestamp_from_datetime([begin_str, end_str], "time")
        e =  end+int(include_last_element*timedelta/100)
        data = np.array([[ts, value] for ts in range(begin,e, timedelta)]) #include last element if is equal to end
        return TimeSeries(data[:, 1],data[:, 0],name,unit)

    @staticmethod
    def split_train_test_val(dataset, train, test=None):
        """
        Dataset - a tuple produced by model_test_iteration_dataset
        train - float in range <0;1>. The train dataset size
        test - float in range <0;1> The test dataset size.
        If test == None then test size is 1-train
        """

        if len(dataset) < 2:
            raise RuntimeError("TimeSeries.split_train_test_val: dataset has to be tuple (x,y)")



        l = len(dataset[0])
        train_int = int(train * l)

        train_data = tuple([d[:train_int] for d in dataset])

        if test is None:
            test_data = tuple([d[train_int:] for d in dataset])
        else:
            test_int = int(test * l)
            test_data = tuple([d[train_int:test_int] for d in dataset])

        if batch_count is not None:
            train_batch_len = int(len(train_data[0])/batch_count)
            #print("batch_len", train_batch_len, [train_data[1][i*train_batch_len:(i+1)*train_batch_len].shape for i in range(batch_count)])
            train_data = tuple(
                [np.vstack([td[i*train_batch_len:(i+1)*train_batch_len][np.newaxis, ...] for i in range(batch_count)]) for td in train_data]
            )
            test_batch_len = int(len(test_data[0])/batch_count)
            #print("batch_len", test_batch_len, [test_data[1][i*test_batch_len:(i+1)*test_batch_len].shape for i in range(batch_count)])

            test_data = tuple(
                [np.vstack([td[i*test_batch_len:(i+1)*test_batch_len][np.newaxis, ...] for i in range(batch_count)]) for td in test_data]
            )
            #test_data = tuple([np.expand_dims(t, axis=0) for t in test_data])

        return train_data, test_data

    def __init__(self, values=None, timestamps=None, name="", unit="", is_x_time=True):
        self._parameter_name = ""
        # data is set of samples of the time series
        self._data = None
        # timestampes corresponding to the data samples
        self._timestamps = None
        # name defines the signal name
        self._name = name
        # unit variable is used in grouping multiple timeseries on a graph
        self._unit = unit
        # variable holds information if class is a real TimeSeries or only set of points where
        # timestamps is x and data is y. It matters while ploting only
        self._is_x_time = is_x_time

        self._is_singular = True

        # init object if all parameters are given
        if timestamps is not None:
            self._set_series(values, timestamps)

    def clone(self):
        """
        Create an exact copy ot the object.
        """
        return TimeSeries(self.data.copy(), self.timestamps.copy(), self.name, self.unit, self._is_x_time)

    '''
        Checks if object is configurated (inited) properly
    '''
    def check_configuration(self):
        ok = True
        if self._data is None: ok = False
        if self._timestamps is None: ok = False

        return ok



    def interpolate(self):
        """
        Function interpolate timeseries
        """
        start = 0
        last_value = self.data[0]

        array = np.zeros(len(self.timestamps))

        for i, t in enumerate(self.timestamps):
            if self.data[i] != last_value or i == len(self.timestamps) - 1:
                count = i - start
                diff = self.data[i] - last_value
                for j in range(start,i):
                    array[j] = diff * (j-start)/count + last_value

                start = i
                last_value = self.data[i]


        return TimeSeries(values=array, timestamps=self.timestamps, name=self.name, unit=self.unit)

    '''
        Change sampling rate for whole timeseries
    '''
    def resampling(self, new_delta=None):
        print("Resampling of TimeSeries: ", self.name)
        if new_delta is not None:
            # print(self.begin, self.end, int((self.end - self.begin) / new_delta))
            timestamps = np.linspace(int(self.begin), int(self.end), int((self.end - self.begin) / new_delta))
        else:
            raise ValueError("You have to specify new_delta parameter!")
        #print(self.begin, self.end, int((self.end - self.begin) / new_delta))
        array = np.zeros(len(timestamps))
        index = 0


        # rising frequency
        if len(timestamps) > len(self):
            for i, t in enumerate(timestamps):
                # go over dataset
                if index != 0:
                    while not (self.timestamps[index - 1] <= t <= self.timestamps[index]) and index < len(self) - 1:
                        # print(i, ts_column[index - 1] , t , ts_column[index], ts_column[index - 1] <= t,t <= ts_column[index])
                        index += 1

                array[i] = self.data[index]

                # the first element in ts_column and timestamps are equal due to np.arange. So only rewrite it
                if index == 0:
                    index += 1
        else:
            # lowering frequency
            print("TimeSeries.resampling: lowering")

            for i,t in enumerate(self.timestamps):
                s = 0
                c = 0
                #print("TimeSeries.resampling: lowering", s,c, t)
                if index != 0:
                    #print("TimeSeries.resampling: lowering", timestamps[index - 1] , t , timestamps[index])
                    while (timestamps[index - 1] <= t <= timestamps[index]) and index < len(timestamps) - 1:
                        s += self.data[i]
                        c += 1
                        index += 1

                    while t > timestamps[index] and index < len(timestamps) - 1:
                        index += 1
                        s += 0
                        c += 0

                if c > 0:
                    array[index] = s / c

                if index == 0:
                    index += 1

        return TimeSeries(values=array, timestamps=timestamps, name=self.name, unit=self.unit)

    '''
        Properties section
    '''
    def timedelta(self, unit="s"):
        available_units = {
            "h": 3600,
            "min": 60,
            "s": 1,
        }
        # check the parameter
        if not unit in available_units:
            unit = "s"

        # function simplified. In production it should check all deltas and raise an exception if in any case it is not
        # equal
        if len(self._timestamps) >= 2:
            # return
            return int((self._timestamps[1] - self._timestamps[0]) / available_units[unit])
        else:
            return 0

    def time_difference(self, other):
        return other.timestamps[0] - self._timestamps[0]





    # function returns first sample timestamp (beginning of timeseries)
    @property
    def begin(self):
        """
        :return: The first timestamp
        """
        return self._timestamps[0]

    # function returns last sample timestamp (ending of timeseries)
    @property
    def end(self):
        """
        :return: The last timestamp
        """
        return self._timestamps[len(self._timestamps) - 1]

    @property
    def head(self):
        return self._data[0]

    @property
    def tail(self):
        return self._data[-1]

    @property
    def name(self):
        return self._name

    @property
    def max(self):
        max = None
        for i,_ in enumerate(self.data):
            if self.data[i] is not None:
               if max is None or max < self.data[i]:
                    max = self.data[i]

        return max

    @property
    def min(self):

        min = None
        for i,_ in enumerate(self.data):
            if self.data[i] is not None:
               if min is None or min > self.data[i]:
                    min = self.data[i]
        return min

    @property
    def mean(self):
        return np.mean(self._data)
    @property
    def sum(self):
        return np.sum(self._data)

    @property
    def weighted_mean(self):
        # if self._is_x_time:
        #     raise RuntimeError("TimeSeries.weighted_mean available only for histograms or other forms of TimeSeries, "
        #                        "which x-axis is not a time. Ensure if you used correct preprocessing.")
        # values are stored in timestamps and weights are in data (like for histograms)
        # if len(self) == 0:
        #     return 0
        return np.sum(self._data * self._timestamps) / np.sum(self._data)

    @property
    def extreme_maximum(self):
        """
        Function returns all local maximum extremes (coordinates [x, y])
        """
        extremes = self.extreme_maximum_indexes
        extremes[:, 0] = [self.timestamps[i] for i in extremes[:, 0].astype(int)]
        return extremes

    @property
    def extreme_maximum_indexes(self):
        extremes = []
        if len(self) > 1:
            first_derivative = self.diff(1).data

            # check for inner extremes. extreme is when first derivative change sign.
            for i, (d1, d2) in enumerate(zip(first_derivative.data[:-1], first_derivative.data[1:])):
                if d1 >= 0 and d2 < 0:
                    extremes.append([i, self.data[i]])

            # check for the last element only if ts contains more than 1 sample. If one then first condition checks
            # for extreme
            if len(self) > 1 and first_derivative[-1] > 0:
                extremes.append([len(self) - 1, self.data[-1]])

        return np.array(extremes)
    @property
    def extreme_minimum(self):
        extremes = self.extreme_minimum_indexes
        extremes[:, 0] = [self.timestamps[i] for i in extremes[:, 0].astype(int)]
        return extremes

    @property
    def extreme_minimum_indexes(self):
        extremes = []
        if len(self) > 1:
            first_derivative = self.diff(1).data

            # check for inner extremes. extreme is when first derivative change sign.
            for i, (d1, d2) in enumerate(zip(first_derivative.data[:-1], first_derivative.data[1:])):
                if d1 <= 0 and d2 > 0:
                    extremes.append([i, self.data[i]])

            # check for the last element only if ts contains more than 1 sample. If one then first condition checks
            # for extreme
            if len(self) > 1 and first_derivative[-1] < 0:
                extremes.append([len(self)-1, self.data[-1]])

        return np.array(extremes)

    @property
    def is_singular(self):
        """
        :return:
        """
        return self._is_singular

    '''
        Function recalculates timestamps into indexes. It makes easier adjusting different timeseries to each other. 
    '''
    @property
    def time_indexes(self):
        #print(self._timestamps[0], self.timedelta())
        return np.array([int(ts) for ts in (self._timestamps / self.timedelta())])

    @name.setter
    def name(self, value):
        self._name = value

    def set_self_name(self, value):
        self.name = value
        return self

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def timestamps(self):
        return self._timestamps

    @timestamps.setter
    def timestamps(self, value):
        self._timestamps = value

    '''
        Private functions section
    '''
    def _set_series(self, values, timestamps):
        if values is not None:
            if isinstance(values, numbers.Number):
                self.data = np.ones(len(timestamps)) * values
            else:
                self.data = values
        else:
            self.data = np.zeros(len(timestamps))
        self.timestamps = timestamps

        if len(self.data) != len(self.timestamps):
            raise RuntimeError(f"Function TimeSeries._set_series: values and timestamps len should be the same size. Given values" \
                  f"len(self.data) == {len(self.data)} and len(self.timestamps) == {len(self.timestamps)}.")

    @property
    def range(self):
        return self.min, self.max

    @property
    def is_x_time(self):
        return self._is_x_time

    def time_range(self):
        return self._timestamps[0], self._timestamps[len(self._timestamps) - 1]

    def most_common_value(self):
        if self._is_x_time:
            raise "Function TimeSeries.most_common_value is disable for TS which has is_x_time == True. " \
                  "Apply TimeSeries.histogram by TimeSeries.perform_function first to transform TS. "
        return np.argmax(self._data) * max(self._timestamps) / len(self._timestamps)

    '''
        Function returns sub series of the whole series
    '''
    def sub_series(self, begin=0, end=-1, step=1):
        # if begin is a timeseries object then return subseries adjusted to timestamps from begin
        if self.__class__.__name__ == begin.__class__.__name__:
            if int(begin.timedelta()) != int(self.timedelta()):
                raise ValueError("Timeseries have different timedeltas. Before sub_series you have to equalize timedeltas with resampling function!")
            dst_timestamps_indexes = begin.time_indexes
            src_timestamps_indexes = self.time_indexes
            dst_l = dst_timestamps_indexes[0]
            dst_r = dst_timestamps_indexes[-1]
            src_l = src_timestamps_indexes[0]
            src_r = src_timestamps_indexes[-1]
            # print((dst_l,dst_r,src_l, src_r) - min(dst_l,dst_r,src_l, src_r))
            # print([dst_l, dst_r])
            # print([src_l, src_r])

            #create new array
            array = np.zeros(len(dst_timestamps_indexes))

            '''
                1. case:   ####dst####                dst_r < str_l dont check it - nothing happens
                                          ####src####
                -------------------------------------
                2. case:   ####dst####                dst_l < src_l < dst_r < src_r
                                ####src####
                -------------------------------------
                3. case:        ####dst####           dst_r > src_r > dst_l > src_l
                           ####src####
                -------------------------------------
                4. case:                  ####dst#### dst_l > src_r dont check it - nothing happens
                           ####src####
                -------------------------------------
                5. case:       #dst#                  src_l <= dst_l < dst_r <= src_r
                           ####src####
                -------------------------------------
                6. case:   ####dst####                dst_l <= src_l < src_r <= dst_r
                            #src#
                -------------------------------------
                in overlapped area function assigns source (self obj.) data to returned object. Outside overlapped area
                function assigns 0 
            '''
            # check overlapping area
            # case 1
            if dst_l < src_l < dst_r < src_r:
                #print("case 2")
                ov_l, ov_r = np.where(dst_timestamps_indexes == src_l)[0][0], np.where(src_timestamps_indexes ==dst_r)[0][0]+1
                array[ov_l:] = self.data[:ov_r]
            if dst_r > src_r > dst_l > src_l:
                #print("case 3")
                ov_l, ov_r = np.where(src_timestamps_indexes == dst_l)[0][0], np.where(dst_timestamps_indexes == src_r)[0][0]+1
                array[:ov_r] = self.data[ov_l:]
            if src_l <= dst_l < dst_r <= src_r:
                #print("case 5")
                ov_l, ov_r = np.where(src_timestamps_indexes == dst_l)[0][0], np.where(src_timestamps_indexes == dst_r)[0][0]+1
                array[:] = self.data[ov_l:ov_r]
            if dst_l <= src_l < src_r <= dst_r:
                #print("case 6")
                ov_l, ov_r = np.where(dst_timestamps_indexes == src_l)[0][0],  np.where(dst_timestamps_indexes == src_r)[0][0]+1
                array[ov_l:ov_r] = self.data[:]

            return TimeSeries(values=array, timestamps=begin.timestamps, name=self.name, unit=self.unit)
        else:
            return TimeSeries(values=self.data[begin:end:step], timestamps=self.timestamps[begin:end:step], name=self.name, unit=self.unit)

    def plot(self, additional_timeseries=[], conv_func=None, plot_type=None, upper_postfix="_top",
             lower_postfix="_bottom", callable_or_function=None, item_per_page=None):
        limits = None
        if self.unit == "%":
            limits = [100, 0]

        arraySlider = ArraySlider(use_global_y_limits = False, one_y_axis=False, debug=True, conv_func=conv_func,
                                  plot_type=plot_type, x_as_time=self._is_x_time, y_range=limits, callable_or_function=callable_or_function, item_per_page=item_per_page)
        additional_timeseries.insert(0, self)

        units = []
        groups = []
        for i, at in enumerate(additional_timeseries):
            if at.unit in units:
                groups.append(units.index(at.unit))
            else:
                units.append(at.unit)
                groups.append(len(units)-1)

        for i, at in enumerate(additional_timeseries):
            label = at.name

            # timeseries is an upper limit ex. coefcient interval of signal
            # ex. signal "M_0" has limits f"M_0{upper_postfix}" and f"M_0{lower_postfix}
            # then those signals want be added separately
            if label.endswith(upper_postfix) or label.endswith(lower_postfix):
                continue

            u = None
            l = None
            # look for coeficient interval
            for ii, at2 in enumerate(additional_timeseries):
                #print(at2.name, f"{label}{upper_postfix}", f"{label}{lower_postfix}")
                if  at2.name == f"{label}{upper_postfix}":
                    u = at2.data
                elif at2.name == f"{label}{lower_postfix}":
                    l = at2.data

            if at.unit != "":
                label += " [" + at.unit + "]"

            arraySlider.plot(self.timestamps, at.data, label=label, signal_group=groups[i], upper=u, lower=l)
        arraySlider.before_show()
        plt.show()

    def window_iteration(self, window_size, verbose=0):
        return TimeSeriesIterator(self, window_size, verbose)

    def model_test_iteration(self, window_size, predict_horizon, verbose=0, skip=0, step=1, predict_window=None, batch_size=None):
        return TimeSeriesModelTestIterator(self, window_size, predict_horizon, verbose, skip, step, predict_window, batch_size)

    def model_test_iteration_dataset(self, window_size, predict_horizon, verbose=0, skip=0, step=1, batch_size=-1, batch_count=-1):
        data = [(window, test_window[-1:]) for window, k, test_window in self.model_test_iteration(window_size=window_size,
                                                                           predict_horizon=predict_horizon,
                                                                           verbose=verbose,
                                                                           skip=window_size+skip,
                                                                           step=step)]
        X = [x[0] for x in data]
        Y = [x[1] for x in data]


        if batch_count == -1:
            batch_count = int(len(data) / batch_size) if batch_size > 0 else 1
        elif batch_count > 0:
            batch_size =  int(len(data) / batch_count)

        X = [X[i * batch_size:(i + 1) * batch_size] for i in range(batch_count)]
        Y = [Y[i * batch_size:(i + 1) * batch_size] for i in range(batch_count)]

        return TimeSeriesDatasetXY((TimeSeriesDataset(X),TimeSeriesDataset(Y)))

    def at(self, arg):
        return self.data[arg]

    def perform_window_function(self, function, window_size, **kwargs):
        """
            Function creates new TS by computing new value using given function "function" on a window. window_size
            defines size of the window. kwargs are passed directly as additional arguments to the function

            function: a function(window, **kwargs) which takes window as TimeSeries and kwargs if needed, and returns
                      a number value. This value will be assign to new TimeSeries data

            return: new TS object created as described above
        """
        array = np.zeros(len(self))
        array[window_size:] = np.array([function(window, **kwargs) for window, i in self.window_iteration(window_size)])

        # array = [0] * window_size + array  # extend at the beginning to equalize lengths
        return TimeSeries(values=array, timestamps=self.timestamps, name=self.name+".WF({0})".format(window_size), unit=self.unit)

    """
        Function limits values in dataset.
    """
    def limit_values(self, max_value = None, min_value = None, method="last"):
        data = np.zeros(len(self.data))
        for i,d in enumerate(self.data):
            if max_value is not None and d > max_value:
                data[i] = self.data[i-1]
            elif min_value is not None and d < min_value:
                data[i] = self.data[i-1]
            else:
                data[i] = self.data[i]

        return TimeSeries(values=data, timestamps=self.timestamps, name=self.name, unit=self.unit)

    def histogram(self, bins=10, max=None, min=None, normalize_0_100 = False, normalize_0_1 = False, fixed_names=False):
        """
        Histogram function computes histogram of values. It divides values into bins. The len of bin is constant and
        is calculated according to equation (max - min) / bins. Function returns percentage [%] participation
        (normalize_0_100 == True), 0..1 participation (normalize_0_1 == True) or number of appearances.
        max and min cannot be None!! adaptive function gives incorrect answers.
        """
        r = (min, max)
        step = (r[1] - r[0]) / bins
        bins_list = list(np.arange(r[0], r[1], step))
        #print("asd", r[0], r[1], step)
        if len(bins_list) <= bins:
            bins_list = bins_list + [r[1]]
        #print("TimeSeries.histogram", r[0], r[1], step, "bins:", bins, len(bins_list))
        his, ts = np.histogram(self.data, bins=bins_list)
        if normalize_0_100:
            his = 100* his / np.sum(his) # normalization
        elif normalize_0_1:
            his = his / np.sum(his) # normalization
        # ts = [ts[0:i].sum() for i,_ in enumerate(ts)]
        if fixed_names:
            name = self.name
        else:
            name = self.name + ".HIS({0})".format(bins)
        return TimeSeries(values=his, timestamps=bins_list[:-1], name=name,
                          unit="%", is_x_time=False)

    def kernel_density_estimation(self, bandwidth=0.2, estimated_x=None, fixed_names=True):
        """
        Compute kernel density estimation for histogram contained by timeseries object
        """
        if self._is_x_time:
            raise "Function TimeSeries.kernel_density_estimation is disable for TS which has is_x_time == True. " \
                  "Apply TimeSeries.histogram by TimeSeries.perform_function first to transform TS. "

        X = []
        for i, time in enumerate(self.timestamps):
            if np.isnan(self.data[i]):
                X.append(time)
            else:
                for _ in range(floor(self.data[i] * 100 / np.nansum(self.data))):
                    X.append(time)

        X = np.array(X).reshape(-1, 1)
        # print(X)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
        if estimated_x is None:
            estimated_x = np.linspace(0, max(self.timestamps), len(self.data))

        if fixed_names:
            name = self.name
        else:
            name = self.name + ".KDE({0})".format(bandwidth)
        return TimeSeries(values=np.exp(kde.score_samples(estimated_x.reshape(-1, 1))), timestamps=estimated_x,
                          name=name, unit="", is_x_time=False)

    def integrate(self, unit="h"):
        """
        Function computes integral from a timeseries, but without respect to the X axis
        :param unit: The time definition for self.timedelta function {h,m,s}
        :return: Integrated TimeSeries data
        """

        return np.sum(self.data) # * self.timedelta(unit=unit)

    '''
        Predefined timeseries operations 
    '''
    def roll(self, shift=0, overwrite_elements=None):
        """
        Function rolls elements in data property. Elements form index i is moved to i+shift. Elements beyond the
        array range are move to the second side. if overwrite_elements is a number then beyond elements are being
        replaced with overwrite_elements
        """
        ts = TimeSeries()
        ts.name = self.name + ".ROLL({0})".format(shift)
        ts.unit = self.unit

        array = np.roll(self.data, shift)
        if overwrite_elements is not None:
            if shift > 0:
                array[:shift] = overwrite_elements
            else:
                array[shift:] = overwrite_elements
        ts._set_series(array, self.timestamps.copy())
        return ts

    def moving_avg(self, window_size, roll=False):
        """
        Function calculates moving average (walking average) on a given samples
        :param window_size: Number of samples taken into average calculation
        :param roll if roll is True then result is pushback and adjusted to the center
        :return: Moving average TimeSeries
        """
        ts = TimeSeries()
        ts.name = self.name + ".MA({0})".format(window_size)
        ts.unit = self.unit
        array = np.zeros(self._data.shape[0])
        array[0:window_size] = self._data[0]
        for i in range(window_size, self._data.shape[0]):
            array[i] = np.mean(self._data[i-window_size: i])

        if roll:
            array = np.roll(array, int(-window_size/2)) # roll elements if flag is set True

        ts._set_series(array, self.timestamps.copy())

        return ts

    def neighborhood_max(self, window_size, roll=False, moving_avg=False):
        """
        Function transform TS into TS that contains in i-th cell the max value obtained for each window
        """
        ts = TimeSeries()
        ts.name = self.name + ".NMX({0})".format(window_size)
        ts.unit = self.unit
        array = np.zeros(self._data.shape[0])
        array[0:window_size] = self._data[0]
        for i in range(window_size, self._data.shape[0]):
            array[i] = np.max(self._data[i-window_size: i])

        if moving_avg:
            if roll:
                array = np.roll(array, int(-window_size / 2))  # roll elements if flag is set True
            array[window_size:] = np.array([np.mean(array[i - window_size: i]) for i in range(window_size, array.shape[0])])

        if roll:
            array = np.roll(array, int(-window_size/2)) # roll elements if flag is set True

        ts._set_series(array, self.timestamps.copy())
        return ts

    def neighborhood_min(self, window_size, roll=False, moving_avg=False):
        """
        Function transform TS into TS that contains in i-th cell the min value obtained for each window
        """
        ts = TimeSeries()
        ts.name = self.name + ".NMI({0})".format(window_size)
        ts.unit = self.unit
        array = np.zeros(self._data.shape[0])
        array[0:window_size] = self._data[0]
        for i in range(window_size, self._data.shape[0]):
            array[i] = np.min(self._data[i-window_size: i])

        if moving_avg:
            if roll:
                array = np.roll(array, int(-window_size / 2))  # roll elements if flag is set True
            array[window_size:] = np.array([np.mean(array[i - window_size: i]) for i in range(window_size, array.shape[0])])

        if roll:
            array = np.roll(array, int(-window_size/2)) # roll elements if flag is set True

        ts._set_series(array, self.timestamps.copy())
        return ts

    def diff(self, order=1, data=None):
        """
        Function computes n-th derivative
        :param order: derivative order
        :param data: parameter used for recursive execution
        :return: n-th derivative TimeSeries
        """
        if data is None:
            data = self.data

        if order > 1:
            data = self.diff(order=order-1, data=data).data

        # array = np.array([data[0]] + [data[i] - data[i-1] for i in range(1, data.shape[0])])
        array = np.array([0] + [data[i] - data[i-1] for i in range(1, data.shape[0])])
        return TimeSeries(values=array, timestamps=self.timestamps, name=self.name+"'", unit=self.unit + "'")

    # def dtw(self, other):
    #     if len(self) != len(other):
    #         raise RuntimeError("Cannot perform dynamic-time-wrapping to different length.")
    #     return dtw.distance_fast(self.data, other.data, use_pruning=True)

    def abs(self):
        return TimeSeries(timestamps=self.timestamps, values=np.abs(self.data), name=f"ABS({self.name})",
                          unit=self.unit)
    def sqrt(self):
        return TimeSeries(timestamps=self.timestamps, values=np.sqrt(self.data), name=f"ABS({self.name})",
                          unit=self.unit)

    def nan_to_num(self, num):
        self.data[np.isnan(self.data)] = num
        return TimeSeries(timestamps=self.timestamps, values=self.data, name=f"{self.name}",
                          unit=self.unit)

    def __len__(self):
        """
        Length operator - number of samples along a time axis
        :return:
        """
        return len(self.data)

    def __eq__(self, other):
        """
        Equals operator returns boolean array. Useful for filtering
        :return:
        """
        if isinstance(other, (list, tuple, np.ndarray)):
            return self.data == other
        elif self.__class__.__name__ == other.__class__.__name__:
            # instance of timeseries
            return self.data == other.data
        else:
            # simple number
            return self.data == other

    def __iter__(self):
        return TimeSeriesIterator(self)

    def get_by_timestamp(self, timestamp):
        if isinstance(arg, slice):
            pass

    def get_by_boolean_array(self, boolean_array):
        return TimeSeries(timestamps=self.timestamps[boolean_array], values=self.data[boolean_array], name=self.name + "BA",
                          unit=self.unit)

    def __getitem__(self, arg):
        name_addon = ""
        if isinstance(arg, slice):
            al = [
                str(arg.start) if arg.start is not None else "",
                str(arg.stop) if arg.stop is not None else "",
                str(arg.step) if arg.step is not None else "",
            ]
            name_addon = "[" + ":".join(al) + "]"
        if isinstance(arg, Number):
            return TimeSeries(timestamps=self.timestamps[arg:arg+1], values=self.data[arg:arg+1], name=self.name + "[{0}]".format(arg), unit=self.unit)
        if isinstance(arg, (list, tuple)):
            # list of booleans for filtering
            return TimeSeries(timestamps=self.timestamps[arg], values=self.data[arg], name=self.name, unit=self.unit)

        return TimeSeries(timestamps=self.timestamps[arg], values=self.data[arg], name=self.name + name_addon, unit=self.unit)

    def __setitem__(self, key, value):

        # due to relative imports it is imposible to check classes with isinstance methods
        # for relative import class is: environment.helpers.TimeSeries.TimeSeries
        # while for other import class is: TimeSeries.TimeSeries
        # so it won't work
        # checking __class__.__name__ allows to workaround this problem
        # print(self.__class__.__name__ == value.__class__.__name__, self.__class__.__name__, value.__class__.__name__)

        if self.__class__.__name__ == value.__class__.__name__:
            if isinstance(key, slice):
                # user has to pass TimeSeries object with the same length as key slice specifies
                # if a range has been assigned
                self.data[key] = value.data
            else:
                # if single value has been assigned to the object
                self.data[key] = value.data[0]

        else:
            self.data[key] = value

    def __str__(self):
        txt = "--- Time Series --- \n"

        if self.check_configuration() == False:
            txt += " - not inited"
            return txt
        txt += " Name: {0}\n".format(self.name)
        txt += " Unit: {0}\n".format(self.unit)
        txt += " Size: {0}\n".format(len(self.data))
        txt += " Max: {0}\n".format(self.max)
        txt += " Min: {0}\n".format(self.min)
        if len(self.timestamps) > 1:
            txt += " Time range: {0} - {1}\n".format(self.timestamps[0], self.timestamps[-1])
        elif len(self.timestamps) == 1: txt += "Time range: {0} - {0}\n".format(self.timestamps[0])
        txt += " Data: \n" + str(self.data)
        return txt

    '''
        Operators
    '''
    def __add__(self, other):
        if len(self) != len(other):
            raise ValueError("Given timeseries has different length {0}, {1}".format(len(self), len(other)))
        array = np.array([self._data[i] + other.data[i] for i in range(len(self))])
        return TimeSeries(values=array, timestamps=self.timestamps, name=self._name + " + " + other.name)

    def __sub__(self, other):
        if len(self) != len(other):
            raise ValueError("Given timeseries has different length {0}, {1}".format(len(self), len(other)))
        array = np.array([self._data[i] - other.data[i] for i in range(len(self))])
        return TimeSeries(values=array, timestamps=self.timestamps, name=self._name + " - " + other.name, unit=self.unit)

    def __div__(self, other):
        if len(self) != len(other):
            raise ValueError("Given timeseries has different length {0}, {1}".format(len(self), len(other)))
        array = np.array([self._data[i] / other.data[i] for i in range(len(self))])
        return TimeSeries(values=array, timestamps=self.timestamps, name=self._name + " / " + other.name, unit=self.unit)

    def __mul__(self, other):

        if isinstance(other, np.ndarray):
            return TimeSeries(values=self.data * other, timestamps=self.timestamps, name=self._name, unit=self.unit)

        elif other.__class__.__name__ == TimeSeries.__name__:
            return TimeSeries(values=self.data * other.data, timestamps=self.timestamps, name=self._name + " * " + other.name, unit=self.unit)

        # if isinstance(other, Number):
        else:
            return TimeSeries(values=self.data * other, timestamps=self.timestamps, name=self._name + " * " + str(other), unit=self.unit)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return TimeSeries(values=self.data / other, timestamps=self.timestamps,
                              name=self._name + " / " + str(other), unit=self.unit)
        elif self.__class__.__name__ == other.__class__.__name__:
            if len(self) != len(other):
                raise ValueError("Given timeseries has different length {0}, {1}".format(len(self), len(other)))
            array = []

            for i in range(len(self)):
                if other.data[i] == 0 or np.isnan(other.data[i]):
                    array.append(0)
                else:
                    array.append(self._data[i] / other.data[i])


            return TimeSeries(values=np.array(array), timestamps=self.timestamps, name=self._name + " / " + other.name,
                              unit=self.unit)
        else:
            raise ValueError("Only TimeSeries number or TimeSeries division is allowed!")

    def reciprocal(self):
        return TimeSeries(values=np.array([1/d if d != 0 else 0 for d in self.data]), timestamps=self.timestamps, name=self._name + "^(-)",
                              unit=self.unit)