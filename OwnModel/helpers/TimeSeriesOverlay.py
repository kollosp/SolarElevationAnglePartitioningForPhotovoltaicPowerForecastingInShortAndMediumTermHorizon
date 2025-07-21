import numpy as np
#import ultraimport
import math
import matplotlib.pyplot as plt
from .MTimeSeries import MTimeSeries
from .TimeSeries import TimeSeries
# ultraimport('__dir__/../helpers/MTimeSeries.py', 'MTimeSeries', globals=globals())
# ultraimport('__dir__/../helpers/TimeSeries.py', 'TimeSeries', globals=globals())
from matplotlib.axis import Axis
from sklearn.neighbors import KernelDensity

class NormalizedMapFormatter:
    def __init__(self, bins, reversed=False):
        self._bins = bins
        self._reversed = reversed

    def __call__(self, x, pos=None):
        el = x / self._bins
        if self._reversed:
            el = 1 - el
        return f"{el:.2f}"

class TimeSeriesOverlay:
    """
    Class which represents a set of overlapped timeseries. Each timeseries is a cut part of origin timeseries.
    Each timeseries is same length. The first and the last are removed to ensure same sizes. The original object is
    immutable during TimeSeriesOverlay object
    """
    def __init__(self,
                 timeseries=None,
                 group_assignment=None,
                 bin_assignment=None,
                 y_bins=None,
                 x_bins=None,
                 filter=None,
                 filter_value=None,
                 mts=None,
                 bandwidth=0.4):
        """
        timeseries - base timeseries
        group_assigment -  a list-like object which defines the relation between timeseries[i] with a group it should be
                        assigned to. len(bin_assigment) has to be equal len(timeseries)
        bin_assigment - a list-like object which defines the relation between timeseries[i] with an element inside the
                        group it should be assigned to. len(bin_assigment) has to be equal len(timeseries)
        filter - a similar list-like object or none. if passed then function checks filter[i] == filter_value and omit
                 timeseries[i] if condition not met.

        Data is represented as a 2D array MTimeSeries. Original ts is transformed into these 2D array A. Values in
        group_assigment and bin_assigment correspond with coordinates in A. group_assigment is row in A and bin_assigment
        is column in A. By "is" it should be understand as corresponds. They are not indexes.
        """
        if mts is not None: #constructor used only in TimeSeriesOverlay functions
            self._mts = mts
        elif timeseries is not None and bin_assignment is not None: # default constructor
            self._mts = TimeSeriesOverlay.build_up(timeseries, group_assignment, bin_assignment, filter, filter_value)
        else:
            raise ValueError("Either 'timeseries' and 'bin_assigment' 'parameters' or mts have been passed in arguments")
        self._bandwidth = bandwidth
        self._heatmap, self._heatmap_x, self._global_max = TimeSeriesOverlay.build_up_cross_section(self._mts,
                                                                                                    y_bins=y_bins,
                                                                                                    x_bins=x_bins,
                                                                                                    bandwidth=self._bandwidth)
        self._x_bins = x_bins
        self._y_bins = y_bins
        self._global_max = 1.01 * self._global_max  # find max timeseries in mts and then max value in that timeseries
                                                        # 1.01 helps with border case when item == global_max and
                                                        # bin = int(item / global_max * len(self._heatmap)) => bin == len(self._heatmap)

        self._filter_thresholds = [] # filter thresholds for each histogram. Has to be inited using build_filters

    @property
    def _are_filters_set(self):
        return len(self._filter_thresholds) > 0

    @property
    def mts(self):
        return self._mts

    @staticmethod
    def build_up_cross_section(mts, y_bins=None, x_bins=None, bandwidth=0.4):
        if x_bins is None:
            x_bins = len(mts)
        if y_bins is None:
            y_bins = x_bins

        heatmap = np.zeros((y_bins,x_bins))
        data = mts.to_numpy()
        global_max = mts.max().max #find max timeseries in mts and then max value in that timeseries
        global_min = 0 #find min timeseries in mts and then min value in that timeseries

        heatmap_x = np.linspace(global_min, global_max, y_bins)
        ht_x = heatmap_x.reshape(-1, 1)
        for i in range(data.shape[1]):
            x = data[:, i]
            x = x[np.isfinite(x)]
            #print(estimated_x, x, global_min, global_max)
            if len(x) > 0:
                kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x.reshape(-1,1)) # [i] gives column as column.
                heatmap[:, i] = np.exp(kde.score_samples(ht_x)) # crosssection
                #print(f"{i}: trapz", np.trapz(heatmap[:, i].ravel(), ht_x.ravel())) #trapezoid
            else:
                heatmap[0, i] = 1 #set 100% that it
                heatmap[1:, i] = 0
            #print("heatmap", heatmap[:, i])

        # print("TimeSeriesOverlay.build_up_cross_section")
        # print(heatmap.shape)
        # print(heatmap.shape)

        return heatmap, heatmap_x, global_max

    @staticmethod
    def build_up(timeseries, group_assignment, bin_assignment, filter=None, filter_value=None):
        """Function make bin assignment. Also checks if filter value is passed. Otherwise, put None into return set"""
        if len(timeseries) != len(bin_assignment) or len(group_assignment) != len(bin_assignment):
            raise ValueError(
                f"bin_assign: Cannot assign if len(timeseries) != len(bin_assigment). Passed values are {len(timeseries)} and {len(bin_assigment)}")
        timeseries = timeseries.data

        unique_groups = np.unique(group_assignment) #define number of unique values
        unique_bins = np.unique(bin_assignment)#define number of unique values
        data = [[[] for _ in enumerate(unique_bins)] for _ in enumerate(unique_groups)] #create 2D rect-shape data array

        #fill data array
        for i, _ in enumerate(timeseries):
            ga = np.argwhere(unique_groups == group_assignment[i])[0][0] #always should be one indic with one index
            ba = np.argwhere(unique_bins == bin_assignment[i])[0][0] #always should be one indic with one index
            # print("build_up", ga, ba)
            if filter is None or filter[i] == filter_value: #if filter exists and is met or if filter is not defined
                data[ga][ba].append(timeseries[i]) #append value to mean array
            # else:
            #     data[ga][ba].append(np.nan)

        #compute means
        data = np.array([[np.nanmean(data[i][j]) if len(data[i][j]) > 0 else np.nan for j,_ in enumerate(unique_bins)] for i,_ in enumerate(unique_groups)])

        # rewrite data array into MTimeSeries
        plain = []
        for d in data:
            plain.append(TimeSeries(values=d, timestamps=list(range(len(unique_bins)))))

        # print([len(i) for i in plain])
        return MTimeSeries(plain)

    @staticmethod
    def build_proportional_filters(mts, heatmap, threshold=0.6):
        filters = []
        data = mts.to_numpy()

        for j in range(data.shape[1]):
            filter = max(heatmap[:, j]) * threshold # default filter - set constant fraction of max value
            filters.append(filter)

        return filters

    @staticmethod
    def build_quantitative_filters(mts, heatmap, threshold=0.6):
        threshold = 1-threshold # reverse logic. Higher threshold less data remains
        filters = []
        data = mts.to_numpy()

        for j in range(data.shape[1]):
            # compute threshold
            hp = heatmap[:, j].copy()
            ref = np.nansum(hp) # reference. Sum of histogram (about 100%). Goal: Set filter_threshold at the level which is above
                                  # threshold (given parameter) percentage of the chart
            th = np.nanmin(hp) # threshold is a minimum value available in histogram
            s = ref

            if threshold * ref > 0:
                while s > threshold * ref:
                    hp[hp <= th] = np.nan
                    th = np.nanmin(hp)
                    s = np.nansum(hp)

            filter = th # default filter - set constant fraction of max value
            filters.append(filter)

        return filters


    def set_filters(self, threshold=None, quantitative=None):
        if threshold is not None:
            self._filter_thresholds = TimeSeriesOverlay.build_proportional_filters(self._mts, self._heatmap, threshold)
        elif quantitative is not None:
            self._filter_thresholds = TimeSeriesOverlay.build_quantitative_filters(self._mts, self._heatmap, quantitative)
        else:
            raise ValueError("TimeSeriesOverlay.set_filters: Cannot set such a filter")
        return self

    def apply_highpass_filter(self):
        data = self._mts.to_numpy()
        ret = np.zeros(data.shape)
        ret.fill(np.nan)
        for i, row in enumerate(data):
            for j, item in enumerate(row):
                if not np.isnan(item):
                    bin = int(item / self._global_max * len(self._heatmap))
                    #if bin == len(self._heatmap)
                    if self._heatmap[bin, j] > self._filter_thresholds[j]:
                        ret[i,j] = data[i,j]

        mts = self._mts.clone()
        mts.from_numpy(ret)
        return TimeSeriesOverlay(mts=mts, x_bins=self._x_bins, y_bins=self._y_bins, bandwidth=self._bandwidth)

    def apply_zeros_filter(self):
        data = self._mts.to_numpy()
        #data[self._heatmap < probability_threshold] = np.nan

        # print("global", global_max)
        ret = data.copy()
        for i, row in enumerate(data):
            for j, item in enumerate(row):
                if item == 0:
                    bin = int(item / self._global_max * len(self._heatmap))
                    #if bin == len(self._heatmap)
                    if self._heatmap[bin, j] > self._filter_thresholds[j]:
                        ret[i,j] = np.nan

        mts = self._mts.clone()
        mts.from_numpy(ret)
        return TimeSeriesOverlay(mts=mts, x_bins=self._x_bins, y_bins=self._y_bins, bandwidth=self._bandwidth)

    def apply_lowpass_filter(self, threshold=None):
        pass

    def apply_combined_HP_ZF_HP(self, pre_hp_quantitative=0.1, zf_threshold=0.5,  hp_quantitative=0.4,
                                plot_before=False, title="apply_combined_HP_ZF_HP"):
        #shortcut for - which was find as best combined filter
        # plain = self.set_filters(quantitative=0.10).apply_highpass_filter()
        # plain = plain.set_filters(threshold=0.5).plot(title="PLAIN_A")
        # plain = plain.apply_zeros_filter().set_filters(quantitative=0.4).plot(title="HP+ZF")
        # plain = plain.apply_highpass_filter().plot(title="HP+ZF+HP")
        # return plain
        tso = self
        if pre_hp_quantitative > 0:
            tso = tso.set_filters(quantitative=pre_hp_quantitative).apply_highpass_filter()
        if zf_threshold > 0:
            tso = tso.set_filters(threshold=zf_threshold).apply_zeros_filter()
        if hp_quantitative > 0:
            tso = tso.set_filters(quantitative=hp_quantitative).apply_highpass_filter()
        if plot_before:
            tso.plot(title=title)

        return tso

    def plot_overlay(self, window_size=12, title="", highlighted_bins=[], ax=None, fig=None, formatter=None,
                     min_max_en=True, x_title="Bins"):
        _fig1, _ax1 = self._mts.plot_static(
            one_colors=['b', 'r'],
            alpha=0.1,
            mtss=[],
            ax=ax,
            fig=fig,
            x_title=x_title,
            vlines=[{"x": bin, "color": "darkorange"} for i, bin in enumerate(highlighted_bins)] if len(highlighted_bins) > 0 else []
        )


        if formatter is not None:
            Axis.set_major_formatter(_ax1.xaxis,formatter)

        if min_max_en:
            _ax1.plot(self._mts.timestamps, self._mts.positive_mean().moving_avg(window_size=window_size, roll=True).data,
                      color="navy")
            _ax1.plot(self._mts.timestamps, self._mts.max().moving_avg(window_size=window_size, roll=True).data, color="navy")
            _ax1.plot(self._mts.timestamps, self._mts.min().moving_avg(window_size=window_size, roll=True).data, color="navy")
        if fig is None:
            _fig1.show()

    def plot_heatmap(self, highlighted_bins=[], ax=None, formatter=None, x_title="Bins"):
        if ax is None:
            fig, _ax = plt.subplots()
        else:
            _ax = ax

        _ax.set_ylabel("Normalized power")
        _ax.set_xlabel(x_title)
        #_ax.set_yticks([i/10 for i in range(0,8,2)])
        img = _ax.imshow(np.flip(self._heatmap, axis=0), cmap='Blues')
        if formatter is not None:
            Axis.set_major_formatter(_ax.xaxis, formatter)
        Axis.set_major_formatter(_ax.yaxis, NormalizedMapFormatter(bins=self._heatmap.shape[0], reversed=True))

        if highlighted_bins is not None:
            for hb in highlighted_bins:
                _ax.axvline(x=hb, color="r")

        if ax is None:
            fig.colorbar(img)
            fig.show()
        return img

    def plot_heatmap_bin(self, highlighted_bins = None, ax=None, formatter=None, y_lim=None):
        if ax is None:
            fig, _ax = plt.subplots()
        else:
            _ax = ax
        for b in highlighted_bins:
            txt = str(b)
            if formatter is not None:
                txt = formatter(b)
            _ax.fill_between(self._heatmap_x, self._heatmap[:, b], label=txt, alpha=0.3)
            _ax.set_yticks([])
            _ax.set_xlim(0, self._mts.max().max)

            if self._are_filters_set:
                _ax.plot(self._heatmap_x, [self._filter_thresholds[b] for _ in enumerate(self._heatmap_x)])

        if y_lim is not None:
            _ax.set_ylim(0,y_lim)

        _ax.legend()
        if ax is None:
            fig.show()

    def plot(self, highlighted_bins=None, bin_number=12, title="", formatter=None,
             min_max_en=True, split_density_and_overlay=False, x_title="Bins"):
        if highlighted_bins is None:
            #period = int(len(self._mts) / 5)
            shift = len(self._mts) / (2*bin_number)
            highlighted_bins = [int(shift + i) for i in np.linspace(shift, len(self._mts)-shift-1, bin_number)]

        ax1, ax2 = None, None
        figs = []
        if not split_density_and_overlay:
            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            figs.append(fig)
        else:
            figa = plt.figure(figsize=(8, 6))
            ax1 = figa.add_subplot(1, 1, 1)
            figs.append(figa)
            figb = plt.figure(figsize=(8, 6))
            ax2 = figb.add_subplot(1, 1, 1)
            figs.append(figb)

        hbl = len(highlighted_bins)
        rows = 3 # number of columns - the logic is reversed
        cols = math.ceil(hbl / rows)

        self.plot_overlay(ax=ax2,fig=figs[0], highlighted_bins=highlighted_bins,
                          formatter=formatter,min_max_en=min_max_en, x_title=x_title)
        img = self.plot_heatmap(ax=ax1, formatter=formatter, x_title=x_title)
        figs[-1].colorbar(img)

        if hbl > 0:
            #print("hbl", hbl)
            fig2, axs = plt.subplots(cols, rows)
            figs.append(fig2)
            y_lim = np.nanmax(self._heatmap[:,highlighted_bins])
            for hb in range(hbl):
                c = int(hb/rows)
                r = hb % rows

                ax = None
                if len(axs.shape) > 1:
                    ax = axs[c, r]
                else:
                    ax = axs[r]
                self.plot_heatmap_bin(highlighted_bins=[highlighted_bins[hb]], ax=ax, formatter=formatter, y_lim=y_lim)

                if r == 0:
                    ax.set_ylabel("Probability Density")
                if c == cols -1:
                    ax.set_xlabel("Normalized Power")

        for f in figs:
            f.suptitle(title)
            f.show()
        return self

    def __str__(self):
        return "TimeSeriesOverlay\n" + str(self._mts)