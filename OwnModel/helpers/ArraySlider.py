import os
import pickle
import io

import numpy as np
import matplotlib.pyplot as plt
import math

from matplotlib.transforms import Bbox
from matplotlib.widgets import TextBox
import matplotlib.cm as cm
import datetime as dt
from datetime import datetime
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from threading import Timer

class DelayedExecution(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

class ArraySliderStatMethodWrapper():
    """
        Create an ArraySliderStatMethodWrapper object for wrapping stats calculator methods

        method - a function which returns tuple(x,y) computed based on window_data (original y)
    """
    def __init__(self, method, **kwargs):
        self._kwargs = kwargs
        self._method = method

    def __call__(self, window_data, **kwargs):
        return self._method(window_data, self._kwargs)

class ArraySlider:
    """
        conv_func - function or callable object transforming zoom area data. It can be used
                    for displaying computed acf, partial acf, fft or other time series statistics.
                    Use ArraySliderStatMethodWrapper wrapper for purpose of conv_func
    """
    def __init__(self,
                 use_global_y_limits=True,
                 one_y_axis=True,
                 show_histograms=False,
                 debug=True,
                 conv_func=None,
                 plot_type=None,
                 x_as_time=True,
                 y_range=None,
                 callable_or_function=None,
                 item_per_page=None):

        self.fig, self.ax = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [4, 1]})
        self._debug = debug
        self._conv_func = conv_func
        self._plot_type = "line" if plot_type is None else plot_type

        self._x_as_time = x_as_time
        self._one_y_axis = one_y_axis
        self._use_global_y_limits = use_global_y_limits
        self._show_histograms = show_histograms
        self._callback = callable_or_function
        self._delayedExecution = None # timer for callback execution (default 5s delay)

        #manualy limit y values
        self._y_range = y_range

        self._one_time_extension = 50
        self._one_time_small_jump = 10

        # configuration of current page
        self.start_page = 0
        self.pages = 100
        self.items_per_page_default = item_per_page if item_per_page is not None else 500
        self.items_per_page = 0
        self.indexes = (0, 0)

        # array of arrays of objects (see in plot function)
        self.y = []
        self.y_enabled = []
        self.y_axis = []
        self.x = None

        self.plots = []
        self.plots_ci = []
        self.plots_y_axis = []
        self._signal_group = False #  set this flag to manually adjust belongment of plots to axis
        # text_box = TextBox(plt.axes([0.1, 0.05, 0.8, 0.075]), 'Evaluate', initial="2**2")
        # text_box.on_submit(None)

        self.verbose_print(f"Constructed. Available commands:\n"
                           f" - 'right' - move one window ahead\n"
                           f" - 'left' - move one window back\n"
                           f" - ']' - move 20 window ahead\n"
                           f" - '[' - move 20 windows back\n"
                           f" - 'up' - move 5 samples ahead\n"
                           f" - 'down' - move 5 samples back\n"
                           f" - 'a' - execute callback function [if available]\n"
                           f" - 'x' - narrow window\n"
                           f" - 'z' - extend window\n"
                           f" - 'm' - make window maximum size\n"
                           f" - 'c' - save window as .svg\n"
                           f" - '0-9' - turn on/off line\n")

    '''
        Print 
    '''
    def print_zoom_area(self):
        self.ax[1].axvspan(self.x[0], self.x[-1], facecolor='w', alpha=1)
        self.ax[1].axvspan(self.x[self.indexes[0]], self.x[self.indexes[1]], facecolor=(.70, .70, .70), alpha=0.9)

    def print_zoom(self):
        it = 0
        #self.ax[0].collections.clear()
        x = self.x[self.indexes[0]:self.indexes[1]+1]

        for ax_i, ax in enumerate(self.y_axis):
            axis_max, axis_min = None,None

            for i,index in enumerate(ax["y_indexes"]):
                y = None
                u = None
                l = None

                # if
                if not self.y_enabled[index]:
                    y = np.zeros(self.indexes[1]-self.indexes[0]+1)
                    y[:] = None
                else:
                    self.y[index]["sub_values"] = self.y[index]["values"][self.indexes[0]:self.indexes[1]+1]
                    y = self.y[index]["sub_values"]


                    if self.y[index]["upper"] is not None:
                        u = self.y[index]["upper"][self.indexes[0]:self.indexes[1] + 1]
                        self.y[index]["sub_values_upper"] = u

                    if self.y[index]["lower"] is not None:
                        l = self.y[index]["lower"][self.indexes[0]:self.indexes[1] + 1]
                        self.y[index]["sub_values_lower"] = l

                if self._conv_func is not None:
                    x, y = self._conv_func(y)
                    if len(x) != len(y):
                        raise ValueError("Using self._conv_func shape mismatch: X({0}) and Y({1}) not equal length".format(len(x), len(y)))

                if self._plot_type == "line":
                    self.plots[it].set_data(x, y)
                elif self._plot_type == "bar":
                    for _y, bar in zip(y, self.plots[it].patches):
                        #print(bar, _y)
                        bar.set_height(_y)

                if self.y_enabled[index]:
                    self.plots[it].set_data(x, y)

                    if u is not None and l is not None:

                        self.ax[0].fill_between(x, l, u, color='b', alpha=.1)



                if self.y_enabled[index]:
                    # self.ax[0].relim()
                    _max = np.nanmax(y)
                    _min = np.nanmin(y)
                    _max = 0 if math.isnan(_max) else _max
                    _min = 0 if math.isnan(_min) else _min

                    if axis_max is None or _max > axis_max:
                        axis_max = _max
                    if axis_min is None or _min < axis_min:
                        axis_min = _min

                    self.ax[0].set_xlim(min(x), max(x))
                    #self.plots_y_axis[it].autoscale_view()
                #go to next plot line object
                it = it + 1


            #if self._conv_func is None:
            axis_max , axis_min = self.compute_limits(axis_max, axis_min)
            if not self._use_global_y_limits:
                self.plots_y_axis[ax_i].set_ylim(axis_min, axis_max)

    def compute_limits(self, axis_max, axis_min):
        #if limits predefined in construction then use it instead
        if self._y_range is not None:
            if len(self._y_range) == 1:
                return self._y_range[0], 0
            elif len(self._y_range) > 1:
                return self._y_range[0], self._y_range[1]

        d = 0.1 * (axis_max - axis_min)
        axis_max = axis_max + d
        axis_min = axis_min - d
        return axis_max , axis_min

    def compute_indexes(self):
        # if the last page
        if self.start_page + self.items_per_page >= len(self.x):
            self.indexes = (len(self.x) - self.items_per_page-1, len(self.x))
        # default switching
        else:
            self.indexes = (self.start_page, self.start_page + self.items_per_page)

        if self.indexes[0] < 0:
            self.indexes = (0, self.items_per_page)

        self.indexes = (int(self.indexes[0]), int(self.indexes[1]))

        if self.indexes[1] - self.indexes[0] != self.items_per_page:
            d = self.indexes[1] - self.indexes[0] - self.items_per_page
            self.indexes = (self.indexes[0], self.indexes[1] + d)
        self.pages = int(len(self.x) / self.items_per_page)


    def reprint(self):
        self.print_zoom()
        self.print_zoom_area()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.verbose_print(
            f"Start page: {self.start_page} | Item per page {self.items_per_page} | Index: {self.indexes}")

    def on_click(self, event):
        if event.key == "right":
            if self.start_page < len(self.x):
                self.start_page = self.start_page + self.items_per_page

        if event.key == "left":
            if self.start_page > 0:
                self.start_page = self.start_page - self.items_per_page

        if event.key == "]":
            if self.start_page + 20 * self.items_per_page < len(self.x):
                self.start_page = self.start_page + 20 * self.items_per_page

        if event.key == "[":
            if self.start_page - 20 * self.items_per_page > 0:
                self.start_page = self.start_page - 20 * self.items_per_page


        if event.key == "up":
            if self.start_page + self._one_time_small_jump  < len(self.x):
                self.start_page = self.start_page + self._one_time_small_jump

        if event.key == "down":
            if self.start_page - self._one_time_small_jump  > 0:
                self.start_page = self.start_page - self._one_time_small_jump

        if event.key == "z":
            if self.items_per_page + self._one_time_extension < len(self.x):
                self.items_per_page = self.items_per_page + self._one_time_extension

        if event.key == "x":
            if self.items_per_page > self._one_time_extension:
                self.items_per_page = self.items_per_page - self._one_time_extension

        if event.key == "m":
            self.items_per_page = len(self.x) - 1

        #save as a vector graphics
        if event.key == "c":
            currentTime = datetime.now().strftime("img%H%M%S")
            buf = io.BytesIO()
            pickle.dump(self.fig, buf)
            buf.seek(0)
            fig2 = pickle.load(buf)
            # remove zoom axis.
            for i, ax in enumerate(fig2.axes):
                if i == 1:
                    fig2.delaxes(ax)
                # else:
                #     axes = ax
            # axes.change_geometry(1, 1, 1)
            fig2.tight_layout()
            fig2.show()
            txt = '{0}/Downloads/{1}.svg'.format(os.path.expanduser('~'), currentTime)

            with open(txt, "x") as f:
                pass
            fig2.savefig(txt, bbox_inches="tight")
            #plt.close(fig2)
            self.verbose_print("Saved in " + txt)
            return

        if '0' <= event.key <= '9':
            index = ord(event.key) - 49
            if len(self.y_enabled) > index:
                self.y_enabled[index] = not self.y_enabled[index]

        # print("Status----")
        # print("Start page:", self.start_page)
        # print("Item per page:", self.items_per_page)

        # next output inside compute_indexes
        self.compute_indexes()

        #execute callback
        if event.key == "a":
            self._callback(self.indexes)

        # if callback not defined then simply reprint.
        if self._callback is None:
            self.reprint()
        else:
            if self._delayedExecution is not None:
                self._delayedExecution.stop()

            self._delayedExecution = DelayedExecution(0.5, self.delayed) # 0.5s delay
            #print(self.indexes)

    def delayed(self):
        self.reprint()

        # if self._callback is not None:
        #     self._callback(self.indexes)

    def plot(self, x, y, label="", signal_group=None, lower=None, upper=None):

        if signal_group is not None:
            self._signal_group = True

        if not np.any([0 if _y is None or math.isnan(_y) else 1 for _y in y]):
            if self._debug:
                print(f"{label} plot rejected due to only nan's value")
            return

        # first initialization
        if self.x is None:
            if self._x_as_time:
                x = [dt.datetime.fromtimestamp(ts) for ts in x]
                # convert timestamps to matplotlib format, it allows to beautify output
                self.x = mdates.date2num(x)
            else:
                self.x = x
            if len(x) >= self.items_per_page_default:
                self.items_per_page = self.items_per_page_default
            else:
                self.items_per_page = len(x) -1
            self.compute_indexes()
            #print(self.indexes)

        _min = np.nanmin(y)
        _max = np.nanmax(y)
        _avg = np.nanmean(y)
        _std = np.nanstd(y)

        _range_min = _avg - 2*_std
        _range_max = _avg + 2*_std
        self.y_enabled.append(True)
        self.y.append({
            "min": _min,
            "max": _max,
            "avg": _avg,
            "std": _std,
            "sub_values": y[self.indexes[0]:self.indexes[1]],
            "sub_values_lower": lower[self.indexes[0]:self.indexes[1]] if lower is not None else None,
            "sub_values_upper": upper[self.indexes[0]:self.indexes[1]] if upper is not None else None,
            "values": y,  # y values
            "label": label,  # title for legend
            "lower": lower, #lower bound
            "upper": upper #upper bound
        })

        is_axis_exists = False
        if not self._signal_group:
            for axis in self.y_axis:
                a_min = axis["avg"] - 2* axis["std"]
                a_max = axis["avg"] + 2* axis["std"]
                # if ranges intersect
                if (a_min <= _range_min <= a_max) or (a_min <= _range_max <= a_max) or self._one_y_axis:
                    axis["y_indexes"].append(len(self.y) -1)
                    is_axis_exists = True
                    break

        else:
            if signal_group is None:
                print("Error in ArraySlider.plot() - if signal_group parameter has been set. Ith has to be provided for all plots!")

            if len(self.y_axis) > signal_group:
                self.y_axis[signal_group]["y_indexes"].append(len(self.y) -1)
                is_axis_exists = True

        # if no corresponding axis found in any method
        if not is_axis_exists:
            self.y_axis.append({
                "min": _min,
                "max": _max,
                "avg": _avg,
                "std": _std,
                "y_indexes": [len(self.y) - 1]
            })

    def x_label(self, label=""):
        self.ax[1].set_xlabel(label)
        pass

    def y_label(self, label):
        self.ax[0].set_ylabel(label)
        self.ax[1].set_ylabel(label)
        pass

    def before_show(self):

        if self._debug:
            for i,ax in enumerate(self.y_axis):
                print("min = {:.3f}, max = {:.3f}, avg = {:.3f}, std = {:.3f} linked:".format(ax["min"],ax["max"],ax["avg"],ax["std"]),
                      ",".join([f"{self.y[i]['label']}(b: {self.y[i]['upper'] is not None})" for i in ax["y_indexes"]]))

        # Define the date format - matplotlib uses the following formats to change timestamps into readable format
        global_date_form = DateFormatter("%Y-%m-%d %H:%M:%S")
        zoom_date_form = DateFormatter("%Y-%m-%d %H:%M:%S")


        # draw plots
        for i,y in enumerate(self.y_axis):
            ax = self.ax[0]
            if i > 0:
                ax = self.ax[0].twinx()
                if i % 2 == 1:
                    #print(ax.spines)
                    ax.spines["right"].set_position(("axes", 0.98 + 0.04 * (i/2)))
                else:
                    ax.yaxis.set_label_position('left')
                    ax.yaxis.set_ticks_position('left')
                    ax.spines.left.set_position(("axes", -0.02 - 0.04 * (i/2)))


            self.plots_y_axis.append(ax)

            ylabel_text = []
            #plot zoom lines and divide axis
            axis_max, axis_min = None,None
            for j,index in enumerate(y["y_indexes"]):
                # plot global lines
                global_plot, = self.ax[1].plot(self.x, self.y[index]["values"])
                # plot zoom lines

                if self._plot_type == "line":
                    x = self.x[self.indexes[0]:self.indexes[1]]
                    y = self.y[index]["sub_values"]
                    u = self.y[index]["sub_values_upper"]
                    l = self.y[index]["sub_values_lower"]
                    zoom_line, = ax.plot(x, y, label=self.y[index]["label"], color=global_plot.get_color())
                    ci_line = None

                    if u is not None and l is not None:
                        ci_line = ax.fill_between(x, l, u, color=global_plot.get_color(), alpha=.1)

                    ax.yaxis.label.set_color(zoom_line.get_color())
                    ax.tick_params(axis='y', color=zoom_line.get_color())
                    ax.spines["left"].set_color(zoom_line.get_color())
                    ax.spines["right"].set_color(zoom_line.get_color())

                elif self._plot_type == "bar":

                    zoom_line = ax.bar(self.x[self.indexes[0]:self.indexes[1]],
                                         self.y[index]["sub_values"], label=self.y[index]["label"],
                                         color=global_plot.get_color())


                zoom_line, = ax.plot(self.x[self.indexes[0]:self.indexes[1]],
                                       self.y[index]["sub_values"], label=self.y[index]["label"], color=global_plot.get_color())

                ylabel_text.append(self.y[index]["label"])

                #compute global limits
                _max, _min = self.y[index]["max"], self.y[index]["min"]
                if axis_max is None or _max > axis_max:
                    axis_max = _max
                if axis_min is None or _min < axis_min:
                    axis_min = _min


                ax.yaxis.label.set_color(zoom_line.get_color())
                ax.tick_params(axis='y',color=zoom_line.get_color())
                ax.spines["left"].set_color(zoom_line.get_color())
                ax.spines["right"].set_color(zoom_line.get_color())



                self.plots_ci.append(ci_line)
                self.plots.append(zoom_line)

            axis_max, axis_min = self.compute_limits(axis_max, axis_min)
            if self._use_global_y_limits:
                ax.set_ylim(axis_min, axis_max)
            ax.set_ylabel(", ".join(ylabel_text))

        self.ax[0].legend(self.plots, [l.get_label() for l in self.plots], loc=0)
        if self._x_as_time:
            self.ax[1].xaxis.set_major_formatter(global_date_form)


        # set formatter only if no converting function has been specified and x is a time
        if self._conv_func is None and self._x_as_time:
            self.ax[0].xaxis.set_major_formatter(zoom_date_form)


        # show data distribution charts
        if self._show_histograms:
            fig, ax = plt.subplots(len(self.y))
            for i,y in enumerate(self.y):

                #print(y["values"].dtype)
                counts, bins = np.histogram(y["values"], bins=100)
                #print(counts,bins)
                ax[i].hist(bins[:-1], bins, weights=counts)

        self.print_zoom_area()
        self.print_zoom()

        self.fig.canvas.mpl_connect('key_press_event', self.on_click)

    def verbose_print(self, *args, **kwargs):
        print(f"ArraySlider:", *args, **kwargs)

    def __str__(self):
        return ""

