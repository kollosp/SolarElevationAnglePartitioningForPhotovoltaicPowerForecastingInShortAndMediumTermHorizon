import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import math
from datetime import datetime

class TimestampsProcessing():
    def __init__(self):
        pass

    @staticmethod
    def day_bins(timestamps, bins=288):
        """Function calculates bin in day. Each moment in particular day is assigned to other bin"""
        sample_per_bin = 3600 * 24 / bins

        is_array = True
        if not isinstance(timestamps, (list, tuple, np.ndarray)):
            timestamps = [timestamps]
            is_array = False
        d = np.zeros(len(timestamps))
        for i, ts in enumerate(timestamps):
            t = datetime.fromtimestamp(ts)
            day_seconds = t.hour * 3600 + t.minute * 60 + t.second
            d[i] = int(day_seconds / sample_per_bin)

        if is_array:
            return d.astype(int)
        else:
            return d[0]

        return a

    @staticmethod
    def date_day_bins(timestamps):
        """Function assigns each day to other bin"""
        if len(timestamps) == 0:
            return  []
        first = timestamps[0]
        a = []
        seconds_per_days = 60*60*24
        for i, ts in enumerate(timestamps):
            a.append(int((ts-first) / seconds_per_days))

        return a