import numpy as np
import matplotlib.pyplot as plt
# from pvlib import location
import pandas as pd
import math
from datetime import datetime

class SolarInsulation():
    def __init__(self):
        pass


    @staticmethod
    def radians2degrees(a):
        return a * 180 / np.pi

    @staticmethod
    def degrees2radians(a):
        return a * np.pi / 180

    @staticmethod
    def sun_maximum_positive_elevation(latitude):
        latitude_radians = SolarInsulation.degrees2radians(latitude)
        #return 90 - 23.45 + latitude
        return np.arcsin(np.cos(latitude_radians) * np.cos(23.45*np.pi/180) + np.sin(latitude_radians) * np.sin(23.45*np.pi/180))

    @staticmethod
    def sun_maximum_negative_elevation(latitude):
        latitude_radians = SolarInsulation.degrees2radians(latitude)
        #return 90 - 23.45 + latitude
        return np.arcsin(-np.sin(latitude_radians) * np.sin(23.45*np.pi/180) - np.cos(latitude_radians) * np.cos(23.45*np.pi/180))

    @staticmethod
    def days_since_1jan(timestamps):
        is_array = True
        if not isinstance(timestamps, (list, tuple, np.ndarray)):
            timestamps = [timestamps]
            is_array = False
        # if array or list has been given as argument
        timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]
        d = np.zeros(len(timestamps))
        for i, t in enumerate(timestamps):
            # days since 1. jan
            jan_1 = t
            jan_1 = jan_1.replace(month=1, day=1)
            d[i] = (t - jan_1).days + 1

        if is_array:
            return d
        else:
            return d[0]


    @staticmethod
    def declination(timestamps):
        is_array = True
        if not isinstance(timestamps, (list, tuple, np.ndarray)):
            timestamps = [timestamps]
            is_array = False
        # if array or list has been given as argument
        timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]
        d = np.zeros(len(timestamps))
        for i, t in enumerate(timestamps):
            # days since 1. jan
            jan_1 = t
            jan_1 = jan_1.replace(month=1, day=1)
            d[i] = (t - jan_1).days + 1

        #return -23.45 * np.cos(360/365 * (d+10))
        ret = -23.45 * np.cos((2 * np.pi * (d+10)/365.25)) * np.pi/180  # compute angle in radians
        if is_array:
            return ret
        else: return ret[0]

    @staticmethod
    def hra(timestamps, longitude_degrees=0):
        is_array = True
        if not isinstance(timestamps, (list, tuple, np.ndarray)):
            timestamps = [timestamps]
            is_array = False
        d = np.zeros(len(timestamps))
        for i, ts in enumerate(timestamps):
            t = datetime.fromtimestamp(ts)
            # t has to be given in utc. Then for each 15 degrees 1 hour in time difference (360degrees/24h = 15)
            # if you use time zones then some adjustment is made.
            d[i] = 15 * np.pi/180 * ((t.hour + longitude_degrees/15) - 12 + t.minute/60)  # hour angle in radians

        if is_array:
            return d
        else:
            return d[0]

    @staticmethod
    def zenith_elevation(timestamp, latitude_degrees):

        dt = datetime.fromtimestamp(timestamp)
        dt = dt.replace(hour=11, minute=0, second=0)
        # dt.hour = 11 # range(24) = 0..23
        # dt.minutes = 0
        # dt.seconds = 0

        ts = datetime.timestamp(dt)
        dec_radians = SolarInsulation.declination(ts)
        # hra = SolarInsulation.hra(ts)
        hra = 0 # zenith is when hra == 0
        latitude_radians = latitude_degrees * np.pi / 180
        # compute hour angle - the angle position of the sun for a given hour
        # compute equation: sin[sin(d)sin(phi)+cos(d)cos(phi)cos(hra)]
        # print(np.pi/2 - dec_radians)
        #return np.cos(latitude_radians) * np.cos(dec_radians), np.sin(latitude_radians) * np.sin(dec_radians)
        return np.arcsin(np.cos(latitude_radians) * np.cos(dec_radians) + np.sin(latitude_radians) * np.sin(dec_radians)) # in radians

        #return  np.sin(dec_radians) + np.cos(dec_radians)

    @staticmethod
    def when_elevation_bin(elevation, timestamp_range, latitude_degrees, longitude_degrees=0, bins=None, positive_only=False):
        i = 0
        for ts in timestamp_range:

            ts_elevation = SolarInsulation.elevation(np.array([ts]), latitude_degrees,
                                                     longitude_degrees, bins, positive_only)
            ts_elevation = ts_elevation[0]
            # works only for elevation bin
            if ts_elevation != elevation:
                return ts, ts_elevation


            i += 1
        return timestamp_range[-1]

    @staticmethod
    def elevation(timestamps, latitude_degrees, longitude_degrees=0, bins=None, positive_only=False):
        if bins and int(bins) % 2 != 0:
            raise ValueError(f"Bins has to be even parity! Given value is: {bins}")
        if timestamps.shape is not None and len(timestamps.shape) > 1:
            raise ValueError("Timestamps array is not 1-dimensional. Make it flatten!")

        is_array = True
        if not isinstance(timestamps, (list, tuple, np.ndarray)):
            timestamps = [timestamps]
            is_array = False
        latitude_radians = SolarInsulation.degrees2radians(latitude_degrees)
        d = np.zeros(len(timestamps))

        temp_mx_globally = SolarInsulation.sun_maximum_positive_elevation(latitude_degrees)
        temp_mi_globally = SolarInsulation.sun_maximum_negative_elevation(latitude_degrees)
        mx = temp_mx_globally # globally maximum elevation (during the longest day)
        mi = temp_mi_globally if not positive_only else 0 # globally and min for specified latitude or 0.

        # compute bin width
        dt = 1
        if bins is not None:
            # if hra is used then with shoudl be twice bigger
            dt = (mx - mi) / (bins/2)

        for i, ts in enumerate(timestamps):
            dec_radians = SolarInsulation.declination(ts)
            hra = SolarInsulation.hra(ts, longitude_degrees)

            # compute hour angle - the angle position of the sun for a given hour
            # compute equation: arcsin[sin(d)sin(phi)+cos(d)cos(phi)cos(hra)]
            d[i] = np.arcsin(np.sin(latitude_radians) * np.sin(dec_radians) + np.cos(latitude_radians) * np.cos(dec_radians) * np.cos(hra))

            if positive_only and d[i] < 0: # remove negatives
                d[i] = 0

            #if bins enabled
            if bins is not None:
                d[i] += abs(temp_mi_globally) # move up - remove negative values
                if hra < 0:
                    d[i] = np.rint(d[i] / dt).astype(int)
                else:
                    # distuinguish by hra
                    d[i] = bins - np.rint(d[i] / dt).astype(int)

        if is_array:
            if bins is not None:
                return d.astype(int)
            else:
                return d.astype(float)
        else:
            return d[0]

    @staticmethod
    def elevation_bins(timestamps, latitude_degrees, bins=20, noon_division=True, positive_only=True, longitude_degrees=0):
        a = SolarInsulation.elevation(timestamps, bins=bins,
                                      latitude_degrees=latitude_degrees, positive_only=positive_only)

        # if noon division then split same angels before and after noon into separated bins
        if noon_division:
            hra = SolarInsulation.hra(timestamps, longitude_degrees)
            mx = np.max(a)
            cond = [h > 0 and 0 < _a <= mx for h, _a in zip(hra, a)]
            #a = np.where(cond, a, mx - a + 100)

        return a

    # @staticmethod
    # def normalize_insulation_to_data_structure(power):
    #     # 1.0 for power == 1000 W/m^2 (there is also degradation factor effect)
    #     return power / 1000
    #
    # @staticmethod
    # def declination_angle(days_since_equinox=0, normalize=False):
    #     # days_since_equinox how many days from equinox (21 march) https://www.sciencedirect.com/topics/engineering/solar-altitude-angle
    #     if normalize:
    #         return (23+27.0/60.0) * np.sin(2*math.pi * days_since_equinox / 365.25)
    #     else:
    #         return np.sin(2*math.pi * days_since_equinox / 365.25)
    #
    # @staticmethod
    # def solar_declination(dates, normalize=False):
    #     times = pd.DatetimeIndex(dates.astype('datetime64[ms]'))
    #     declination = np.zeros(len(times))
    #     for i, t in enumerate(times):
    #         march_21 = 0
    #         #day after eqiunox
    #         if t.month > 3 or (t.month == 3 and t.day >= 21):
    #             march_21 = t.replace(month=3, day=21)
    #
    #         #day before eqiunox
    #         else:
    #             march_21 = t.replace(year = t.year-1, month=3, day=21)
    #         days_since_equinox = (t - march_21).days
    #
    #         declination[i] = SolarInsulation.declination_angle(days_since_equinox, normalize=normalize)
    #
    #     return declination
    #
    # @staticmethod
    # def print_declination(lat, long, date):
    #     days = np.linspace(0., 364, 365)
    #     Q = SolarInsulation.declination_angle(days)
    #     #print(Q)
    #     fig, ax = plt.subplots()
    #     ax.plot(days, Q)
    #     ax.set_xlim(0, 364);
    #     # ax.set_xticks([-90, -60, -30, -0, 30, 60, 90])
    #     ax.set_xlabel('Days')
    #     ax.set_ylabel('Angle')
    #     ax.grid()
    #     # ax.set_title('Daily average insolation on March 21')
    #     plt.show()
    #
    # @staticmethod
    # def elevation_angle(lat, timestamps):
    #     np.sin()
    #
    # @staticmethod
    # def print_solar_position(lat, long, date):
    #     Q, date = SolarInsulation.solar_position(lat, long, date)
    #     Q = Q['elevation']
    #     # print(Q)
    #     fig, ax = plt.subplots()
    #     ax.plot(date, Q)
    #     #ax.set_xlim(0, 364);
    #     # ax.set_xticks([-90, -60, -30, -0, 30, 60, 90])
    #     ax.set_xlabel('Days')
    #     ax.set_ylabel('Angle')
    #     ax.grid()
    #     # ax.set_title('Daily average insolation on March 21')
    #     plt.show()
    #
    # @staticmethod
    # def insulation(lat, long, date):
    #     times = pd.DatetimeIndex(date.astype('datetime64[ms]'))
    #     loc = location.Location(lat, long)
    #     weather = loc.get_clearsky(times)
    #     # print(weather)
    #     # fig, ax = plt.subplots()
    #     # ax.plot(times, weather['ghi'])
    #     # ax.plot(times, weather['dhi'])
    #     # ax.plot(times, weather['dni'])
    #     # plt.show()
    #     return weather, times
    #
    #
    #
    # @staticmethod
    # def solar_elevation(lat, long, timestamps):
    #     print(timestamps)
    #     loc = location.Location(lat, long)
    #     pos = loc.get_solarposition(timestamps)
    #     return pos['elevation'].to_numpy()
    #
    # @staticmethod
    # def solar_position(lat, long, date):
    #     times = pd.DatetimeIndex(date.astype('datetime64[ms]'))
    #     loc = location.Location(lat, long)
    #     pos = loc.get_solarposition(times)
    #     # print(weather)
    #     # fig, ax = plt.subplots()
    #     # ax.plot(times, weather['ghi'])
    #     # ax.plot(times, weather['dhi'])
    #     # ax.plot(times, weather['dni'])
    #     # plt.show()
    #     pos['elevation'] = pos['elevation'] / 90
    #
    #     return pos, times

