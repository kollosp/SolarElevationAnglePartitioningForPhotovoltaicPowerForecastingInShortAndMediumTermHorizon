import numpy as np
from datetime import datetime
ultraimport('__dir__/../helpers/SolarInsulation.py', 'SolarInsulation', globals=globals())

class MetricDay:
    def __init__(self, latitude_degrees, metric):
        self._latitude_degrees = latitude_degrees
        self._metric = metric

    def __call__(self, reference_ts, computed_ts, upper=None, lower=None):
        elevation = SolarInsulation.elevation(reference_ts.timestamps,
                                   latitude_degrees=self._latitude_degrees,
                                   positive_only=True)
        elevation = elevation > 0
        return self._metric(
            reference_ts.get_by_boolean_array(elevation),
            computed_ts.get_by_boolean_array(elevation),
            upper[elevation] if upper is not None else None,
            lower[elevation] if lower is not None else None,
        )

    def __str__(self):
        return f"Day({self._metric})"

    @property
    def unit(self):
        return f"Day({self._metric})"


class MetricYear:
    def __init__(self, year, metric):
        self._year = year
        self._metric = metric

    def __call__(self, reference_ts, computed_ts, upper=None, lower=None):
        ts = reference_ts.timestamps
        ts_boolean = [datetime.fromtimestamp(t).year == self._year for t in ts]

        return self._metric(
            reference_ts.get_by_boolean_array(ts_boolean),
            computed_ts.get_by_boolean_array(ts_boolean),
            upper[elevation] if upper is not None else None,
            lower[elevation] if lower is not None else None,
        )

    def __str__(self):
        return f"Year({self._metric}, {self._year})"

    @property
    def unit(self):
        return f"Year({self._metric}, {self._year})"


class MetricSeason:
    WINTER=0
    SPRING=1
    SUMMER=2
    AUTUMN=3

    @staticmethod
    def season_string(season):
        if season == MetricSeason.WINTER:
            return "Winter"
        elif season == MetricSeason.SPRING:
            return "Spring"
        elif season == MetricSeason.SUMMER:
            return "Summer"
        elif season == MetricSeason.AUTUMN:
            return "Autumn"
        else:
            return "---"

    @staticmethod
    def get_season(date):

        month = date.month * 100
        day = date.day
        month_day = month + day  # combining month and day
        #print("MetricSeason", month_day)
        if (month_day >= 301) and (month_day <= 531):
            return MetricSeason.SPRING
        elif (month_day > 531) and (month_day < 901):
            return MetricSeason.SUMMER
        elif (month_day >= 901) and (month_day <= 1130):
            return MetricSeason.AUTUMN
        elif (month_day > 1130) or (month_day <= 229):
            return MetricSeason.WINTER
        else:
            raise IndexError("Invalid Input")

        return season


    def __call__(self, reference_ts, computed_ts, upper=None, lower=None):
        ts = reference_ts.timestamps
        seasons_boolean = [MetricSeason.get_season(datetime.fromtimestamp(t)) == self._season for t in ts]

        return self._metric(
            reference_ts.get_by_boolean_array(seasons_boolean),
            computed_ts.get_by_boolean_array(seasons_boolean),
            upper[elevation] if upper is not None else None,
            lower[elevation] if lower is not None else None,
        )

    def __init__(self, season, metric):
        self._season= season
        self._metric= metric

    def __str__(self):
        return f"Season({self._metric}, {MetricSeason.season_string(self._season)})"

    @property
    def unit(self):
        return f"Season({self._metric}, {MetricSeason.season_string(self._season)})"

class MetricYearSeason:
    def __init__(self, year, season, metric):
        self._year = year
        self._season = season
        self._metric = metric

    def compare_date_and_year(self, date):
        return MetricSeason.get_season(date) == self._season and date.year == self._year

    def __call__(self, reference_ts, computed_ts, upper=None, lower=None):
        ts = reference_ts.timestamps
        seasons_boolean = [self.compare_date_and_year(datetime.fromtimestamp(t)) for t in ts]

        return self._metric(
            reference_ts.get_by_boolean_array(seasons_boolean),
            computed_ts.get_by_boolean_array(seasons_boolean),
            upper[elevation] if upper is not None else None,
            lower[elevation] if lower is not None else None,
        )

    def __str__(self):
        return f"YearSeason({self._metric}, {MetricSeason.season_string(self._season)}, {self._year})"

    @property
    def unit(self):
        return f"YearSeason({self._metric}, {MetricSeason.season_string(self._season)}, {self._year})"