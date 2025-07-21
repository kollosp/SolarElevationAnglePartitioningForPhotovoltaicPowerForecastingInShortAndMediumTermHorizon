class TimeSample():
    def __init__(self, timestamp, value):
        self._timestamp = timestamp
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def timestamp(self):
        return self._timestamp

    @value.setter
    def value(self, value):
        self._value = value

    @timestamp.setter
    def timestamp(self, val):
        self._timestamp = val