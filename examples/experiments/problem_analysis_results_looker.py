if __name__ == "__main__": import __config__
import pandas as pd
import numpy as np
import os
from dimensions import Elevation
from matplotlib import pyplot as plt

from utils.Plotter import Plotter

def print_full(x):
    pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 2000)
    # pd.set_option('display.float_format', '{:20,.2f}'.format)
    # pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

if __name__ == "__main__":

    pass
    # plotter = Plotter(df.index, [df[i] for i in [f"{j}_Power" for j in range(0,4)]] + [elevation], debug=False)
    # plotter.show()
    # plt.show()


