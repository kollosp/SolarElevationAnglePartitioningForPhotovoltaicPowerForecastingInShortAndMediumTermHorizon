import copy
import sys
import time
import traceback
from datetime import datetime
from itertools import product
from typing import Tuple
if __name__ == "__main__": import __config__

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import argparse
from problem_analysis_run_experiment import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Short Experiment -- less models and less time only to check',
        description='THIS IS A SHORTER VERSION OF EXPERIMENTAL SCRIPT. Program performs experiments. It iterate over all provided combinations of forecasting horizons, '
                    'different models, data sets, n-values, n_steps and dimensions (features). This script does not use '
                    'any arguments. This message is only for information. The results of this script execution are saved '
                    'in \'cm\' directory. Output consists of two types files the \'concat\' files contains aggregated '
                    'model results, while \'prediction\' files contain prediction results. The results can be processed '
                    'by Results Analysis script (problem_analysis_results).')
    args = parser.parse_args()
    parser.print_help()

    main(
        forecasting_horizons=[72,144,288], #24h
        prediction_step_len=72, #it must be equal to the shortest forecasting horizon
        models = [
            (lambda *args: make_pipeline(PolynomialFeatures(12), LinearRegression()), "LR(12)"),
            (lambda *args: make_pipeline(PolynomialFeatures(16), LinearRegression()), "LR(16)"),
            (lambda *args: RandomForestRegressor(), "RF")
        ],
        instance_values = [
            0, 1, 2
        ],
        n_values = [
            None #1  #, 56, 84
        ], n_steps_values = [None],
        covered_dimensions={"y","SolarDay%", "Declination", "Elevation"},
        excluded_dimensions=[]
    )

