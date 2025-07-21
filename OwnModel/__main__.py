from operator import truediv

if __name__ == '__main__': import __config__
import os
import pandas as pd

from OwnModel import OwnModel
import matplotlib.pyplot as plt
from utils.Plotter import Plotter

def main():
    file_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1] + [f"..{os.sep}datasets{os.sep}dataset.csv"])
    dataset = pd.read_csv(file_path, low_memory=False)
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
    dataset.index = dataset['timestamp']
    dataset.drop(columns=["timestamp"], inplace=True)
    dataset = dataset[:2 * 360 * 288].loc["2020-04-18":]
    lat = max(dataset["0_Latitude"])
    models = [OwnModel(
        latitude_degrees = lat,
        elevation_angle_bins = 30,
        power_bins=30,
        moving_avg_window = 12,
        force_policy = OwnModel.POLICY_OPTIMISTIC
    ),OwnModel(
        latitude_degrees = lat,
        elevation_angle_bins = 30,
        power_bins=30,
        moving_avg_window = 12,
        force_policy = OwnModel.POLICY_QUANTITIVE,
        interpolation=True
    )]

    prod = dataset["0_Power"]
    for m in models:
        m.fit(prod, predict_horizon=288)
    for i,m in enumerate(models):
        dataset[f"y_hat_{i}"] = m.predict(prod)

    plotter = Plotter(dataset.index, [dataset[c] for c in ["0_Power"] + [f"y_hat_{i}" for i,_ in enumerate(models)]], debug=False)
    plotter.show()

    plt.show()

if __name__ == '__main__':
    main()