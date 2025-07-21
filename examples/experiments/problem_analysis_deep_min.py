from itertools import product
from typing import Tuple
if __name__ == "__main__": pass

import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dimensions import ChainDimension, Elevation

from SlidingWindowExperiment import SlidingWindowExperimentBase
from ANN.model_wrappers import *

import warnings
warnings.filterwarnings('ignore')

class SWE(SlidingWindowExperimentBase):
    def __init__(self, **kwargs):
        super(SWE, self).__init__(**kwargs)
        latitude_degrees = kwargs.get("latitude_degrees")
        longitude_degrees = kwargs.get("longitude_degrees")
        #self.chain is dimension generator. Each class (object) in transformer array creates at least one additional
        # column in returned array. Column is named by "dimension_name" and may be referenced by name in SWE call.
        # to create new Dimension class simple extend BaseDimension in dimensions directory.
        self.chain = ChainDimension(transformers=[
            # y is always here!
            # DayProgress(dimension_name="Day%"),
            # Declination(dimension_name="Declination"),
            Elevation(
                dimension_name="Elevation",
                latitude_degrees=latitude_degrees,
                longitude_degrees=longitude_degrees),
            # Vectorize(lagged=10, step_per_lag=10, base_dimensions=["y"]),
            # SolarDayProgress(
            #     scale=0.01,
            #     latitude_degrees=latitude_degrees,
            #     longitude_degrees=longitude_degrees)
        ])

    # method to be overwritten -
    def fit_generate_dimensions_and_make_dataset(self,train_ts:pd.DataFrame, fh:int, predict_window_length:int) -> Tuple[pd.Series, pd.DataFrame]:
        train_ts_ex = self.chain.fit_transform(train_ts)
        # Use .loc[] for proper DataFrame slicing
        train_ds_y = train_ts_ex.loc[train_ts_ex.index[fh:], "y"]  # y contains ground truth
        train_ds_x = train_ts_ex.iloc[:-fh].copy()  # Create explicit copy for feature data
        return train_ds_y, train_ds_x

    # method to be overwritten
    def predict_generate_dimensions_and_make_dataset(self, predict_ts:pd.DataFrame, test_ts:pd.DataFrame, fh:int) -> Tuple[pd.Series, pd.DataFrame]:
        test_ts_ex = self.chain.transform(predict_ts)
        test_ds_y = test_ts.copy()  # Create explicit copy
        test_ds_x = test_ts_ex.copy()  # Create explicit copy
        return test_ds_y, test_ds_x

def test(n=10, n_steps=28, instance=0):
    file_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2] + [f"..{os.sep}datasets{os.sep}dataset.csv"])
    dataset = pd.read_csv(file_path, low_memory=False)
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
    dataset.index = dataset['timestamp']
    dataset.drop(columns=["timestamp"], inplace=True)
    dataset = dataset[:2*360*288].loc["2020-04-18":]

    latitude_degrees = dataset[f"{instance}_Latitude"][0]
    longitude_degrees = dataset[f"{instance}_Longitude"][0]

    df = pd.DataFrame({}, index=dataset.index)
    df["power"] = dataset[f"{instance}_Power"]

    df = df.dropna()

    swe = SWE(latitude_degrees=latitude_degrees, longitude_degrees=longitude_degrees)
    swe.register_dataset(df)

    dims = [d for d in swe.all_dims if not d in ["Elevation"]]
    dimensions = len(dims)  # base dimensions
    
    available_models = [
        ModelMLP,
        ModelLSTM
    ]
    
    for model_class in available_models:
        model_base = model_class(input_shape=(n, dimensions), output_shape=1)
        model_elevation = model_class(input_shape=(n, dimensions+1), output_shape=1)
        
        model_name = model_base.__str__()
        
        swe.register_model(model_base, f"{model_name}_base", dims=dims, n=n, n_step=n_steps)
        swe.register_model(model_elevation, f"{model_name}_elevation", dims=["Elevation"] + dims, n=n, n_step=n_steps)

    swe.register_metric(mean_absolute_error, "MAE")
    swe.register_metric(mean_squared_error, "MSE")

    _, metrics_df, _ = swe()
    os.makedirs("cm", exist_ok=True)

    print(metrics_df)
    metrics_df["name"] = metrics_df.index
    metrics_df = metrics_df.reset_index(drop=True)
    metrics_df.to_csv("past_concat__new_cnn.csv", mode='a', index=False, header=False)
    
    return metrics_df

if __name__ == "__main__":
    n_values = [
        10, 28#, 56, 84
    ]
    n_steps_values = [10]
    instance_values = [0, 1, 2]

    experiment_configs = [
        {"n": n, "n_steps": n_steps, "instance": instance}
        for n, n_steps, instance in product(n_values, n_steps_values, instance_values)
    ]

    csv_path = "cm/past_all_experiments_concat__new_cnn.csv"
    os.makedirs("cm", exist_ok=True)
    first = True
    for config in experiment_configs:
        try:
            print(f"Running experiment: n={config['n']}, n_steps={config['n_steps']}, instance={config['instance']}")

            metrics_df = test(
                n=config["n"],
                n_steps=config["n_steps"],
                instance=config['instance']
            )

            metrics_df["instance"] = config['instance']
            metrics_df["n"] = config["n"]
            metrics_df["n_steps"] = config["n_steps"]

            base_cols = ["instance", "n", "n_steps"]
            if "name" in metrics_df.columns:
                base_cols = ["name"] + base_cols

            if isinstance(metrics_df.columns, pd.MultiIndex):
                metrics_df.columns = ['_'.join([str(i) for i in col if i]) for col in metrics_df.columns.values]

            cols = base_cols + [col for col in metrics_df.columns if col not in base_cols]
            metrics_df = metrics_df[cols]

            metrics_df.to_csv(csv_path, mode='a', index=True, header=first)
            first = False

        except Exception as e:
            print(f"Error running experiment: {e}")
            continue
