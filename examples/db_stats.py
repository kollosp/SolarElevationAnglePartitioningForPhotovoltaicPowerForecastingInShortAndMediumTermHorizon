import __config__
import pandas as pd
import os
from functools import reduce
from utils.DataframeChainProcessor import DataframeChainProcessor
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Database properties analysis',
        description='Script reads database csv and check is properties. It generates latex table as response.')
    args = parser.parse_args()

    file_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1] + [f"..{os.sep}datasets{os.sep}dataset.csv"])
    dataset = pd.read_csv(file_path, low_memory=False)
    # self.full_data = self.full_data[30:]
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
    dataset.index = dataset['timestamp']
    dataset.drop(columns=["timestamp"], inplace=True)

    dcp = DataframeChainProcessor()

    df = dcp.make_step("_Step1", dataset, lambda d: d[["0_Power", "1_Power", "2_Power"]])

    df1 = dcp.make_step("_Step0.3", df, lambda d: dcp.transform_columns_to_multi_index(d))
    df1 = dcp.make_step("_Step0.4", df1, lambda d: d.stack())
    df1 = dcp.make_step("_Step0.5", df1, lambda d: d.reset_index("Dataset"))
    df1 = dcp.make_step("_Step0.6", df1, lambda d: d.groupby("Dataset").agg([lambda x: x.dropna().first_valid_index(), lambda x: x.dropna().last_valid_index()]))
    df1 = dcp.make_step("_Step0.7", df1, lambda d: dcp.flatten_index(d, "Attribute"))
    df1 = dcp.make_step("_Step0.8", df1, lambda d: d.rename(columns={"Power_<lambda_0>": "Begin Date", "Power_<lambda_1>": "End Date"}))
    df1 = dcp.make_step("_Step0.9", df1, lambda d: dcp.drop_timestamp_column_time(d, "Begin Date"))
    df1 = dcp.make_step("_Step0.10", df1, lambda d: dcp.drop_timestamp_column_time(d, "End Date"))
    df1 = dcp.make_step("Time Range", df1, lambda d: dcp.add_column(d, "Days", lambda x: (x["End Date"] - x["Begin Date"])))


    df2 = dcp.make_step("_Step2.2", df, lambda d: d.fillna(0))
    df2 = dcp.make_step("_Step2.3", df2, lambda d: dcp.transform_columns_to_multi_index(d))
    df2 = dcp.make_step("_Step2.4", df2, lambda d: d.stack())
    df2 = dcp.make_step("_Step2.5", df2, lambda d: d.reset_index("Dataset"))
    df2 = dcp.make_step("Max for each dataset", df2, lambda d: d.groupby("Dataset").max())

    df3 = dcp.make_step("_Step1", dataset, lambda d: d[["0_MountedPower", "1_MountedPower", "2_MountedPower"]])
    df3 = dcp.make_step("_Step2.3", df3, lambda d: dcp.transform_columns_to_multi_index(d))
    df3 = dcp.make_step("_Step2.4", df3, lambda d: d.stack())
    df3 = dcp.make_step("_Step2.5", df3, lambda d: d.reset_index("Dataset"))
    df3 = dcp.make_step("Mounted Power", df3, lambda d: d.groupby("Dataset").max())

    df4 = dcp.make_step("_Step1", dataset, lambda d: d[["0_Latitude", "1_Latitude", "2_Latitude"]])
    df4 = dcp.make_step("_Step2.3", df4, lambda d: dcp.transform_columns_to_multi_index(d))
    df4 = dcp.make_step("_Step2.4", df4, lambda d: d.stack())
    df4 = dcp.make_step("_Step2.5", df4, lambda d: d.reset_index("Dataset"))
    df4 = dcp.make_step("Mounted Power", df4, lambda d: d.groupby("Dataset").max())

    df5 = dcp.make_step("_Step1", dataset, lambda d: d[["0_Longitude", "1_Longitude", "2_Longitude"]])
    df5 = dcp.make_step("_Step2.3", df5, lambda d: dcp.transform_columns_to_multi_index(d))
    df5 = dcp.make_step("_Step2.4", df5, lambda d: d.stack())
    df5 = dcp.make_step("_Step2.5", df5, lambda d: d.reset_index("Dataset"))
    df5 = dcp.make_step("Mounted Power", df5, lambda d: d.groupby("Dataset").max())

    df_f = reduce(lambda  left,right: pd.merge(left,right,left_index=True, right_index=True, how='inner'), [df1, df2, df3, df4, df5])
    print(df_f.to_latex())