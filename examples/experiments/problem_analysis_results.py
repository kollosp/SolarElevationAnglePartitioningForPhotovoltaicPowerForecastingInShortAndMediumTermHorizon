if __name__ == "__main__": import __config__
import pandas as pd

from SlidingWindowExperiment.StatisticalAnalysis import StatisticalAnalysis


import numpy as np

import os


def add_column(df, column_name, callback=lambda i,row: ""):
    for i, row in df.iterrows():
        print(i,row)
        df[i,column_name] = callback(i, row)
    return df

def main(directory, tested_dimensions):
    sa = StatisticalAnalysis(directory, tested_dimensions=tested_dimensions)
    concat_df = sa.concat_df
    model_reference = sa.get_model_and_its_reference_df()
    print(f"Metrics in report: {sa.get_available_metric_columns()}")

    results_df = sa.get_results_df(metric="MAE")
    # pd.merge(model_reference, )
    mean_df = sa.get_model_improvement_ratio_df()
    ttest_df = sa.get_ttest_df()
    print(ttest_df)

    mean_df.to_csv(f"cm{os.sep}sa.csv", sep=",", decimal='.', index=False)

    # concat_df["RefModel"] = concat_df["Model"].str.replace("Elevation", "").str.replace(",,", ",")
    # concat_df["RefModel"] = concat_df["RefModel"].str.replace("Declination", "").str.replace(",,", ",")
    # concat_df["RefModel"] = concat_df["RefModel"].str.replace("SolarDay%", "").str.replace(",,", ",")
    # concat_df["RefModel"] = concat_df["RefModel"].str.replace(",]", "]")
    # tests_only = concat_df[ concat_df["isRef"] == False]
    # tests_and_refs = pd.merge(left=tests_only, right=concat_df[["Model", "prediction_path","concat_file_path"]],
    #          how="left", right_on="Model", left_on="RefModel")
    # # concat_df = add_column(concat_df, "RefModel", lambda i,row: row["Model"].replace("Elevation", ""))
    # print(concat_df)
    # print(tests_and_refs[["Model_x", "isRef", "Model_y"
    # #    , "prediction_path_x", "concat_file_path_x",  'prediction_path_y', 'concat_file_path_y'
    # ]])
    # # for i,row in concat_df.iterrows():
    # #     print(row)



if __name__ == "__main__":
    main(f"cm{os.sep}ex1", tested_dimensions=["Declination","Elevation","SolarDay%"])