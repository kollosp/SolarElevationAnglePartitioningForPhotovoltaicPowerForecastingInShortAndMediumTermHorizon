if __name__ == "__main__": import __config__
from utils.DataframeChainProcessor import DataframeChainProcessor
import pandas as pd
from SlidingWindowExperiment.StatisticalAnalysis import StatisticalAnalysis
import os
import argparse

def refactor_latex_table(latex_str: str):
    return (latex_str.replace("_0", "").replace("_1", "").replace("_2", "")
            .replace("%", "\\%")
            .replace("Declination", "D")
            .replace("Elevation", "E")
            .replace("SolarDay", "S")
            .replace("instance", "\\ds{}"))

def main(directories, safilepath):
    dpc = DataframeChainProcessor()
    sa = StatisticalAnalysis(directories)
    concat =  sa.concat_df
    df = dpc.make_step("_init",concat, lambda d: d[["model", "instance", "fh", "MAE_Mean", "MAE_Std", "MAE_Samples"]])
    df = dpc.make_step("_drop index", df, lambda d: d.reset_index().drop(columns=["index"]))
    df = dpc.make_step("truncated table", df, lambda d: pd.concat([d.head(5), d.tail(5)]))
    s1 = df.to_latex()
    print(s1)

    rep = pd.read_csv(safilepath, sep=",", decimal='.')

    df = dpc.make_step("_init2",rep,
                       lambda d: d[["model", "ref_model", "metric", "instance", "fh", "ttest_p", "ttest_rejectH0", "IR", "MetricStat_Mean", "MetricStat_ref_Mean"]])
    df = dpc.make_step("_drop index", df, lambda d: d.reset_index().drop(columns=["index"]))
    df = dpc.make_step("_remove PT", df, lambda d: d.loc[d['metric'] != "PT"])
    df = dpc.make_step("_truncated table", df, lambda d: pd.concat([d.head(5).reset_index().drop(columns=["index"]), d.tail(5)]))
    print(refactor_latex_table(df.to_latex(float_format="%.3f")) )

    df = dpc.make_step("filter data to have only one row for each testcase",
                       rep,
                       lambda d: d.loc[
                           (d["ref_model"].str.contains("[y]", regex=False)) &
                           (d["ref_model"].str.split("_").str[0] == d["model"].str.split("_").str[0]) &
                           (d["ref_model"].str.split("]").str[-1] == d["model"].str.split("]").str[-1])
                       ])

    without_pt_df = dpc.make_step("remove PT metric",
                       df, lambda d: d.loc[d["metric"] != "PT"])

    best_metric_results_df = dpc.make_step("Pivot metric best results",
                       without_pt_df, lambda d: pd.pivot_table(d, "MetricStat_Mean",
                                                    index=["fh"],
                                                    columns=["metric", "instance"],
                                                    aggfunc="min"))
    print(refactor_latex_table(best_metric_results_df.to_latex(float_format="%.3f")))
    df = dpc.make_step("_Pivot metric avg results",
                       without_pt_df, lambda d: pd.pivot_table(d, "MetricStat_Mean",
                                                    index=["fh"],
                                                    columns=["metric", "instance"],
                                                    aggfunc=["mean","std"]))


    avg_metric_results_df = dpc.make_step("Pivot metric avg results",
                       df, lambda d: pd.concat([d.loc[:, "mean"],  d.loc[:, "std"]]))

    print(refactor_latex_table(avg_metric_results_df.to_latex(float_format="%.3f")))
    df = dpc.make_step("_stack df",
                       best_metric_results_df, lambda d: d.stack().stack().reset_index())
    _renamed_columns_df = dpc.make_step("_raname columns",
                       df, lambda d: d.rename(columns={0:"Value"}))
    df = dpc.make_step("_add label column (model config string)",
                       _renamed_columns_df, lambda d: d.apply(lambda row:
                                             [
                                                 row["fh"],
                                                 row["instance"],
                                                 row["metric"],
                                                 without_pt_df.loc[
                                                     (without_pt_df["fh"] == row["fh"]) &
                                                     (without_pt_df["instance"] == row["instance"]) &
                                                     (without_pt_df["metric"] == row["metric"]) &
                                                     (without_pt_df["MetricStat_Mean"] == row["Value"]), "model"].max()
                                             ], axis=1, result_type="expand"))
    df = dpc.make_step("_name columns",df, lambda d: d.rename(columns={0:"fh", 1: "instance", 2:"metric",3: "model"}))

    best_metric_model_names_results_df = dpc.make_step("pivot table (set index and unstack",df, lambda d: d.set_index(["fh", "instance", "metric"]).unstack(["metric"]))

    df = dpc.make_step("_add label column (model config string)",
                       _renamed_columns_df, lambda d: d.apply(lambda row:
                                                              [
                                                                  row["fh"],
                                                                  row["instance"],
                                                                  row["metric"],
                                                                  without_pt_df.loc[
                                                                      (without_pt_df["fh"] == row["fh"]) &
                                                                      (without_pt_df["instance"] == row["instance"]) &
                                                                      (without_pt_df["metric"] == row["metric"]) &
                                                                      (without_pt_df["MetricStat_Mean"] == row[
                                                                          "Value"]), "ttest_p"].max()
                                                              ], axis=1, result_type="expand"))
    df = dpc.make_step("_name columns", df,
                       lambda d: d.rename(columns={0: "fh", 1: "instance", 2: "metric", 3: "model"}))

    best_metric_model_names_pvalue_df = dpc.make_step("pivot table (set index and unstack", df,
                                                       lambda d: d.set_index(["fh",  "metric", "instance"]).sort_index().unstack(
                                                           ["metric", "instance"]))
    df = dpc.make_step("Remove tests that are not statististically significatn",
                       without_pt_df, lambda d: d.loc[(d["ttest_rejectH0"] == 1) & (d["IR"] > 0)])
    avg_improvement_ratio_df = dpc.make_step("Pivot improvement ratio avg results",
                       df, lambda d: pd.pivot_table(d, "IR",
                                                    index=["fh"],
                                                    columns=["metric", "instance"],
                                                    aggfunc=["mean"],
                                                    margins=True,
                                                    margins_name='Total'))

    print(refactor_latex_table(best_metric_model_names_results_df.to_latex(float_format="%.5f")) )
    print(refactor_latex_table(best_metric_model_names_pvalue_df.to_latex(float_format="%.3f")) )
    print(refactor_latex_table(avg_improvement_ratio_df.to_latex(float_format="%.3f")) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Article Tables Generator',
        description='This script generates tables that were used in the article. This script can be run on Results Analysis output')
    parser.add_argument('inputdir', help="Directory that contains output .csvs from experimental script")
    parser.add_argument('safilepath', help="Name of the Result Analysis (problem_analysis_results) output file")

    args = parser.parse_args()
    parser.print_help()

    main([args.inputdir], args.safilepath)