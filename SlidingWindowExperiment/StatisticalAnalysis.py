import os
from typing import List

import numpy as np
import pandas as pd


class StatisticalAnalysis:
    """
        Class performs statistical analysis of results obtained by SlidingWindowExperimentsBase-based experiments classes
    """
    def __init__(self, concat_files_directory:str, **kwargs):
        self.modelCnf = "modelCnf"
        self._prediction_path_column = kwargs.pop('prediction_path_column', "prediction_path")
        self._model_name_column = kwargs.pop('model_name_column', "_model_name")
        self._model_dimensions_column = kwargs.pop('model_dimensions_column', "_model_dimensions")
        self._ref_model_dimensions_column = kwargs.pop('ref_model_dimensions_column', "_ref_model_dimensions")

        self._model_column = kwargs.pop('model_column', "model") # transformed model configuration column with ordered dimensions
        self._ref_model_column = kwargs.pop('ref_model_column', "ref_model") # transformed ref model column with order dimensions

        self._tested_dimensions = kwargs.pop('tested_dimensions', [])
        self._concat_df = self.load_concat_files(self.find_concat_files_in_directory(concat_files_directory))
        self.process_df()

    @property
    def concat_df(self) -> pd.DataFrame:
        return self._concat_df

    @staticmethod
    def find_concat_files_in_directory(concat_files_directory: str):
        """
        function finds all 'concat' files located in given directory. Concat files should be created
        from SlidingWindowExperimentsBase metric_df results (may be concatenated)
        """
        r = []
        for file in os.listdir(concat_files_directory):
            if "concat" in file:
                r.append(os.path.join(concat_files_directory, file))
        return r


    def load_concat_files(self, files:List[str]):
        """
        Function loads all files and concatenates them into a single dataframe. After the concatenation is done, function
        checks types of columns that contains 'Mean' or 'Std' substrings in their names. Those columns must be floats.
        They represent metrics summary.
        """
        df = pd.DataFrame()
        for file in files:
            read = pd.read_csv(file, delimiter=",", header=0, decimal=".")
            read["concat_file_path"] = file
            read[self._prediction_path_column] = os.path.dirname(file) +os.sep +read[self._prediction_path_column].str.split("/").str[-1]
            df = pd.concat([df, read])

        metrics = self._get_metric_columns(df)
        for column in metrics:
            df.loc[:, column] = pd.to_numeric(df[column], errors='coerce')
            # cast Mean and std columns to float. If experiment returned exception those column
            # will contain its textual representation. Those values will be replaced with nans
            df = df[~df[column].isna()] # remove NaN column
        return df

    def get_modelCnf_model_name_conversion_dict(self):
        """
        Function returns a dictionary that can be used to map modelCnf with model name e.g.:
        {
        'KNN(10)_0[y,Declination,Elevation,SolarDay%]': 'KNN(10)_0[Declination,Elevation,SolarDay%,y]',
         'LR(12)_0[y,Declination,Elevation,SolarDay%]': 'LR(12)_0[Declination,Elevation,SolarDay%,y]', ...
        }
        """
        return self.concat_df.set_index(self.modelCnf)[self._model_column].to_dict()

    def get_predictions_directory(self):
        """
        Function returns directory where predictions dataframes are stored. Directory is keyed by prediction path
        """
        predictions_to_load = self.concat_df[self._prediction_path_column]
        predictions_to_load = np.unique(predictions_to_load)
        dataframes = {}
        model_name_conversion_dict = self.get_modelCnf_model_name_conversion_dict()

        # load all necessary prediction files to json
        for prediction in predictions_to_load:
            df = pd.read_csv(prediction, delimiter=",", header=0, decimal=".")
            # get only prediction error metrics
            metric_columns = [c for c in df.columns if ":" in c and not "FT" in c] # metrics are marked as <model>:<metric>
            if "Unnamed: 0" in metric_columns:
                metric_columns.remove("Unnamed: 0")
            df = df[metric_columns]
            df.dropna(inplace=True) # drop empty rows (not all rows contains metrics -- only one per day)
            #split  <model>:<metric> into multiindex
            mi = [tuple(c.split(":")) for c in df.columns]
            mi = [(model_name_conversion_dict[c[0]], c[1]) for c in mi] #change modelCnf to model
            df.columns = pd.MultiIndex.from_tuples(mi)
            dataframes[prediction] = df

        return dataframes

    def get_ttest_df(self):
        # mean_df = self.get_results_df()#.stack([0])
        model_join_df = self.get_model_and_its_reference_df()
        df = self.concat_df[[self._model_column, self._prediction_path_column]]
        combined_predictions_df = pd.merge(model_join_df, df, on=self._model_column)
        # this table contains 4 columns: model, ref_model, prediction_path and ref_prediction_path.
        ref_prediction_path_column = "ref_"+self._prediction_path_column
        combined_predictions_df = pd.merge(
            combined_predictions_df,
            df.rename(columns={self._prediction_path_column: ref_prediction_path_column}),
            on=self._model_column)

        metrics = self.get_available_metric_columns()
        # predictions_df = self.get_predictions_directory()


        return combined_predictions_df


    def get_model_improvement_ratio_df(self, metric=None, model_fiter_str=""):
        """
        Function performs simple mean-based models' experimental results between models and their reference. It calculates
        improvement ratio IR: 100% * (<ref metric> - <model metric>) / <ref metric>. Positive IR means that model is
        better than its reference.
        metric: a filter for metric it is passed to get_results_df
        model_fiter_str: a filter for model name (not used yet)
        """
        mean_df = self.get_results_df(metric=metric)
        mean_df.columns = mean_df.columns.reorder_levels(order=[1, 0])

        mean_df = mean_df.loc[:, "Mean"]
        metrics = mean_df.columns  # store list of metrics to use it for calculations
        mean_df.reset_index(inplace=True) #drop index to make merges easier
        model_join_df = self.get_model_and_its_reference_df()
        suffixes = ('', '_ref')
        combined_mean_df = pd.merge(mean_df, model_join_df,  left_on=self._model_column, right_on=self._model_column)
        combined_mean_df = pd.merge(combined_mean_df, mean_df, left_on=self._ref_model_column,
                                    right_on=self._model_column, suffixes=suffixes)
        combined_mean_df["vs."] = combined_mean_df[self._model_column] + " vs. " + combined_mean_df[self._ref_model_column]
        ret = combined_mean_df[[self._model_column]]

        for metric in metrics:
            x = combined_mean_df[metric + suffixes[1]]
            y = combined_mean_df[metric + suffixes[0]]

            ret.loc[:, metric] = (x - y) / x

        ret.set_index("model", inplace=True)

        # ret.loc["Improved", :] =  ret.aggregate(lambda x: x[x >= 0].count(),axis=0)
        # ret.loc["Worsed", :] =  ret.aggregate(lambda x: x[x < 0].count(),axis=0)
        # ret.loc["Avg. IR", :] =  ret.loc["Improved", :] / (ret.loc["Worsed", :] + ret.loc["Improved", :])
        # ret = pd.merge(ret, combined_mean_df[[self._model_column, "vs."]], on=self._model_column, how="left")
        # ret = pd.merge(ret, self.concat_df[[self._model_column, "instance", self._model_name_column]], on=self._model_column, how="left")
        # ret["vs."].fillna("", inplace=True)
        return ret

    def get_available_metric_columns(self):
        return np.unique([x.split("_")[0] for x in self._get_metric_columns()])

    def _get_metric_columns(self, df=None):
        """
        Function return list of metrics available in dataset
        df: Optional dataframe. If None self.concat_df is used.
        """
        suffixes = ["Mean", "Std"]
        if df is None:
            df = self.concat_df
        return [x for x in df.columns if any(suffix in x for suffix in suffixes)]

    def get_results_df(self, metric=None):
        """
        function returns view from concat_df. It contains real model name self.modelCnf, model name with ordered dimensions
        model and metric results. Metric are find by "Mean" or "Std" suffixes
        metric: selected metric to be included in table (element from self.get_available_metric_columns()) if metric_name is None, return all metrics
        """
        metric_columns = self._get_metric_columns()
        ret = self.concat_df[metric_columns + [self._model_column]]
        ret.set_index(self._model_column, inplace=True)
        ret.columns = pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in ret.columns], names=["Metric", "Attr."])

        if metric is not None:
            return ret[metric]
        else:
            return ret

    def get_model_and_its_reference_df(self, tested_dimensions=None, default_reference_dims="y"):
        """
        Function transforms self.modelCnf from concat_df table. It finds reference model for each model that was evaluated
        during tests and is included in concat_df table. Function returns two-column table containing model and model_ref
        columns. The returned table can be used to join result tables.

        tested_dimensions:
            list of tested dimensions that must be removed from Model column to find reference model.
        default_reference_dims:
            string containing default reference model dimensions. It is used when after dimensions removal dimension list
            is empty.
        """
        if tested_dimensions is None:
            tested_dimensions = self._tested_dimensions
        concat_df = self.concat_df[[self._model_column, "instance"]]
        concat_df[self._model_name_column] = concat_df[self._model_column].str.split("_").str[0]
        # extract group inside brackets [...]
        concat_df[self._model_dimensions_column] = concat_df[self._model_column].str.extract("\[(.+)\]")
        # removes all tested_dimensions from
        concat_df[self._ref_model_dimensions_column] = concat_df[self._model_dimensions_column].apply(
            lambda x: ",".join([xx for xx in x.split(",") if not xx in tested_dimensions])
            # lambda x: type(x)
        )
        # in case of this experiment default ref model is that one which used "y" as dimension
        concat_df[self._ref_model_dimensions_column] = concat_df[self._ref_model_dimensions_column].replace("",default_reference_dims)
        concat_df[self._ref_model_column] = \
            concat_df[self._model_name_column] + "_" + \
            concat_df["instance"].astype(str) + "[" + concat_df[self._ref_model_dimensions_column] + "]"


        return concat_df[[self._model_column, self._ref_model_column]]


    def process_df(self):
        
        self._concat_df.rename(columns={"Unnamed: 0": self.modelCnf}, inplace=True)
        # self._concat_df[self._prediction_path_column] = self._concat_df[self._prediction_path_column].str.split("/").str[
        #     -1]

        self._concat_df[self._model_name_column] = self._concat_df[self.modelCnf].str.split("_").str[0]
        # extract group inside brackets [...]
        self._concat_df[self._model_dimensions_column] = self._concat_df[self.modelCnf].str.extract("\[(.+)\]")
        # removes test instances which has no dimensions
        self._concat_df = self._concat_df[self._concat_df[self._model_dimensions_column].notna()]
        # sort dimensions alphabetically to avoid mixing and enabling string comparison
        self._concat_df[self._model_dimensions_column] = self._concat_df[self._model_dimensions_column].apply(
            lambda x: ",".join(sorted(x.split(",")))
        )
        self._concat_df[self._model_column] = \
            self._concat_df[self._model_name_column] + "_" + \
            self._concat_df["instance"].astype(str) + "[" + self._concat_df[self._model_dimensions_column] + "]"

        self._concat_df.drop(columns=[self._model_dimensions_column],inplace=True)