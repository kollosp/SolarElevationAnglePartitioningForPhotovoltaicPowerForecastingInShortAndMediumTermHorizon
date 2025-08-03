import pandas as pd
class DataframeChainProcessor:
    """
    Class organizes a series of dataframe transformations. It displays each process step and ensures that dataframe is
    always copied and it can be reused at any moment
    """
    def __init__(self):
        self.steps = {}

    def make_step(self, step_name: str, df, func):
        df_new = func(df)
        if step_name[0] != "_":
            print(step_name)
            if hasattr(df_new,"columns"):
                print("Columns:", df_new.columns.tolist())
            if hasattr(df_new,"shape"):
                print("Shape:", df_new.shape)
            print(df_new)
            print("============")
        self.steps[step_name] = df_new
        return df_new

    def transform_columns_to_multi_index(self, df):
        df_new = df.copy(deep=True)
        df_new.columns = pd.MultiIndex.from_tuples([c.split("_")[::-1] for c in df.columns.to_list()],
                                                   names=["Attribute", "Dataset"])
        return df_new

    def flatten_index(self, df, name=""):
        df_new = df.copy(deep=True)
        df_new.columns = pd.Index(["_".join(c) for c in df_new.columns.to_list()], name=name)
        return df_new

    def drop_timestamp_column_time(self, df, column: str = 'date'):
        df_new = df.copy(deep=True)
        df_new[column] = pd.to_datetime(df[column]).dt.date
        return df_new

    def add_column(self, df, column, func):
        df_new = df.copy(deep=True)
        df_new[column] = func(df)
        return df_new