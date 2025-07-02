import ultraimport, os
from datetime import datetime

ultraimport('__dir__/../helpers/MTimeSeries.py', 'MTimeSeries', globals=globals())
ultraimport('__dir__/../helpers/TimeSeries.py', 'TimeSeries', globals=globals())
ultraimport('__dir__/../helpers/TimeSeries.py', 'TimeSeriesSamplingConverter', globals=globals())
ultraimport('__dir__/../helpers/StandardStorage.py', 'StandardStorage', globals=globals())
ultraimport('__dir__/../helpers/StandardStorage.py', 'read_csv_database_file', globals=globals())
ultraimport('__dir__/../helpers/ConfigIterator.py', 'ConfigIterator', globals=globals())
ultraimport('__dir__/../helpers/TimeSeriesExamination.py', 'TimeSeriesExamination', globals=globals())
ultraimport('__dir__/../helpers/TimeSeriesExamination.py', 'MetricMAE', globals=globals())
ultraimport('__dir__/../helpers/TimeSeriesExamination.py', 'MetricMAPE', globals=globals())
ultraimport('__dir__/../helpers/TimeSeriesExamination.py', 'MetricRMSE', globals=globals())
ultraimport('__dir__/../helpers/TimeSeriesExamination.py', 'MetricRRMSE', globals=globals())
ultraimport('__dir__/../helpers/TimeSeriesExamination.py', 'MetricIntegralError', globals=globals())
ultraimport('__dir__/../helpers/TimeSeriesExamination.py', 'MetricMBE', globals=globals())
ultraimport('__dir__/../helpers/TimeSeriesExamination.py', 'MetricRMBE', globals=globals())
ultraimport('__dir__/../helpers/MetricFilters.py', 'MetricDay', globals=globals())

class ExaminationModule:
    def __init__(self):
        pass

    @staticmethod
    def predict(model_creation_functions=[], verbose=0, skip=0,
                output_directory=None, tssc=None, create_tse=None, create_metrics=None, **kwargs):
        file_path = None
        if output_directory is not None:
            if not os.path.isdir(output_directory):
                os.mkdir(output_directory)

            dt = datetime.now()
            dt = dt.strftime("%Y%m%d%H%M%S")

            file_path = os.path.join(output_directory, dt)

        if tssc is None:
            tssc = TimeSeriesSamplingConverter()

        statistics_table = None
        combined_mts = MTimeSeries()
        combined_errors = MTimeSeries()
        for i, (c, indexes, cc) in enumerate(ConfigIterator(args=kwargs, verbose=verbose, skip=skip)):
            #kwargs should contain field like: pv_prod = {{pv_prod: ..., latitude_degrees: ..., exogenous_mts: ...}}
            pv_prod =cc["pv_prod"] # read X from kwargs
            exogenous_mts = None
            if "exogenous_mts" in cc:
                exogenous_mts = cc["exogenous_mts"] # read exogenous for X from kwargs
            # latitude_degrees = cc["latitude_degrees"]

            # create all specified models and pass kwargs into them.
            tse = None
            # create using passed method or use default one
            if create_tse is None:
                tse = ExaminationModule.create_tse(pv_prod, tssc, verbose, exogenous_mts=exogenous_mts)
            else:
                tse = create_tse(pv_prod, tssc, verbose, exogenous_mts=exogenous_mts)

            # create using passed method or use default one
            if create_metrics is None:
                tse = ExaminationModule.create_metrics(tse, tssc, **cc)
            else:
                tse = create_metrics(tse, tssc, **cc)

            for model_creation_function in model_creation_functions:
                tse = model_creation_function(tse, tssc, **cc)

            # make test
            mts, errors = tse.train_test_sample_ahead()

            # store data
            # copy by database name (same name will override and avoid duplications)
            if not pv_prod.name in combined_mts.names:
                combined_mts[pv_prod.name] = pv_prod

            for j, model in enumerate(tse.models):
                # model names according to table file.
                combined_mts[f"M{i * len(tse.models) + j}"] = mts[f"M{j}"]

            statistics = tse.statistics()
            if statistics_table is None:
                statistics_table = statistics
            else:
                statistics_table.append_from_other_table_printer(statistics)

            if i % 2 == 0 and i > 0:
                print(f"Runtime store: Results stored in {output_directory} as {file_path}")
                format = statistics_table.format
                statistics_table.format = "csv"
                statistics_table.write_file(file_path + ".csv")
                statistics_table.format = format
                # combined_mts.save_excel_format(file_path + ".xlsx")
            print(f"Statistics after {i}-th configuration")
            print(statistics_table)

        if output_directory is not None:
            # save to output directory
            statistics_table.format = "csv"
            statistics_table.write_file(file_path + ".csv")
            # combined_mts.save_excel_format(file_path + ".xlsx")
            print(f"Results stored in {output_directory} as {file_path}")
            #combined_mts.plot()

        return combined_mts # return results mts

    @staticmethod
    def create_tse(pv_prod : TimeSeries,
                   tssc : TimeSeriesSamplingConverter,
                   verbose : int,
                   exogenous_mts : MTimeSeries=None):
        """
            Factory function that creates TimeSeries Examination object by taking timeseries (X)
             and exogenous (Additional observations),
        """
        tse = TimeSeriesExamination(directory="/tmp/PV/", verbose=verbose)
        tse.register_x_timeseries(pv_prod, tssc, exogenous_mts=exogenous_mts)

        return tse

    @staticmethod
    def create_metrics(tse, tssc, **kwargs):
        tse.register_metric(MetricMAE())
        tse.register_metric(MetricRMSE())

        tse.register_metric(MetricMBE())
        tse.register_metric(MetricRMBE())
        tse.register_metric(MetricRRMSE())
        tse.register_metric(MetricIntegralError())
        tse.register_metric(MetricDay(kwargs["latitude_degrees"], MetricMAPE()))
        tse.register_metric(MetricDay(kwargs["latitude_degrees"], MetricMAE()))
        tse.register_metric(MetricDay(kwargs["latitude_degrees"], MetricRMSE()))
        tse.register_metric(MetricDay(kwargs["latitude_degrees"], MetricMBE()))
        tse.register_metric(MetricDay(kwargs["latitude_degrees"], MetricRMBE()))
        tse.register_metric(MetricDay(kwargs["latitude_degrees"], MetricRRMSE()))
        tse.register_metric(MetricDay(kwargs["latitude_degrees"], MetricIntegralError()))

        return tse