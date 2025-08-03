# One-Day-Ahead Photovoltaic Power Prediction Based on

Repository contains code and dataset supporting research "One-Day-Ahead Photovoltaic Power Prediction Based on
Local Weather Properties Analysis" ECAI'2025

## Initialization
At the very beginning the python venv must be initialized and packages listed in requirements.txt needs to be installed.

```
    ~/$ python3 -m venv .venv
    ~/$ source .venv/bin/activate
    (.venv):~/$ pip install -r requirements.txt
``` 

## Dataset preparation

First of all the dataset file `datasets/orgdataset.tar.xz` need to be unpacked to `datasets/orgdataset.csv`. This file is an original dataset that contains raw data. Those data needs to be cleared and 
transform before processing. To achieve that use the command below.  
```
    (.venv):~/$python datasets/process_data.py
```
It will transform original dataset into its clear version. New file will appear inside `datasets/dataset.csv`

## Database display

To display dataset as interactive chart. Use arrows to move over the data.
```
(.venv):~/$python  examples/problem_analysis_database_looker.py
```
To create latex table that describes dataset the following script can be executed
```
(.venv):~/$python examples/db_stats.py
```

## Experiments (full version)
The following scrit crates and runs experiments that were used in the research. During tests its execution took 50hours.
For checking purposes we recommend to refer to Experiments (shorted version) section
```
   (.venv):~/$python  examples/experiments/problem_analysis_run_experiment.py
```

## Experiments (shorted version)
The following scrit crates and runs on linear regression and random forest models (a part of experiments)
```
   (.venv):~/$python  examples/experiments/problem_analysis_run_experiment_min.py
```

## Experiments results processing 
once any of experimental script finish its execution results will be stored in ```cm``` directory. Results
consists of 'concat' and 'prediction' files. The preferable way to work with results is first to move them to dedicated 
directory e.g. ```cm/all``` and after that transform them into more useful form
use the following script. 

```
   (.venv):~/$python  examples/experiments/problem_analysis_results.py cm/all cm/sa.csv
```

The executed script creates file ```cm/sa.csv``` that stores model comparision table with calculated
improvement ratios and Welch's tests.

## Experiments result tables generation
Most of the tables used in article were generated automatically based on ```cm/sa.csv``` file. script   
```
   (.venv):~/$python  examples/experiments/problem_analysis_results_table.py cm/all cm/sa.csv
```
was used for this purpose. 

## Extras

For more information about used ANN architectures refer to `ANN/model_wrappers.py`. However, the use of the
architectures proposed was limited by the hardware possibilities.

