# BayesCL Analysis

This contains analysis scripts and notebooks for processing and visualizing
the results from BayesCL experiments.

* `tables.ipynb`: Converts experiment results into LaTeX tables and performs statistical tests.
    * `tables/`: Contains generated LaTeX tables.
* `plots.ipynb`: Generates plots from experiment results.
    * `plots/`: Contains generated plots in PNG and PDF formats.
* `final_dataframes/`: Contains final data frames in CSV and Parquet formats for analysis.
    * `summary.parquet`: Summary statistics for each experiment. Each row looks like:
    
        ```python
        @dataclass
        class SummaryRecord:
            dataset: str
            method: str
            run_id: str
            n_tasks: int
            accuracy_all_avg: float
            accuracy_final: float
            accuracy_seen_avg: float
            ace_all_avg: float
            ace_final: float
            ace_seen_avg: float
            backward_transfer: float
            brier_all_avg: float
            brier_final: float
            ece_final: float
            ece_seen_avg: float
            ece_all_avg: float
            forward_transfer: float
            sce_all_avg: float
            sce_final: float
            sce_seen_avg: float
            asce_final: float
        ```
    * `time_series.parquet`: Time series data for each experiment. Each row looks like:
        ```python
        @dataclass
        class TimeSeriesRecord:
            dataset: str
            method: str
            run_id: str
            task: int
            accuracy_all: float
            accuracy_seen: float
            ace_all: float
            ace_seen: float
            brier_all: float
            ece_all: float
            ece_seen: float
            sce_all: float
            sce_seen: float
        ```
    * `calibration.parquet`: Calibration data for each experiment. Each row looks like:
        ```python
        @dataclass
        class CalibrationRecord:
            dataset: str
            method: str
            run_id: str
            bin: int
            bin_probability: float
            bin_frequency: float
            bin_weight: float
        ```

* `hpsearch_dataframes/`: Contains data frames from hyperparameter searches in CSV and Parquet formats.
    * `values_0` is accuracy (on previous tasks averaged over each task)
    * `values_1` is ECE (on previous tasks averaged over each task)
* `update_dataframes.py`: Utility script to convert an archive of run results into data frames for analysis.
    * We have included it for reference, but it is not intended to be run as-is. It will need modification to work correctly for you.
