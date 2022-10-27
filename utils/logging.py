"""Module hosting classes and functions concerning logging"""
import logging
import json
import pathlib
import time


class MetricsLogger:
    """
    Log metrics to .jsonl file
    """

    def __init__(self, configuration: dict):
        self.metriclogpath = (
            pathlib.Path(configuration["outputroot"])
            .joinpath(configuration["run_name"])
            .joinpath("logs")
            .joinpath(configuration["metric_log_name"])
        )
        self.reinitialize = configuration["reinitialize_metric_logs"]
        if self.metriclogpath.exists():
            if self.reinitialize:
                logging.info(
                    "%s exists, deleting...", self.metriclogpath.absolute()
                )
                self.metriclogpath.unlink()

    def log(self, record=None, **kwargs):
        """
        Assumption: no newlines in the input.
        """
        if record is None:
            record = {}
        record.update(kwargs)
        record["_stamp"] = time.time()
        with open(
            self.metriclogpath.absolute(), "a", encoding="ascii"
        ) as log_fp:
            log_fp.write(json.dumps(record, ensure_ascii=True) + "\n")


class Logger:
    """
    Class for logging model parameters
    """

    def __init__(self, configuration: dict):
        self.logroot = (
            pathlib.Path(configuration["outputroot"])
            .joinpath(configuration["run_name"])
            .joinpath("logs")
        )
        self.reinitialize = configuration["reinitialize_parameter_logs"]
        self.metrics = []
        self.logstyle = configuration["logstyle"] # One of '%3.3f' or like '%3.3e'

    def reinit(self, metric: str):
        """
        Delete log for metric and create a new file

        Args:
            metric (str): metric to be reinitialized
        """
        metricpath = self.logroot.joinpath(f"{metric}.log")
        if metricpath.exists():
            if self.reinitialize:
                # Only print the removal mess
                if "sv" in metric:
                    if not any("sv" in item for item in self.metrics):
                        print("Deleting singular value logs...")
                else:
                    print(f"{metricpath.absolute()} exists, deleting...")
                metricpath.unlink()

    # Log in plaintext;
    def log(self, iteration: int, **kwargs):
        """
        Log arbitrary values given as keyword arguments
        """
        for metric, value in kwargs.items():
            if metric not in self.metrics:
                if self.reinitialize:
                    self.reinit(metric)
                self.metrics += [metric]
            with open(
                self.logroot.joinpath(f"{metric}.log").absolute(),
                "a",
                encoding="ascii",
            ) as arg_fp:
                arg_fp.write(f"{iteration}: %s\n" % (self.logstyle % value))
