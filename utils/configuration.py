"""Module for all configuration related classes and functions"""
import json
import pathlib
from datetime import datetime


def initialize_directories(configuration: dict):
    """
    Configures the root folder where the directory for the new run is created.
    Additional subdirectories 'samples', 'logs' and 'weights' are created in
    the run directory.

    Saves the current configuration in the run folder

    Args:
        configuration (dict): dictionary with the current configuration

    Raises:
        AssertionError: Raised if the root folder does not exists.
        RuntimeError: Raised if the run directory already exists
                      and 'resume' is False.
        RuntimeError: Raised if the creation of the subdirectories fails.
    """
    # Paths
    outputrootpath = pathlib.Path(configuration["outputroot"])
    runpath = outputrootpath.joinpath(configuration["run_name"])
    configpath = runpath.joinpath(
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_config.json"
    )
    samplepath = runpath.joinpath("samples")
    weightspath = runpath.joinpath("weights")
    logspath = runpath.joinpath("logs")

    # resume
    resume = configuration["resume"]

    if not outputrootpath.exists():
        raise AssertionError(
            f"Output root folder '{outputrootpath.absolute()}' does not exist"
        )

    # Create new run directory
    try:
        runpath.mkdir(exist_ok=resume)
    except FileExistsError as error:
        # Run directory already exists and resume is False
        raise RuntimeError(
            "'resume' is set to False and run directory "
            f"'{runpath.absolute()}' already exists."
        ) from error

    # Dump configuration annotated with the current timestamp
    # prevents overwriting on resuming
    with open(configpath.absolute(), "w", encoding="utf-8") as config_fp:
        json.dump(configuration, config_fp, indent=4)

    # Create sample, weights and logs directories
    try:
        samplepath.mkdir(exist_ok=resume)
        weightspath.mkdir(exist_ok=resume)
        logspath.mkdir(exist_ok=resume)
    except FileExistsError as error:
        raise RuntimeError(
            "Error creating run subdirectories."
        ) from error
