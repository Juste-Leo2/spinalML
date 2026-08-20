import os
import random

from utils.math_metrics import clear_math_log


def pytest_addoption(parser):
    parser.addoption(
        "--debug-math", 
        action="store_true", 
        default=False, 
        help="Generate true_math_errors.log with detailed precision errors"
    )


def pytest_sessionstart(session):
    seed = int(os.environ.get("SPINALML_SEED", "42"))
    os.environ["SPINALML_SEED"] = str(seed)
    os.environ["RANDOM_SEED"] = str(seed)
    random.seed(seed)
    if session.config.getoption("--debug-math"):
        os.environ["DEBUG_MATH"] = "1"
        clear_math_log()