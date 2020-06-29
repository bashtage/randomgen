"""
Experiment to examine quality when using a poor choice of
seed values: 1, 2, 4, 8, 16, ... so that each has a single
non-zero bit.
"""

from collections import defaultdict
import itertools
import json
from multiprocessing import Manager
import os

import jinja2

from configuration import ALL_BIT_GENS, DSFMT_WRAPPER, OUTPUT, SPECIALS
from joblib import Parallel, cpu_count, delayed
from randomgen import DSFMT
from shared import get_logger, test_single

with open("seed-correlation.jinja") as tmpl:
    TEMPLATE = jinja2.Template(tmpl.read())


def setup_configuration_files():
    streams = {}
    parameters = {}
    for bitgen in ALL_BIT_GENS:
        key = bit_generator = bitgen.__name__
        if bitgen in SPECIALS:
            config = SPECIALS[bitgen]
            args = [value for value in config.values()]
            for arg_set in itertools.product(*args):
                kwargs = {key: arg for key, arg in zip(config.keys(), arg_set)}
                key = "-".join(
                    [bit_generator]
                    + [f"{key}-{value}" for key, value in kwargs.items()]
                )
                parameters[key] = (bitgen, kwargs)
        else:
            parameters[key] = (bitgen, {})
    for key in parameters:
        extra_initializiation = ""
        bitgen, kwargs = parameters[key]
        bit_generator = bitgen.__name__
        kwargs_repr = repr(kwargs)
        if bitgen == DSFMT:
            extra_initializiation = DSFMT_WRAPPER
        streams[key] = TEMPLATE.render(
            streams=8,
            kwargs=kwargs_repr,
            bit_generator=bit_generator,
            output=OUTPUT[bitgen],
            extra_initializiation=extra_initializiation,
        )
    return streams


if __name__ == "__main__":
    logger = get_logger("prng-tester")

    import argparse

    parser = argparse.ArgumentParser(
        description="Test alternative with bad seed values",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-mt",
        "--multithreaded",
        action="store_true",
        help="Pass the --multithreaded flag in PractRand's RNG_Test.",
    )
    parser.add_argument(
        "-rt",
        "--run-tests",
        action="store_true",
        help="Run the tests. If False, only the test configuration files are output.",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=str,
        default="1GB",
        help="Set the size of data ot test using PractRand",
    )
    parser.add_argument(
        "-n",
        "--n-jobs",
        type=int,
        help="The number of jobs to simultaneously run when using --parallel",
    )
    parser.add_argument(
        "-f",
        "--folding",
        type=int,
        default=1,
        help="The number of folds to use: 0, 1 or 2.",
    )
    parser.add_argument(
        "-ex", "--expanded", action="store_true", help="Use the expanded test suite",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-run even if result exists",
    )

    args = parser.parse_args()

    assert args.folding in (0, 1, 2)

    results_file = "results-seed-correlation.json"
    configurations = setup_configuration_files()

    results = defaultdict(dict)
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf8") as existing:
            results.update(json.load(existing))
    manager = Manager()
    lock = manager.Lock()
    configuration_keys = list(configurations.keys())
    final_configuration_keys = []
    for key in configuration_keys:
        if key in results and args.size in results[key] and not args.force:
            logger.info(f"Skipping {key} with size {args.size}")
            continue
        final_configuration_keys.append(key)
    configuration_keys = final_configuration_keys
    test_args = []
    for key in configuration_keys:
        test_args.append(
            [
                key,
                configurations,
                args.size,
                args.multithreaded,
                args.folding,
                args.expanded,
                args.run_tests,
                lock,
                results_file,
            ]
        )
    cpu_per_job = 2 + args.multithreaded + args.expanded + (args.folding - 1)
    n_jobs = args.n_jobs if args.n_jobs else (cpu_count() - 1) // cpu_per_job
    n_jobs = max(n_jobs, 1)
    logger.info(f"Running in parallel with {n_jobs}.")
    logger.info(f"{len(test_args)} configurations to test")
    parallel_results = Parallel(n_jobs, verbose=50)(
        delayed(test_single)(*ta) for ta in test_args
    )
