"""
The main test program for quality assurance in randomgen
"""

from collections import defaultdict
import itertools
import json
from multiprocessing import Manager
import os
import random
import time

from configuration import (
    ALL_BIT_GENS,
    DEFAULT_ENTOPY,
    DSFMT_WRAPPER,
    JUMPABLE,
    OUTPUT,
    SPECIALS,
    TEMPLATE,
)
from joblib import Parallel, cpu_count, delayed
from shared import get_logger, test_single

from randomgen import DSFMT, SFC64

DEFAULT_STREAMS = (4, 8196)


def configure_stream(
    bit_gen, kwargs=None, jumped=False, streams=8196, entropy=DEFAULT_ENTOPY
):
    bit_generator = bit_gen.__name__
    extra_code = extra_initialization = ""
    if bit_gen == SFC64 and kwargs["k"] == "weyl":
        extra_code = f"""\
base = rg.SFC64(seed_seq)
weyl = base.weyl_increments({streams})
bitgens = [rg.SFC64(seed_seq, k=k) for k in retain]
        """
    elif bit_gen == DSFMT:
        bit_generator = "Wrapper32"
        extra_initialization = DSFMT_WRAPPER
        # return configure_dsfmt(streams, entropy=entropy)
    kwargs = {} if kwargs is None else kwargs
    kwargs_repr = str(kwargs)
    return TEMPLATE.render(
        bit_generator=bit_generator,
        entropy=entropy,
        jumped=jumped,
        streams=streams,
        kwargs=kwargs_repr,
        output=OUTPUT[bit_gen],
        extra_initialization=extra_initialization,
        extra_code=extra_code,
    )


def setup_configuration_files(
    entropy=DEFAULT_ENTOPY, skip_single=False, num_streams=DEFAULT_STREAMS
):
    streams = {}
    for bitgen in ALL_BIT_GENS:
        name = bitgen.__name__
        if bitgen not in SPECIALS:
            if not skip_single:
                streams[name] = configure_stream(bitgen, entropy=entropy)
            for num_stream in num_streams:
                key = name + f"-streams-{num_stream}"
                streams[key] = configure_stream(
                    bitgen, streams=num_stream, entropy=entropy
                )
            if bitgen not in JUMPABLE:
                continue
            for num_stream in num_streams:
                key = name + f"-jumped-streams-{num_stream}"
                streams[key] = configure_stream(
                    bitgen, streams=num_stream, jumped=True, entropy=entropy
                )
        else:
            config = SPECIALS[bitgen]
            args = [value for value in config.values()]
            for arg_set in itertools.product(*args):
                kwargs = {key: arg for key, arg in zip(config.keys(), arg_set)}
                key = "-".join(
                    [name] + [f"{key}-{value}" for key, value in kwargs.items()]
                )
                if not skip_single:
                    streams[key] = configure_stream(
                        bitgen, kwargs=kwargs, entropy=entropy
                    )
                for num_stream in num_streams:
                    full_key = key + f"-streams-{num_stream}"
                    streams[full_key] = configure_stream(
                        bitgen, kwargs=kwargs, streams=num_stream, entropy=entropy
                    )
                if bitgen not in JUMPABLE:
                    continue
                for num_stream in num_streams:
                    full_key = key + f"-jumped-streams-{num_stream}"
                    streams[full_key] = configure_stream(
                        bitgen,
                        kwargs=kwargs,
                        streams=num_stream,
                        jumped=True,
                        entropy=entropy,
                    )
    return {k: streams[k] for k in sorted(streams.keys())}


if __name__ == "__main__":
    logger = get_logger("prng-tester")

    import argparse

    parser = argparse.ArgumentParser(
        description="Test alternative configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-mt",
        "--multithreaded",
        action="store_true",
        help="Pass the --multithreaded flag in PractRand's RNG_Test.",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Use multuprocessing to run the test in parallel.",
    )
    parser.add_argument(
        "-rt",
        "--run-tests",
        action="store_true",
        help="Run the tests. If False, only the test configuration files are output.",
    )
    parser.add_argument(
        "-r",
        "--randomize",
        action="store_true",
        help="Execute in a random order by shuffling.",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=str,
        default="1GB",
        help="Set the size of data ot test using PractRand",
    )
    parser.add_argument(
        "-e",
        "--entropy",
        type=int,
        default=DEFAULT_ENTOPY,
        help="Set the global entropy used in the base SeedSequence used in all of "
        "the test runs",
    )
    parser.add_argument(
        "-n",
        "--n-jobs",
        type=int,
        help="The number of jobs to simultaneously run when using --parallel",
    )
    parser.add_argument(
        "-mj",
        "--max-jobs",
        type=int,
        help="The maximum number of jobs to execute before exiting",
    )
    parser.add_argument(
        "-f",
        "--folding",
        type=int,
        default=1,
        help="The number of folds to use: 0, 1 or 2.",
    )
    parser.add_argument(
        "-ex",
        "--expanded",
        action="store_true",
        help="Use the expanded test suite",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if result exists",
    )
    parser.add_argument(
        "-ss",
        "--skip-single",
        action="store_true",
        help="Skip single bit-generator configuration.",
    )
    default_streams = ",".join([str(s) for s in DEFAULT_STREAMS])
    parser.add_argument(
        "--streams",
        default=default_streams,
        help="Comma separated list of streams to test, e.g, 4,8,16,32,64.",
    )
    parser.add_argument(
        "-rf",
        "--results-file",
        default="results.json",
        help="Name of the file in which to store results",
    )

    args = parser.parse_args()

    assert args.folding in (0, 1, 2)
    streams = [int(s) for s in args.streams.split(",")]

    configurations = setup_configuration_files(
        entropy=args.entropy, skip_single=args.skip_single, num_streams=streams
    )
    if args.run_tests:
        print("Running tests...")

        time.sleep(0.5)

    results = defaultdict(dict)
    results_file = args.results_file
    if os.path.exists(results_file):
        with open(results_file, encoding="utf8") as existing:
            results.update(json.load(existing))
    manager = Manager()
    lock = manager.Lock()
    configuration_keys = list(configurations.keys())
    if args.randomize:
        random.shuffle(configuration_keys)
        logger.info("Randomizing the execution order")
    final_configuration_keys = []
    for key in configuration_keys:
        if key in results and args.size in results[key] and not args.force:
            logger.info(f"Skipping {key} with size {args.size}")
            continue
        final_configuration_keys.append(key)
    configuration_keys = final_configuration_keys
    if args.max_jobs:
        configuration_keys = configuration_keys[: args.max_jobs]
    if args.parallel:
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
    else:
        logger.info("Running in series")
        for key in configuration_keys:
            result = test_single(
                key,
                configurations,
                size=args.size,
                multithreaded=args.multithreaded,
                run_tests=args.run_tests,
                lock=lock,
            )
