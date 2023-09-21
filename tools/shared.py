import json
import logging
import os
import subprocess
import sys


def get_logger(name=None):
    if name is None:
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(levelname)s] [%(asctime)s] %(name)-16s : %(message)s ", "%Y-%m-%d %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def test_single(
    key,
    configurations,
    size="1GB",
    multithreaded=True,
    folding=2,
    expanded=True,
    run_tests=False,
    lock=None,
    results_file="results.json",
):
    file_name = os.path.join(os.path.dirname(__file__), f"{key.lower()}.py")
    file_name = os.path.abspath(file_name)
    logger = get_logger(key)
    with open(file_name, "w") as of:
        of.write(configurations[key])

    input_format = "stdin32" if "output = 32" in configurations[key] else "stdin64"
    if not run_tests:
        return

    cmd = [
        "python",
        "practrand-driver.py",
        "-if",
        file_name,
        "|",
        "RNG_test",
        input_format,
        "-tlmax",
        size,
        "-te",
        "1" if expanded else "0",
        "-tf",
        str(folding),
    ]
    if multithreaded:
        cmd += ["-multithreaded"]
    logger.info("Executing " + " ".join(cmd))

    ps = subprocess.Popen(
        " ".join(cmd),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = ps.communicate()[0]
    try:
        os.unlink(file_name)
    except Exception:
        logger.warning(f"Unable to unlink {file_name}")
    if lock is not None:
        with lock:
            if not os.path.exists(results_file):
                with open(results_file, "w") as handle:
                    json.dump({}, handle)
            with open(results_file) as handle:
                results = json.load(handle)
            if key not in results:
                results[key] = {}
            results[key][size] = output.decode("utf8")
            with open(results_file, "w") as handle:
                json.dump(results, handle, indent=4, sort_keys=True)
    if "FAIL" in output.decode("utf8"):
        logger.warning("FAIL " + " ".join(cmd))
    else:
        logger.info("Completed " + " ".join(cmd))
    return key, size, output.decode("utf8")
