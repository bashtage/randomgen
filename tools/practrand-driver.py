#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example usage:

python practrand-driver.py --jumped -bg ThreeFry \
       -n 8192 | ./RNG_test stdin64 -tlmax 512GB

It is recommended to use the patched version that increases the buffer size,
e.g., practrand-0.93-bigbuffer.patch

Modified from https://gist.github.com/rkern/6cf67aee7ee4d87e1d868517ba44739c/

"""
import json
import logging
import sys

import numpy as np

import randomgen as rg

CONFIG = {
    rg.PCG32: {"output": 32, "seed": 64, "seed_size": 64},
    rg.PCG64: {"output": 64, "seed": 128, "seed_size": 128},
    rg.ThreeFry: {"output": 64, "seed": 256, "seed_size": 64},
    rg.Xoshiro256: {"output": 64, "seed": 256, "seed_size": 64},
    rg.Philox: {"output": 64, "seed": 256, "seed_size": 64},
    rg.SFMT: {"output": 64, "seed": 128, "seed_size": 32},
    rg.LXM: {"output": 64, "seed": 128, "seed_size": 32},
}


def gen_interleaved_bytes(bitgens, n_per_gen=1024, output=32):
    astype = np.uint32 if output == 32 else np.uint64
    view = np.uint64
    while True:
        draws = [g.random_raw(n_per_gen).astype(astype).view(view) for g in bitgens]
        interleaved = np.column_stack(draws).ravel()
        bytes_chunk = bytes(interleaved.data)
        yield bytes_chunk


def bitgen_from_state(state):
    cls = getattr(rg, state["bit_generator"])
    bitgen = cls()
    bitgen.state = state
    return bitgen


def jumped_state(bit_generator, n_streams=2, entropy=None):
    bitgen = getattr(rg, bit_generator)
    config = CONFIG[bitgen]
    seed = config["seed"]
    if entropy is None:
        entropy = rg.random_entropy(seed // 32)
        if config["seed_size"] == 64:
            entropy = entropy.view(np.uint64)
            if config["seed"] == 64:
                entropy = entropy[0]
        elif config["seed_size"] == 128:
            entropy = int(entropy[0]) + int(entropy[1]) * 2 ** 64
        elif config["seed_size"] == 256:
            base = int(0)
            for i in range(4):
                base += int(entropy[i]) * (2 ** (64 * i))
            entropy = base
        elif config["seed_size"] != 32:
            raise NotImplementedError
    else:
        seed_size = config["seed_size"]
        if seed_size in (32, 64):
            _entropy = []
            while entropy > 0:
                low = entropy % 2 ** seed_size
                _entropy.append(low)
                entropy = entropy >> seed_size
            dtype = np.uint32 if seed_size == 32 else np.uint64
            entropy = np.array(_entropy, dtype=dtype)
        elif seed_size in (128, 256):
            entropy = entropy % 2 ** seed_size
        else:
            raise NotImplementedError

    bg = bitgen(entropy)
    bitgens = [bg]
    for i in range(n_streams - 1):
        bg = bg.jumped()
        bitgens.append(bg)
    return bitgens


def dump_states(bitgens, file=sys.stderr, disp=False):
    def array_to_list(d):
        for key in d:
            if isinstance(d[key], dict):
                d[key] = array_to_list(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = d[key].tolist()
        return d

    text = json.dumps([array_to_list(g.state) for g in bitgens], indent=4)
    if disp:
        print(text, file=file)
    return text


def from_json(filename):
    with open(filename) as f:
        states = json.load(f)

    def list_to_array(d):
        for key in d:
            if isinstance(d[key], dict):
                d[key] = list_to_array(d[key])
            elif isinstance(d[key], list):
                d[key] = np.array(d[key])
        return d

    bitgens = [bitgen_from_state(list_to_array(s)) for s in states]
    return bitgens


def main():
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-bg", "--bit_generator", type=str, default="PCG64", help="BitGenerator to use."
    )
    parser.add_argument(
        "--load", action="store_true", help="Load BitGenerators from JSON file."
    )
    parser.add_argument(
        "--save", action="store_true", help="Save BitGenerators to JSON file."
    )
    parser.add_argument(
        "-j", "--jumped", action="store_true", help="Use jumped() to get new streams."
    )
    parser.add_argument("-s", "--seed", type=int, help="Set a single seed")
    parser.add_argument(
        "-n",
        "--n-streams",
        type=int,
        default=2,
        help="The number of streams to interleave",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default="",
        help="JSON filename. Does not save if empty",
    )

    args = parser.parse_args()
    filename = args.filename
    if not filename:
        filename = args.bit_generator.lower() + ".json"
        logging.log(logging.INFO, "Default filename is " + filename)
    if args.load:
        logging.log(logging.INFO, "Loading bit generator config from " + filename)
        bitgens = from_json(filename)
    elif args.jumped:
        msg = "Creating {n} bit generators of {bg_type}".format(
            n=args.n_streams, bg_type=args.bit_generator
        )
        logging.log(logging.INFO, msg)
        bitgens = jumped_state(
            args.bit_generator, n_streams=args.n_streams, entropy=args.seed
        )
    else:
        logging.log(logging.WARN, "You are only testing a single stream")
        bitgens = jumped_state(args.bit_generator, n_streams=1, entropy=args.seed)

    if args.save:
        logging.log(logging.INFO, "Saving bit generator config to " + filename)
        dumped = dump_states(bitgens, disp=False)
        with open(filename, "w") as out:
            out.write(dumped)

    output = CONFIG[bitgens[0].__class__]["output"]
    logging.log(logging.INFO, "Output bit size is {0}".format(output))
    for chunk in gen_interleaved_bytes(bitgens, output=output):
        sys.stdout.buffer.write(chunk)


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, IOError):
        logging.log(logging.INFO, "Pipe broken, assuming complete")

    sys.stderr.close()
