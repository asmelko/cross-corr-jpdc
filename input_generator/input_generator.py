import argparse
from pathlib import Path
from enum import Enum

from typing import Tuple

import numpy as np


class OutputFormats(Enum):
    CSV = "csv"


DEFAULT_OUTPUT_FORMAT = OutputFormats.CSV
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "data.csv"


def generate(size: Tuple[int, int], format: OutputFormats, output_path: Path):
    rng = np.random.default_rng()
    # vals = rng.integers(0, 256, size=size)
    print("Generating random matrix")
    vals = rng.random(size=size)

    if format == OutputFormats.CSV:
        print(f"Saving data to {str(output_path)}")
        with output_path.open(mode='w') as f:
            f.write(f'{size[0]},{size[1]}\n')
            np.savetxt(f, vals, delimiter=',')


def _generate(args: argparse.Namespace):
    generate(tuple(args.size), args.format, args.output_path.absolute())


def add_arguments(parser: argparse.ArgumentParser):
    input_subparsers = parser.add_subparsers(required=True, dest="input",
                                             description="Generating and transforming input")

    input_generate = input_subparsers.add_parser("generate", help="Generate input matrix")
    input_generate.add_argument("-o", "--output_path",
                                # TODO: Add different extensions based on output format
                                default=DEFAULT_OUTPUT_PATH,
                                type=Path,
                                help=f"Output file path (defaults to {str(DEFAULT_OUTPUT_PATH)})")
    input_generate.add_argument("-f", "--format",
                                type=OutputFormats,
                                choices=list(OutputFormats),
                                default=DEFAULT_OUTPUT_FORMAT,
                                help=f"Output file format (defaults to {DEFAULT_OUTPUT_PATH})")
    input_generate.add_argument("size",
                                nargs=2,
                                type=int,
                                help=f"Size of the generated matrix")
    input_generate.set_defaults(action=_generate)


def main():
    parser = argparse.ArgumentParser(description="Tool for generating and transforming input data.")
    add_arguments(parser)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
