import argparse
from pathlib import Path
from enum import Enum

from typing import Tuple

import numpy as np


class OutputFormats(Enum):
    CSV = "csv"


DEFAULT_OUTPUT_FORMAT = OutputFormats.CSV
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "data.csv"


def save(num_matrices: int, values: np.ndarray, format: OutputFormats, output_path: Path):
    if format == OutputFormats.CSV:
        print(f"Saving data to {str(output_path)}")
        with output_path.open(mode='w') as f:
            np.savetxt(f, values, delimiter=',', header=f'{int(values.shape[0] / num_matrices)},{values.shape[1]},{num_matrices}')


def generate(rows: int, columns: int):
    rng = np.random.default_rng()
    print("Generating random matrix")
    return rng.random(size=(rows, columns))


def _generate_matrix(args: argparse.Namespace):
    values = generate(args.num_matrices * args.rows, args.columns)
    save(args.num_matrices, values, args.format, args.output_path.absolute())


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
                                help=f"Output file format (defaults to {DEFAULT_OUTPUT_FORMAT})")
    input_generate.add_argument("rows",
                                type=int,
                                help=f"Number of rows of the generated matrix")
    input_generate.add_argument("columns",
                                type=int,
                                help=f"Number of columns of the generated matrix")
    input_generate.add_argument("num_matrices",
                                nargs="?",
                                type=int,
                                default=1,
                                help=f"Number of matrices of given size to generate (default {1})")
    input_generate.set_defaults(action=_generate_matrix)


def main():
    parser = argparse.ArgumentParser(description="Tool for generating and transforming input data.")
    add_arguments(parser)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
