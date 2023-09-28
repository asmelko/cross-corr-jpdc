#!/usr/bin/env python
# coding: utf-8

import os
import sys

from shared import Benchmark

from pathlib import Path
from typing import List, Tuple, Dict

benchmark = Benchmark.load(Path.cwd().parent / "benchmarking" / "jpdc" / sys.argv[1])

os.mkdir(sys.argv[1])

for group_name, group in benchmark.groups.items():
    os.mkdir(f"{sys.argv[1]}/{group_name}")
    headers = set()
    for run in group.runs:
        header = run.name not in headers
        headers.add(run.name)
        run.data.to_csv(f"{sys.argv[1]}/{group_name}/{run.name}.csv", mode="a+", header=header)
