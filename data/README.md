# Provided dataset

The dataset is mainly provided to allow fast testing when implementing the cross-correlation algorithms. It is not ment for benchmarking or comprehensive validation of the implementation, for which the benchmarking tool is provided.

The benchmarking tool can also generated input files and validation files. All tools in this repository read and write files with the format described in this section.

The data files follow a simple naming scheme, where the name is made up of the following parts separated by underscores:
- **ina**, short for **in**put **a**rray of matrices.
- Number of rows in each matrix, i.e. the size of the *y* axis.
- Number of columns in each matrix, i.e. the size of the *x* axis.
- Number of matrices in the file.
- Sequential ID for files with the same number of matrices of the same size.

The files are simple CSV files which on their first line contain a CSV comment with 3 numbers:
1. Number of rows in each matrix.
2. Number of columns in each matrix.
3. Number of matrices in the file.

Each matrix is stored in row major order, one row per line, with rows of all matrices concatenated into one big csv table.

Another data contained in the directory are validation files, which are precomputed results of the input matrices. The name is made up the following parts separated by underscores:

- **valid**, identifying validation file.
- Computation type, one of `one_to_one`, `one_to_many`, `n_to_mn`, `n_to_m`.
- Identification of the input files, made up of:
    1. The number of input matrix rows.
    2. The number of input matrix columns.
    3. Number of matrices in the left input file.
    4. Number of matrices in the right input file.
    5. Sequential ID of the left input file.
    6. Sequential ID of the right input file.

Again it is a csv file with the same format as the input files.