# Instructions for GAIA users

## 1 - Matrix Row/Column Sums

Your first task is to create a simple matrix row and column sum application in CUDA. The code skeleton is already given to you in `matrix_sums.cu`. Edit that file, paying attention to the `FIXME` locations so that the output runs such as:

```
row sums correct!
column sums correct!
```

After editing the code, compile it using the following:

```
$ module load cuda/12.0
$ nvcc -o matrix_sums matrix_sums.cu
```

The module load command selects a CUDA compiler for your use. The module load command must only be done once per session/login. `nvcc` is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++.

To run your code, we will use a straightforward Slurm command:

```
$ srun -p SEDE -A curso --gres=gpu:1 ./matrix_sums
```

## 2 - Profiling

We will introduce a new tool—the `nsys profile` profiler. First, we will use `nsys profile` to measure kernel execution times and then gather additional metrics to help interpret our observations.

Before proceeding, make sure you have completed Task 1. Then, use the following command to launch `nsys profile`: (*You may wish to expand your terminal window to more easily read the output*)

```
srun --reservation=curso --gres=gpu:1 nsys profile --stats=true -t cuda --cuda-memory-usage=true matrix_sums
```

Review the generated output:

- Which lines report the kernel execution durations?
- Do the reported kernel durations appear to be the same or different?
- Given your understanding, would you expect the durations to be identical or vary?

# Instructions for COLAB users

https://colab.research.google.com/drive/10seerH4tF6KUoTvvPvMI8xXamWaxl3y0?usp=sharing
