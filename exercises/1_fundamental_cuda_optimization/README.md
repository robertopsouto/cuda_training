# Instructions for GAIA users

## Matrix row/Column sums

Your first task is to create a simple matrix row and column sum application in CUDA. The code skeleton is already given to you in `matrix_sums.cu`. Edit that file, paying attention to the FIXME locations so that the output runs such as:

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
$ srun -p gpu -A <SEU_GRUPO> --gres=gpu:1 ./matrix_sums
```

# Instructions for COLAB users

https://colab.research.google.com/drive/10seerH4tF6KUoTvvPvMI8xXamWaxl3y0?usp=sharing
