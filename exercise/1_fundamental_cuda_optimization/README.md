# Instructions for GAIA users

## Matrix row/Column sums

Your first task is to create a simple matrix row and column sum application in CUDA.  The code skeleton is already given to you in matrix_sums.cu   Edit that file, paying attention to the FIXME locations, so that the output when run is like this:

```
row sums correct!
column sums correct!
```

After editing the code, compile it using the following:

```
$ module load cuda/9.1.85`
$ nvcc -o matrix_sums matrix_sums.cu
```

The module load command selects a CUDA compiler for your use. The module load command must only be done once per session/login.  nvcc is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++

To run your code, we will use a straightforward Slurm command:

`$ srun -N 1 -p hsw_p100 ./matrix_sums`

If you need help, look at matrix_sums_solution.cu for a complete example.

## Profiling

We'll introduce something new: the profiler (in this case, nvprof).  We'll use nvprof first to time the kernel execution times, and then to gather some "metric" information that will possibly shed light on our observations.

It's necessary to complete task 1 first.  Then, launch nvprof as follows:
(you may want to make your terminal session wide enough to make the output easy to read)

`$ srun -N 1 -p hsw_p100 nvprof ./matrix_sums`

What does the output tell you?
Can you locate the lines that identify the kernel durations?
Are the kernel durations the same or different?
Would you expect them to be the same or different?

Next, launch nvprof as follows:

`$ srun -N 1 -p hsw_p100 nvprof --metrics gld_efficiency ./matrix_sums`

In this case, we have asked for the metric "gld_efficiency," which is global load efficiency.  This metric corresponds closely to the slides in the presentation that covered memory access patterns, with corresponding percentages of associated "efficiency" of utilization of the memory subsystem (reads from global memory).

Do the kernels (row_sum, column_sum) have the same or different efficiencies?
Why?

How does this correspond to the observed kernel execution times for the first profiling run?

Can we improve this?  (stay tuned for the next CUDA training session)
