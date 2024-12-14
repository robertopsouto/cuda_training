1. comparing reductions

For your first task, the code is already written for you.  We will compare 3 of the reductions given during the presentation: the naive atomic-only reduction, the classical parallel reduction with atomic finish, and the warp shuffle reduction (with atomic finish).

compile it using the following:

module load cuda/9.1.85
nvcc -o reductions reductions.cu

The module load command selects a CUDA compiler for your use. The module load command only needs to be done once per session/login.  nvcc is the CUDA compiler invocation command.  They syntax is generally similar to gcc/g++

To run your code, we will use a very simple slurm command:

srun -N 1 -p hsw_p100 nvprof ./reductions

This will run the code with the profiling in its most basic mode, which is sufficient.  We want to compare kernel execution times.  What do you notice about kernel execution times?  Probably, you won't see much difference between the parallel reduction with atomics and the warp shuffle with atomics kernel.  Can you theorize why this may be?  Our objective with these will be to approach theoretical limits.  The theoretical limit for a typical reduction would be determined by the memory bandwidth of the GPU.  To calculate the attained memory bandwidth of this kernel, divide the total data size in bytes (use N from the code in your calculation) by the execution time (which you can get from the profiler).  How does this number compare to the memory bandwidth of the GPU you are running on?  (You could run bandwidthTest sample code to get a proxy/estimate).

Now edit the code to change N from ~8M to 163840 (=640*256)

recompile and re-run the code with profiling.  Is there a bigger percentage difference between the execution time of the reduce_a and reduce_ws kernel?  Why might this be?

bonus: edit the code to change N from ~8M to ~32M.  recompile and run.  What happened? Why?

2. create a different reduction (besides sum)

For this exercise, you are given a fully-functional sum-reduction code, similar to the code used for exercise 1 above, except that we will use the 2-stage reduction method without atomic finish. If you wish you can compile and run it as-is to see how it works.  Your task is to modify it (*only the kernel*) so that it creates a proper max-finding reduction.  That means that the kernel should report the maximum value in the data set, rather than the sum of the data set.  You are expected to use a similar parallel-sweep-reduction technique.  If you need help, refer to the solution.

nvcc -o max_reduction max_reduction.cu
srun -N 1 -p hsw_p100 ./max_reduction

3. revisit row_sums from hw2

For this exercise, start with the matrix_sums.cu code from hw2.  As you may recall, the row_sums kernel was reading the same data set as the column_sums kernel, but running noticeably slower.  We now have some ideas how to fix it.  See if you can implement a reduction-per-row, to allow the row-sum kernel to approach the performance of the column sum kernel.  There are probably several ways to tackle this problem.  To see one approach, refer to the solution.

You can start just by compiling the code as-is, and running the profiler to remind yourself of the performance (discrepancy).

nvcc -o matrix_sums matrix_sums.cu
srun -N 1 -p hsw_p100 nvprof ./matrix_sums

Remember from the previous session our top 2 CUDA optimization priorities: lots of threads, and efficient use of the memory subsystem.  The original row_sums kernel definitely misses the mark for the memory objective.  What we've learned about reductions should guide you.  There are probably several ways to tackle this:

 - write a straightforward parallel reduction, run it on a row, and use a for-loop to loop the kernel over all rows
 - assign a warp to each row, to perform everything in one kernel call
 - assign a threadblock to each row, to perform everything in one kernel call
 - ??

Since the (given) solution may be somewhat unusual, I'll give some hints here if needed:

 - the chosen strategy will be to assign one block per row
 - we must modify the kernel launch to launch exactly as many blocks as we have rows
 - the kernel can be adapted from the reduction kernel (atomic is not needed here) from the reduce kernel code in exercise 1 above.
 - since we are assigning one block per row, we will cause each block to perform a block-striding loop, to traverse the row.  This is conceptually similar to a grid striding loop, except each block is striding individually, one per row.  Refresh your memory of the grid-stride loop, and see if you can work this out.
 - with the block-stride loop, you'll need to think carefully about indexing

After you have completed the work and are getting a successful result, profile the code again to see if the performance of the row_sums kernel has improved:

nvcc -o matrix_sums matrix_sums.cu
srun -N 1 -p hsw_p100 nvprof ./matrix_sums

your actual performance here (compared to the fairly efficient column_sums kernel) will probably depend quite a bit on the algorithm/method you choose.  See if you can theorize how the various choices may affect efficiency or optimality.  If you end up with a solution where the row_sums kernel actually runs faster than the column_sums kernel, see if you can theorize why this may be.  Remember the two CUDA optimization priorities, and use these to guide your thinking.


