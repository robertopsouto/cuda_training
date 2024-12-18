# GAIA Users
## 1 - Comparing Reductions

For your first Assignment, the code has already been written for you. We will compare 3 of the reductions given during the presentation:
- The naive atomic-only reduction
- The classical parallel reduction with atomic finish
- The warp shuffle reduction (with atomic finish)

Compile it using the following:

```
$ module load cuda/12.0
$ nvcc -o reductions reductions.cu
```

The `module load` command selects a CUDA compiler for your use. The `module load` command must only be done once per session/login. `nvcc` is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++.

To run your code, use the following Slurm command:

```
$ srun --reservation=curso --gres=gpu:1 nsys profile --stats=true -t cuda --cuda-memory-usage=true reductions
```

This will run the code with the profiling in its most basic mode, which is sufficient. We want to compare kernel execution times. What do you notice about kernel execution times?  You will see little difference between the parallel reduction with atomics and the warp shuffle with atomics kernel. Can you theorize why this may be? Our objective with these will be to approach theoretical limits. 

The memory bandwidth of the GPU would determine the theoretical limit for a typical reduction. To calculate the attained memory bandwidth of this kernel, divide the total data size in bytes (use `N` from the code in your calculation) by the execution time (which you can get from the profiler). How does this number compare to the memory bandwidth of the GPU you are running on? (You could run `bandwidthTest` sample code to get a proxy/estimate).

Now edit the code to change `N` from `~8M` to `163840 (=640*256)`.

Could you recompile and re-run the code with profiling? Is there a significant percentage difference between the execution time of the `reduce_a` and `reduce_ws` kernel?  Why might this be?

**Bonus**: edit the code to change `N` from `~8M` to `~32M`. Could you recompile and run? What happened? Why?

## 2 - Create a Different Reduction (Besides Sum)

For this Assignment, you are given a fully functional sum-reduction code, similar to the code used for exercise 1 above, except that we will use the 2-stage reduction method without an atomic finish. You can compile and run it as-is to see how it works. Your task is to modify it (*only the kernel*) so that it creates a proper max-finding reduction. That means the kernel should report the maximum value in the data set rather than the sum of the data set. You are expected to use a similar parallel-sweep-reduction technique.

```
$ nvcc -o max_reduction max_reduction.cu
$ srun --reservation=curso --gres=gpu:1 ./max_reduction
```

## 3 - Revisit `row_sums` from *Lecture 1*

For this Assignment, start with the `matrix_sums.cu` code from *Lecture 1*.  As you may recall, the `row_sums` kernel read the same data set as the `column_sums` kernel but ran noticeably slower. We now have some ideas on how to fix it. See if you can implement a reduction-per-row to allow the row-sum kernel to approach the performance of the column-sum kernel. There are several ways to tackle this problem. To see one approach, please take a look at the solution.

You can start by compiling the code as-is and running the profiler to remind yourself of the performance (discrepancy).

```
$ nvcc -o matrix_sums matrix_sums.cu
$ srun --reservation=curso --gres=gpu:1 nsys profile --stats=true -t cuda --cuda-memory-usage=true matrix_sums
```

Remember our top 2 CUDA optimization priorities from the previous session: **lots of threads** and **efficient use of the memory subsystem** a high rate of **bits-in-flight**. The original `row_sums` kernel misses the mark for the memory objective. What we've learned about reductions should guide you.  There are probably several ways to tackle this:

- Write a straightforward parallel reduction, run it on a row, and use a for-loop to loop the kernel over all rows
- Assign a warp to each row to perform everything in one kernel call
- Assign a thread block to each row to perform everything in one kernel call
- ??

Since the (given) solution may be somewhat unusual, I'll give some hints here if needed:

 - The chosen strategy will be to assign one block per row
 - We must modify the kernel launch to launch exactly as many blocks as we have rows
 - The kernel can be adapted from the reduction kernel (atomic is not needed here) from the reduce kernel code in exercise 1 above
 - Since assigning one block per row, we will cause each block to perform a block-striding loop to traverse the row. This is conceptually similar to a grid striding loop, except each block is striding individually, one per row. Refresh your memory of the grid-stride loop, and see if you can work this out
 - With the block-stride loop, you'll need to think carefully about indexing

After you have completed the work and are getting a successful result, profile the code again to see if the performance of the row_sums kernel has improved:

```
$ nvcc -o matrix_sums matrix_sums.cu
$ srun --reservation=curso --gres=gpu:1 nsys profile --stats=true -t cuda --cuda-memory-usage=true matrix_sums
```

Your actual performance here (compared to the reasonably efficient `column_sums` kernel) will probably depend quite a bit on the algorithm/method you choose. See if you can theorize how the various choices affect efficiency or optimality. 

If you end up with a solution where the `row_sums` kernel runs faster than the `column_sums` kernel, see if you can theorize why this may be the case. Remember the two CUDA optimization priorities, and use these to guide your thinking.

# COLAB Users
[https://colab.research.google.com/drive/1MWozOyzm_87ScuKAQu9knWZeZf3KUvX8?usp=sharing](https://colab.research.google.com/drive/1MWozOyzm_87ScuKAQu9knWZeZf3KUvX8?usp=sharing)
