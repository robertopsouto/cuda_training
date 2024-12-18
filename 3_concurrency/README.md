
# GAIA Users
## 1 - Investigating Copy-compute Overlap

You are given a code for your first task that performs a silly computation element-wise on a vector.  You can initially compile, run, and profile the code with the following commands:

```
$ module load cuda/12.0
$ nvcc -o overlap overlap.cu
$ srun --reservation=curso --gres=gpu:1 ./overlap 
```

In this case, the output will show the elapsed time of the non-overlapped version of the code. This code copies the entire vector to the device, launches the processing kernel, and copies the whole vector back to the host. Also, try the profiling:

```
$ srun --reservation=curso --gres=gpu:1 nsys profile --stats=true -t cuda --cuda-memory-usage=true overlap
```

This `nsys profile` output should show you the sequence of operations (`cudaMemcpy` Host to Device, kernel call, and `cudaMemcpy` Device To Host). The kernel has an initial "warm-up" run; disregard this.  You should be able to witness that each operation's start and duration indicate no overlap.

Your objective is to create a fully overlapped code version for you. Use your knowledge of streams to make a version of the code that will issue the work in chunks, and for each chunk, perform the copy to device, kernel launch, and copy to host in a single stream, then modify the stream for the next chunk.

The work has been started for you in the code section after the `#ifdef` statement.  Look for the `FIXME` tokens, and replace each `FIXME` with the appropriate code to complete this task.

When you have something ready to test, compile it with this additional switch:

```
$ nvcc -o overlap overlap.cu -DUSE_STREAMS
```

A verification check will ensure you have processed the entire vector correctly in chunks.  If you pass the verification test, the program will display the elapsed time of the streamed version. 

You should be able to get to at least 2X faster (i.e., half the duration) of the non-streamed version. You can also run this code with the `nsys profile` using the command given above.  

This will generate a lot of output, however if you examine some of the operations at the end of the output, you should be able to confirm that there is indeed overlap of operations, based on the starting time and duration of each operation.

# 2 - Simple Multi-GPU

This exercise provides a simple code that performs four kernel calls sequentially on a single GPU. You're welcome to compile and run the code as-is if you don't mind. It will display the overall duration of the four kernel calls. Your task is to modify this code to run each kernel on a separate GPU (the `Slurm` reservation should put you on a machine that has 4 GPUs available to you).  After completion, confirm that the execution time is substantially reduced. Compile the code with:

```
$ nvcc -o multi multi.cu
```

You can run the code with:

```
$ srun --reservation=curso --gres=gpu:4 multi
```

**Hint**: This exercise might be simpler than you think.  You won't need to do anything with streams at all for this.  You'll only need to make a simple modification to each of the for-loops.
