
# GAIA Users
## 1 - Investigating Copy-compute Overlap

You are provided with a code for your initial assignment that performs an element-wise computation on a vector. To begin, you can compile, execute, and profile the code using the following commands:

```
$ module load cuda/12.0
$ nvcc -o overlap overlap.cu
$ srun --reservation=curso --gres=gpu:1 overlap 
```

In this scenario, the output will display the elapsed execution time for the non-overlapped version of the code. The process involves transferring the entire vector to the device, launching the processing kernel, and then copying the results back to the host. Additionally, consider enabling profiling for further analysis.

```
$ srun --reservation=curso --gres=gpu:1 nsys profile --stats=true -t cuda --cuda-memory-usage=true overlap
```

Your objective is to implement a fully overlapped version of the code. Apply your understanding of CUDA streams to divide the workload into multiple chunks. For each chunk, issue the device transfer, kernel launch, and host transfer within a single stream, then adjust the stream configuration before processing the next chunk.

The initial implementation has been provided after the `#ifdef` directive. Locate each `FIXME` comment within the code and replace it with the appropriate code segments necessary to complete the assignment.

When you have code ready for testing, compile it using the following additional switch:

```
$ nvcc -o overlap overlap.cu -DUSE_STREAMS
```

A verification check will ensure you have processed the entire vector correctly in chunks. If you pass the verification test, the program will display the elapsed time of the streamed version. 

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

**Hint**: This exercise might be simpler than you think. You won't need to do anything with streams, and you'll only need to make a simple modification to each of the for-loops.
