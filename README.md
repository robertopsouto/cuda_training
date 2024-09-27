## Petrobras CUDA Training

### **Module 0: GPU Architecture Overview**
1. **GPU Architecture Overview**
   - **Fermi, Kepler, Maxwell, Pascal, Volta, Ampere, Hopper Architectures** - Understanding the evolution and key features of NVIDIA GPU architectures.
   - **Execution Model** - How threads and blocks are executed on multiprocessors.
     
### **Module 1: Introduction to CUDA C Programming (Pre-Training)**
1. **Overview of CUDA**
   - **What is CUDA?** - Introduction to CUDA architecture and its purpose in general-purpose computing.
   - **CUDA C/C++ Basics** - Understanding the extensions to standard C/C++ for heterogeneous programming.
   - **Heterogeneous Computing** - Terminology and concepts (host, device, etc.).

2. **CUDA Programming Fundamentals**
   - **GPU Kernels** - Writing and launching CUDA kernels, including the use of `__global__` and kernel launch syntax.
   - **Memory Management** - Basic device memory management using `cudaMalloc`, `cudaFree`, and `cudaMemcpy`.
   - **Parallel Programming** - Launching parallel kernels, understanding blocks and threads, and using `blockIdx.x` and `threadIdx.x`.

3. **Vector Addition Example**
   - **Parallel Vector Addition** - Implementing vector addition using CUDA, including memory allocation and kernel launch.
   - **Thread and Block Configuration** - Understanding how to configure threads and blocks for efficient parallel execution.

4. **Exercise**

### **Module 2: Fundamental CUDA Optimization**

1. **Latency Hiding**
   - **Thread and Warp Scheduling** - How the GPU hides latency by switching between threads and warps.
   - **Launch Configuration** - Determining the optimal number of threads and blocks to launch.

2. **Memory Hierarchy and Access Patterns**
   - **Local Storage, Shared Memory, and Global Memory** - Understanding the different types of memory and their characteristics.
   - **Caching and Non-Caching Loads** - How caching affects memory access patterns and performance.
   - **Coalescing and Bank Conflicts** - Strategies for optimizing memory access patterns to maximize throughput.

### **Module 3: Atomics, Reductions, and Warp Shuffle**
1. **Atomics and Reductions**
   - **Atomic Operations** - Introduction to atomic functions and their use in parallel reductions.
   - **Classical Parallel Reduction** - Understanding tree-based reduction methods and their implementation.
   - **Warp Shuffle** - Using warp shuffle for intra-warp communication and reduction.

2. **Warp Shuffle Techniques**
   - **Warp-Level Reduction** - Implementing reduction using warp shuffle[3].
   - **Broadcast and Prefix Sum** - Additional uses of warp shuffle for broadcast and prefix sum operations.

3. **Cooperative Groups**
   - **Introduction to Cooperative Groups** - Understanding the concept of cooperative groups and their levels (block, grid, multi-grid)[3].
   - **Cooperative Group Functions** - Using `__syncthreads`, `this_thread_block`, `this_grid`, and `this_multi_grid`.

### **Module 4: Unified Memory**
1. **Unified Memory Basics**
   - **Introduction to Unified Memory** - Understanding the concept and benefits of unified memory.
   - **Managed Memory Allocation** - Using `cudaMallocManaged` and accessing managed memory from CPU and GPU.

2. **Use Cases**
   - **Deep Copy and Linked Lists** - Simplifying complex data structures with unified memory.
   - **C++ Objects** - Overloading `new` and `delete` for managed memory.

3. **Demand Paging and Oversubscription**
   - **Pascal and Beyond** - Understanding demand paging and GPU memory oversubscription.
   - **Performance Tuning** - Using prefetching and memory hints for optimal performance.

### **Module 5: GPU Performance Analysis**
