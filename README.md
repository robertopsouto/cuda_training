## Petrobras CUDA Training

### **Module 0: GPU Architecture Overview**
1. **GPU Architecture Overview**
   - **Fermi, Kepler, Maxwell, Pascal, Volta, Ampere, Hopper Architectures** - Understanding NVIDIA GPU architectures' evolution and key features.
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
   **Local Storage, Shared Memory, and Global Memory**: Understand the different types of memory and their characteristics.
   - **Caching and Non-Caching Loads** - How caching affects memory access patterns and performance.
   - **Coalescing and Bank Conflicts** - Strategies for optimizing memory access patterns to maximize throughput.

3. **Exercise**   
   
### **Module 3: Atomics, Reductions, and Warp Shuffle**
1. **Atomics and Reductions**
   - **Atomic Operations** - Introduction to atomic functions and their use in parallel reductions.
   - **Classical Parallel Reduction** - Understanding tree-based reduction methods and their implementation.
   - **Warp Shuffle** - Using warp shuffle for intra-warp communication and reduction.

2. **Warp Shuffle Techniques**
   - **Warp-Level Reduction** - Implementing reduction using warp shuffle.
   - **Broadcast and Prefix Sum** - Additional uses of warp shuffle for broadcast and prefix sum operations.

3. **Cooperative Groups**
   - **Introduction to Cooperative Groups** - Understanding the concept of cooperative groups and their levels (block, grid, multi-grid).
   - **Cooperative Group Functions** - Using `__syncthreads`, `this_thread_block`, `this_grid`, and `this_multi_grid`.

4. **Exercise**

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

4. **Exercise**
   
### **Module 5: CUDA Concurrency and Cooperative Groups**

### **Module 6: Thurst**

### **Module 7: Advanced Performance Optimization in CUDA**
1. **Cooperative Grid Arrays (CGA)**
   - A new level in the hierarchy of parallelism. Cooperative thread arrays (CTA).

3. **Memory Model**
   - CUDA weak memory model discussion and explicit synchronization.
     
5. **Asynchronous Barriers**
   - Synchronization points to ensure memory ordering within a specified scope (e.g., block, cluster, device, system).
     
7. **Asynchronous Data Copies**
   - **LDGSTS** - Asynchronous copy instruction to bypass registers, reducing scoreboard stalls and L1 bandwidth usage.
   - **TMA** - Asynchronous nD block copies between global memory and shared memory (or distributed shared memory in clusters) on Hopper architecture.
     
9. **Distributed Shared Memory (DSMEM)**
   - Shared memory access by all CTAs within a CGA, enabling data sharing across thread blocks.
      
10. **Exercise**

### **Module 8: Developer Tools Deep Dive**

### **Module 9: GPU Performance Analysis**

### **Module 10: **

### **Module 11: **
