## Petrobras CUDA Training
     
### **Module 0: Introduction to CUDA C Programming (Pre-Training)**
1. **GPU Architecture Overview**
   - Ampere & Hopper Architectures
   - CUDA Execution Model
     
2. **Overview of CUDA**
   - What is CUDA?
   - CUDA C/C++ Basics
   - Heterogeneous Computing

3. **CUDA Programming Fundamentals**
   - GPU Kernels
   - Memory Management
   - Parallel Programming

4. **Vector Addition Example**
   - Parallel Vector Addition
   - Thread and Block Configuration

5. **Exercise**

### **Module 1: Fundamental CUDA Optimization**

1. **Latency Hiding**
   - Thread and Warp Scheduling
   - Launch Configuration

2. **Memory Hierarchy and Access Patterns**
   - Local Storage, Shared Memory, and Global Memory
     
3. **Understanding Bottlenecks**
   - Memory-bound vs compute-bound codes
   - Extracting Bandwidth
   -  Memory Alignment

3. **Exercise**
   
### **Module 2: Atomics, Reductions, Warp Shuffle, CG**
1. **Atomics and Reductions**
   - Atomic Operations
   - Classical Parallel Reduction
   - Warp Shuffle

2. **Warp Shuffle Techniques**
   - Warp-Level Reduction
   - Broadcast and Prefix Sum

3. **Cooperative Groups**
   - Introduction to Cooperative Groups
   - Cooperative Group Functions
   - C++ Atomics

4. **Exercise**

### **Module 3: Memory Spaces + Grace-Hopper**
1. **Unified Memory Basics**
   - Introduction to Unified Memory
   - Managed Memory Allocation

2. **Use Cases**
   - Deep Copy and Linked Lists
   - C++ Objects

3. **Demand Paging and Oversubscription**
   - Pascal and Beyond
   - Performance Tuning

4. **Grace-Hopper**
    - ???

5. **Exercise**
   
### **Module 4: CUDA Concurrency**
1. **Pinned Memory**
2. **CUDA Streams**
3. **Multi-GPU Concurrency**

### **Module 5: Performance Libraries and More**
1. **Fast Fourier Transform with cuFFT**
2. **Compression with nvCOMP**
3. **ISO C++ GPU Programming**
4. **GPU Programming with Python**

### **Module 6: Advanced Performance Optimization in CUDA**
1. **Cooperative Grid Arrays (CGA)**   
2. **Memory Model**      
4. **Asynchronous Barriers**     
5. **Asynchronous Data Copies**     
6. **Distributed Shared Memory (DSMEM)**     
7. **Exercise**

### **Module 7: Developer Tools Deep Dive & GPU Performance Analysis**

### **Module 8: Coding From Scratch a Finite-difference Propagation Kernel**

### **Module 9: EnergySDK Samples Optimizations Discussion**
   - **RTM**
   - **FWI**
   - **SRME**
   - **Kirchhoff**
