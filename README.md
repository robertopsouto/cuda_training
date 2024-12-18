## Petrobras CUDA Training
     
### **DAY 1: Fundamental CUDA Optimization**

1. **Latency Hiding**
   - Thread and Warp Scheduling
   - Launch Configuration

2. **Memory Hierarchy and Access Patterns**
   - Local Storage, Shared Memory, and Global Memory
     
3. **Understanding Bottlenecks**
   - Memory-bound vs compute-bound codes
   - Extracting Bandwidth
   -  Memory Alignment

4. **Assignment**
   
### **DAY 2: Atomics, Reductions, Warp Shuffle**
1. **Atomics and Reductions**
   - Atomic Operations
   - Classical Parallel Reduction
   - Parallel Reduction + Atomics

2. **Warp Shuffle Techniques**
   - Warp-Level Reduction
   - Reduction With Warp Shuffle

3. **Assignment**
   
### **DAY 3: CUDA Concurrency**
1. **Pinned Memory**
   - Definition and Benefits
   - Usage and Functions
   - Implications for Host Memory
2. **CUDA Streams**
   - Overview, Purpose, and Semantics
   - Creation, Usage, and Copy-compute overlap
   - Stream behavior examples and Default stream considerations
3. **Multi-GPU Concurrency**
    - Device management
    - Streams across multiple GPUs
    - Device-to-device data copying
    - Peer-to-peer transfers
4. **Assignment**
   
### **DAY 4: CUDA Performance Optimization**

A different perspective of all the delivered content, using the GTC presentation as a baseline [CUDA Performance Optimization.](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62191/)

### **DAY 5: Practical Advice, Example Codes Diving (Wave Propagators)**

## Nsight Systems & Nsight Compute

To proper open ".nsys-rep" files, please download the tool **Nsight Systems** [[link](https://developer.nvidia.com/nsight-systems/get-started)]

Similarly, use **Nsight Compute** tool to open ".ncu-rep" files [[link](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)].
