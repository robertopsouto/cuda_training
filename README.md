## Petrobras CUDA Training
     
### **DAY 1: Fundamental CUDA Optimization**

1. **Latency Hiding**
   - Thread and warp scheduling
   - Launch configuration

2. **Memory Hierarchy and Access Patterns**
   - Local storage, shared memory, and global memory
     
3. **Understanding Bottlenecks**
   - Memory-bound vs compute-bound codes
   - Extracting bandwidth
   - Memory alignment

4. **Assignment**
   
### **DAY 2: Atomics, Reductions, Warp Shuffle**
1. **Atomics and Reductions**
   - Atomic operations
   - Classical parallel reduction
   - Parallel reduction + atomics

2. **Warp Shuffle Techniques**
   - Warp-Level reduction
   - Reduction with warp shuffle

3. **Assignment**
   
### **DAY 3: CUDA Concurrency**
1. **Pinned Memory**
   - Definition and benefits
   - Usage and functions
   - Implications for host memory
     
2. **CUDA Streams**
   - Overview, purpose, and semantics
   - Creation, Usage, and copy-compute overlap
   - Stream behavior examples and default stream considerations
     
3. **Multi-GPU Concurrency**
    - Device management
    - Streams across multiple GPUs
    - Device-to-device data copying
    - Peer-to-peer transfers
      
4. **Assignment**
   
### **DAY 4: CUDA Performance Optimization**

An alternate perspective on all delivered materials, using the GTC presentation as a baseline for CUDA performance optimization [CUDA Performance Optimization.](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62191/)

### **DAY 5: Practical Advice, Example Codes Diving (Wave Propagators)**

## Nsight Systems & Nsight Compute

To proper open ".nsys-rep" files, please download the tool **Nsight Systems** [[link](https://developer.nvidia.com/nsight-systems/get-started)]

Similarly, use **Nsight Compute** tool to open ".ncu-rep" files [[link](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)].
