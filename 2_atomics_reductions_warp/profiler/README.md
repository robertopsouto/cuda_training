## NCU Profiler Generation Commands

```
$ ncu -o profile_atomic_red  --target-processes all --set full --nvtx --import-source yes -k atomic_red -c 6 ./reductions

$ ncu -o profile_reduce_a  --target-processes all --set full --nvtx --import-source yes -k reduce_a -c 6 ./reductions

$ ncu -o profile_reduce_ws  --target-processes all --set full --nvtx --import-source yes -k reduce_ws -c 6 ./reductions
```
