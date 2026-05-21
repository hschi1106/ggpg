# Tests

The upstream executable runs its C++ smoke tests at startup. This fork extends that path with serializer coverage in `src/tests.hpp`.

Recommended checks:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=debug
cmake --build build/debug
./build/debug/gpg -train dataset/diabetes_train.csv -g 1 -disable_ims -ff mse -verbose
```

CUDA checks require a CUDA build:

```bash
GPG_USE_CUDA=1 cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=release -DGPG_USE_CUDA=ON
cmake --build build/cuda
./build/cuda/gpg -train dataset/diabetes_train.csv -g 1 -disable_ims -ff mse -backend gpu_exact_gom -gpu_check_correctness -verbose
./build/cuda/gpg -train dataset/diabetes_train.csv -g 1 -disable_ims -ff mse -backend gpu_batch_gom -gpu_batch_size 8 -gpu_check_correctness -verbose
```
