# CUSOLVER

Julia bindings for the [NVIDIA CUSOLVER](http://docs.nvidia.com/cuda/cusolver) library. CUSOLVER is a high-performance direct-solver matrix linear algebra library.

## Introduction

`CUSOLVER.jl` provides bindings to a subset of the CUSOLVER library. It's built on top of `CUBLAS.jl`, `CUSPARSE.jl` and `CUDArt.jl`. `CUSOLVER.jl` currently wraps all the dense solvers and the sparse solvers are in progress.

The dense CUSOLVER API is designed to mimic the LAPACK API. I've tried to achieve consistency with the Julia base LAPACK bindings so that you can use CUSOLVER as a drop-in replacement. `CUSOLVER.jl` will use the `CUSPARSE.jl` custom types for ease-of-use.

## Current Features

`CUSOLVER.jl` currently supports a subset of all the CUSPARSE functionality. What is implemented right now:
- [x] Dense API
    - [x] Dense Linear Solvers
        - [x] potrf!
        - [x] potrs!
        - [x] getrf!
        - [x] getrs!
        - [x] geqrf!
        - [x] ormqr!
        - [x] sytrf!
    - [x] Dense Eigensolvers
        - [x] gebrd!
        - [x] gesvd!
- [x] Sparse API
    - [ ] High level API
        - [ ] csrlsvlu!
        - [ ] csrlsvqr!
        - [ ] csrlsvchol!
        - [ ] csrlsqvqr!
        - [ ] csreigvsi!
        - [ ] csreigs!
    - [ ] Low level API
        - [ ] csrsymrcm!
        - [ ] csrsymmdq!
        - [ ] csrsymamd!
        - [ ] csrperm!
        - [ ] csrqrBatched!

## Contributing

Contributions are very welcome! If you write wrappers for one of the `CUSOLVER.jl` functions, please include some tests in `test/runtests.jl` for your wrapper. Ideally test each of the types the function you wrap can accept, e.g. `Float32`, `Float64`, and possibly `Complex64`, `Complex128`.
