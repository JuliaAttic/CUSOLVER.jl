# CUSOLVER

**Build status**: [![](https://ci.maleadt.net/buildbot/julia/badge.svg?builder=CUSOLVER.jl:%20Julia%200.5%20(x86-64)&badge=Julia%20v0.5)](https://ci.maleadt.net/buildbot/julia/builders/CUSOLVER.jl%3A%20Julia%200.5%20%28x86-64%29) [![](https://ci.maleadt.net/buildbot/julia/badge.svg?builder=CUSOLVER.jl:%20Julia%200.6%20(x86-64)&badge=Julia%200.6)](https://ci.maleadt.net/buildbot/julia/builders/CUSOLVER.jl%3A%20Julia%200.6%20%28x86-64%29)

**Code coverage**: [![Coverage Status](https://codecov.io/gh/JuliaGPU/CUSOLVER.jl/coverage.svg)](https://codecov.io/gh/JuliaGPU/CUSOLVER.jl)

Julia bindings for the [NVIDIA CUSOLVER](http://docs.nvidia.com/cuda/cusolver) library. CUSOLVER is a high-performance direct-solver matrix linear algebra library.

## Introduction

`CUSOLVER.jl` provides bindings to a subset of the CUSOLVER library. It's built on top of `CUBLAS.jl`, `CUSPARSE.jl` and `CUDArt.jl`. `CUSOLVER.jl` currently wraps all the dense solvers and the sparse solvers are in progress.

The dense CUSOLVER API is designed to mimic the LAPACK API. I've tried to achieve consistency with the Julia base LAPACK bindings so that you can use CUSOLVER as a drop-in replacement. `CUSOLVER.jl` will use the `CUSPARSE.jl` custom types for ease-of-use.

## Current Features

`CUSOLVER.jl` currently supports a subset of all the CUSOLVER functionality. What is implemented right now:
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
    - [x] High level API
        - [x] csrlsvlu!
        - [x] csrlsvqr!
        - [x] csrlsvchol!
        - [x] csrlsqvqr!
        - [x] csreigvsi!
        - [x] csreigs!
    - [ ] Low level API
        - [ ] csrsymrcm!
        - [ ] csrsymmdq!
        - [ ] csrsymamd!
        - [ ] csrperm!
        - [ ] csrqrBatched!

## Contributing

Contributions are very welcome! If you write wrappers for one of the `CUSOLVER.jl` functions, please include some tests in `test/runtests.jl` for your wrapper. Ideally test each of the types the function you wrap can accept, e.g. `Float32`, `Float64`, and possibly `Complex64`, `Complex128`.
