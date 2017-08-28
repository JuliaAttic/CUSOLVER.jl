# CUSOLVER

**Build status**: [![][buildbot-julia05-img]][buildbot-julia05-url] [![][buildbot-julia06-img]][buildbot-julia06-url]

**Code coverage**: [![Coverage Status](https://codecov.io/gh/JuliaGPU/CUSOLVER.jl/coverage.svg)](https://codecov.io/gh/JuliaGPU/CUSOLVER.jl)

[buildbot-julia05-img]: http://ci.maleadt.net/shields/build.php?builder=CUSOLVER-julia05-x86-64bit&name=julia%200.5
[buildbot-julia05-url]: http://ci.maleadt.net/shields/url.php?builder=CUSOLVER-julia05-x86-64bit
[buildbot-julia06-img]: http://ci.maleadt.net/shields/build.php?builder=CUSOLVER-julia06-x86-64bit&name=julia%200.6
[buildbot-julia06-url]: http://ci.maleadt.net/shields/url.php?builder=CUSOLVER-julia06-x86-64bit

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
