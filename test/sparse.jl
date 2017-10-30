using CUSOLVER
using CUDAdrv
using Base.Test

import CUSPARSE: CudaSparseMatrixCSR

m = 12
n = 10

@testset for elty in [Float32, Float64, Complex64, Complex128]
    @testset "csrlsvlu!" begin
        A = sparse(rand(elty,n,n))
        b = rand(elty,n)
        x = zeros(elty,n)
        tol = convert(real(elty),1e-6)
        x = CUSOLVER.csrlsvlu!(A,b,x,tol,one(Cint),'O')
        @test x ≈ full(A)\b
        A = sparse(rand(elty,m,n))
        @test_throws DimensionMismatch CUSOLVER.csrlsvlu!(A,b,x,tol,one(Cint),'O')
        A = sparse(rand(elty,n,n))
        b = rand(elty,m)
        x = zeros(elty,n)
        @test_throws DimensionMismatch CUSOLVER.csrlsvlu!(A,b,x,tol,one(Cint),'O')
        b = rand(elty,n)
        x = zeros(elty,m)
        @test_throws DimensionMismatch CUSOLVER.csrlsvlu!(A,b,x,tol,one(Cint),'O')
    end

    @testset "csrlsvqr!" begin
        A     = sparse(rand(elty,n,n))
        d_A   = CudaSparseMatrixCSR(A)
        b     = rand(elty,n)
        d_b   = CuArray(b)
        x     = zeros(elty,n)
        d_x   = CuArray(x)
        tol   = convert(real(elty),1e-4)
        d_x   = CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
        h_x   = collect(d_x)
        @test h_x ≈ full(A)\b
        A     = sparse(rand(elty,m,n))
        d_A   = CudaSparseMatrixCSR(A)
        @test_throws DimensionMismatch CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
        A = sparse(rand(elty,n,n))
        b = rand(elty,m)
        x = zeros(elty,n)
        @test_throws DimensionMismatch CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
        b = rand(elty,n)
        x = zeros(elty,m)
        @test_throws DimensionMismatch CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
    end

    @testset "csrlsvchol!" begin
        A     = rand(elty,n,n)
        A     = sparse(A*A') #posdef
        d_A   = CudaSparseMatrixCSR(A)
        b     = rand(elty,n)
        d_b   = CuArray(b)
        x     = zeros(elty,n)
        d_x   = CuArray(x)
        tol   = 10^2*eps(real(elty))
        d_x   = CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
        h_x   = collect(d_x)
        @test h_x ≈ full(A)\b rtol=1e-4
        b     = rand(elty,m)
        d_b   = CuArray(b)
        @test_throws DimensionMismatch CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
        b     = rand(elty,n)
        d_b   = CuArray(b)
        x     = rand(elty,m)
        d_x   = CuArray(x)
        @test_throws DimensionMismatch CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
        A     = sparse(rand(elty,m,n))
        d_A   = CudaSparseMatrixCSR(A)
        @test_throws DimensionMismatch CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
    end

    @testset "csreigvsi" begin
        A     = sparse(rand(elty,n,n))
        d_A   = CudaSparseMatrixCSR(A)
        evs   = eigvals(full(A))
        x_0   = CuArray(rand(elty,n))
        μ,x   = CUSOLVER.csreigvsi(d_A,convert(elty,evs[1]),x_0,convert(real(elty),1e-6),convert(Cint,1000),'O')
        @test μ ≈ evs[1]
        A     = sparse(rand(elty,m,n))
        d_A   = CudaSparseMatrixCSR(A)
        @test_throws DimensionMismatch CUSOLVER.csreigvsi(d_A,convert(elty,evs[1]),x_0,convert(real(elty),1e-6),convert(Cint,1000),'O')
        A     = sparse(rand(elty,n,n))
        d_A   = CudaSparseMatrixCSR(A)
        x_0   = CuArray(rand(elty,m))
        @test_throws DimensionMismatch CUSOLVER.csreigvsi(d_A,convert(elty,evs[1]),x_0,convert(real(elty),1e-6),convert(Cint,1000),'O')
    end
    @testset "csreigs" begin
        celty = complex(elty)
        A   = rand(real(elty),n,n)
        A   = sparse(A + A')
        num = CUSOLVER.csreigs(A,convert(celty,complex(-100,-100)),convert(celty,complex(100,100)),'O')
        @test num <= n
        A     = sparse(rand(celty,m,n))
        d_A   = CudaSparseMatrixCSR(A)
        @test_throws DimensionMismatch CUSOLVER.csreigs(A,convert(celty,complex(-100,-100)),convert(celty,complex(100,100)),'O')
    end
    @testset "csrlsqvqr!" begin
        A = sparse(rand(elty,n,n))
        b = rand(elty,n)
        x = zeros(elty,n)
        tol = convert(real(elty),1e-4)
        x = CUSOLVER.csrlsqvqr!(A,b,x,tol,'O')
        @test x[1] ≈ full(A)\b
        A = sparse(rand(elty,n,m))
        x = zeros(elty,n)
        @test_throws DimensionMismatch CUSOLVER.csrlsqvqr!(A,b,x,tol,'O')
        A = sparse(rand(elty,n,n))
        b = rand(elty,m)
        x = zeros(elty,n)
        @test_throws DimensionMismatch CUSOLVER.csrlsqvqr!(A,b,x,tol,'O')
        b = rand(elty,n)
        x = zeros(elty,m)
        @test_throws DimensionMismatch CUSOLVER.csrlsqvqr!(A,b,x,tol,'O')
    end
end
