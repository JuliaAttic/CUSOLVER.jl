using CUSOLVER
using CUDArt
using Base.Test

import CUSPARSE: CudaSparseMatrixCSR

m = 12
n = 10

##################
# test_csrlsvlu! #
##################
function test_csrlsvlu!(elty)
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

###################
# test_csrlsqvqr! #
###################
function test_csrlsqvqr!(elty)
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

##################
# test_csrlsvqr! #
##################
function test_csrlsvqr!(elty)
    A     = sparse(rand(elty,n,n))
    d_A   = CudaSparseMatrixCSR(A)
    b     = rand(elty,n)
    d_b   = CudaArray(b)
    x     = zeros(elty,n)
    d_x   = CudaArray(x)
    tol   = convert(real(elty),1e-4)
    d_x   = CUSOLVER.csrlsvqr!(d_A,d_b,d_x,tol,one(Cint),'O')
    h_x   = to_host(d_x)
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

####################
# test_csrlsvchol! #
####################
function test_csrlsvchol!(elty)
    A     = rand(elty,n,n)
    A     = sparse(A*A') #posdef
    d_A   = CudaSparseMatrixCSR(A)
    b     = rand(elty,n)
    d_b   = CudaArray(b)
    x     = zeros(elty,n)
    d_x   = CudaArray(x)
    tol   = 10^2*eps(real(elty))
    d_x   = CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
    h_x   = to_host(d_x)
    @test h_x ≈ full(A)\b
    b     = rand(elty,m)
    d_b   = CudaArray(b)
    @test_throws DimensionMismatch CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
    b     = rand(elty,n)
    d_b   = CudaArray(b)
    x     = rand(elty,m)
    d_x   = CudaArray(x)
    @test_throws DimensionMismatch CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
    A     = sparse(rand(elty,m,n))
    d_A   = CudaSparseMatrixCSR(A)
    @test_throws DimensionMismatch CUSOLVER.csrlsvchol!(d_A,d_b,d_x,tol,zero(Cint),'O')
end

##################
# test_csreigvsi #
##################
function test_csreigvsi(elty)
    A     = sparse(rand(elty,n,n))
    d_A   = CudaSparseMatrixCSR(A)
    evs   = eigvals(full(A))
    x_0   = CudaArray(rand(elty,n))
    μ,x   = CUSOLVER.csreigvsi(d_A,convert(elty,evs[1]),x_0,convert(real(elty),1e-6),convert(Cint,1000),'O')
    @test μ ≈ evs[1]
    A     = sparse(rand(elty,m,n))
    d_A   = CudaSparseMatrixCSR(A)
    @test_throws DimensionMismatch CUSOLVER.csreigvsi(d_A,convert(elty,evs[1]),x_0,convert(real(elty),1e-6),convert(Cint,1000),'O')
    A     = sparse(rand(elty,n,n))
    d_A   = CudaSparseMatrixCSR(A)
    x_0   = CudaArray(rand(elty,m))
    @test_throws DimensionMismatch CUSOLVER.csreigvsi(d_A,convert(elty,evs[1]),x_0,convert(real(elty),1e-6),convert(Cint,1000),'O')
end

################
# test_csreigs #
################
function test_csreigs(elty)
    A   = rand(real(elty),n,n)
    A   = sparse(A + A')
    num = CUSOLVER.csreigs(A,convert(elty,complex(-100,-100)),convert(elty,complex(100,100)),'O')
    @test num <= n
    A     = sparse(rand(elty,m,n))
    d_A   = CudaSparseMatrixCSR(A)
    @test_throws DimensionMismatch CUSOLVER.csreigs(A,convert(elty,complex(-100,-100)),convert(elty,complex(100,100)),'O')
end

types = [Float32, Float64, Complex64, Complex128]
@testset for test_func in [
        test_csreigvsi,
        test_csrlsvlu!,
        test_csrlsvchol!,
        test_csrlsvqr!,
        test_csrlsqvqr!]
    @testset for elty in types
        test_func(elty)
    end
end
@testset "csreigs" begin
    @testset for elty in types
        test_csreigs(complex(elty))
    end
end
