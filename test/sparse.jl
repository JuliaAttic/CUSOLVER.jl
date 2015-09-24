using CUSOLVER
using CUDArt
using Base.Test

import CUSPARSE: CudaSparseMatrixCSR

m = 15
n = 10

##################
# test_csrlsvlu! #
##################
function test_csrlsvlu!(elty)
    A     = sparse(rand(elty,n,n))
    d_A   = CudaSparseMatrixCSR(A)
    b     = rand(elty,n)
    d_b   = CudaArray(b)
    x     = zeros(elty,n)
    d_x   = CudaArray(x)

    d_x   = CUSOLVER.csrlsvlu!(d_A,d_b,d_x,10^2*eps(elty),convert(Cint,0),'O')
    h_x   = to_host(d_x)
    @test h_x ≈ A\b
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

    d_x   = CUSOLVER.csrlsvqr!(d_A,d_b,d_x,10^2*eps(elty),convert(Cint,0),'O')
    h_x   = to_host(d_x)
    @test h_x ≈ A\b
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

    d_x   = CUSOLVER.csrlsvchol!(d_A,d_b,d_x,10^2*eps(elty),convert(Cint,0),'O')
    h_x   = to_host(d_x)
    @test h_x ≈ A\b
end

types = [Float32, Float64, Complex64, Complex128]
for elty in types
    test_csrlsvchol!(elty)
    test_csrlsvqr!(elty)
    test_csrlsvlu!(elty)
end
