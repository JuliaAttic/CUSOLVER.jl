using CUSOLVER
using CUDArt
using Base.Test

import CUSPARSE: CudaSparseMatrixCSR

m = 2
n = 2

##############
# test_issym #
##############
function test_issym(elty)
    A     = rand(elty,n,n)
    A     = sparse(A+A.')
    @test CUSOLVER.issym(A)
end

##################
# test_csrlsvlu! #
##################
function test_csrlsvlu!(elty)
    A     = spdiagm(rand(elty,n))
    d_A   = CudaSparseMatrixCSR(A)
    b     = rand(elty,n)
    d_b   = CudaArray(b)
    x     = zeros(elty,n)
    d_x   = CudaArray(x)
    d_x   = CUSOLVER.csrlsvlu!(d_A,d_b,d_x,convert(real(elty),1e-6),convert(Cint,1),'O')
    h_x   = to_host(d_x)
    @test h_x ≈ A\b
end

##################
# test_csrlsvqr! #
##################
function test_csrlsvqr!(elty)
    #A     = sparse(rand(elty,n,n))
    #A     = sparse(convert(Matrix{elty},[1.0 2.0; 3.0 4.0]))
    rows  = convert(Vector{Cint},[0,2,4])
    cols  = convert(Vector{Cint},[0,1,0,1])
    vals  = convert(Vector{elty},[1,2,3,4])
    d_rows = CudaArray(rows)
    d_cols = CudaArray(cols)
    d_vals = CudaArray(vals)
    d_A   = CudaSparseMatrixCSR(elty,d_rows,d_cols,d_vals,(2,2))
    h_A   = to_host(d_A.rowPtr)
    println(h_A)
    h_A   = to_host(d_A.colVal)
    println(h_A)
    h_A   = to_host(d_A.nzVal)
    println(h_A)
    println(d_A.nnz)
    b     = rand(elty,2)
    d_b   = CudaArray(b)
    x     = zeros(elty,2)
    d_x   = CudaArray(x)

    d_x   = CUSOLVER.csrlsvqr!(d_A,d_b,d_x,convert(real(elty),1e-4),convert(Cint,1),'Z')
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

##################
# test_csreigvsi #
##################
function test_csreigvsi(elty)
    A     = sparse(rand(elty,n,n))
    println(A)
    d_A   = CudaSparseMatrixCSR(A)
    evs   = eigvals(full(A))
    x_0   = CudaArray(rand(elty,n))
    μ,x   = CUSOLVER.csreigvsi(d_A,convert(elty,evs[1]),x_0,convert(real(elty),1e-6),convert(Cint,1000),'O')
    @test μ ≈ evs[1]
end

#types = [Float32, Float64, Complex64, Complex128]
#rtypes = [Float32, Float64, Float32, Float64]
types = [Float32]
for elty in types
    test_issym(elty)
    #test_csreigvsi(elty)
    #test_csrlsvqr!(elty)
    #test_csrlsvlu!(elty)
    #test_csrlsvchol!(elty)
end
