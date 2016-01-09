using CUSOLVER
using CUDArt
using Base.Test

m = 15
n = 10
k = 1

##############
# test_potrf #
##############

function test_potrf!(elty)
    A    = rand(elty,n,n)
    A    = A*A' #posdef
    d_A  = CudaArray(A)
    d_A  = CUSOLVER.potrf!('U',d_A)
    h_A  = to_host(d_A)
    cA,_ = LAPACK.potrf!('U',A)
    @test h_A ≈ cA
end

##############
# test_potrs #
##############
function test_potrs!(elty)
    A     = rand(elty,n,n)
    A     = A*A' #posdef
    d_A   = CudaArray(A)
    d_A   = CUSOLVER.potrf!('U',d_A)
    h_A   = to_host(d_A)
    B     = rand(elty,n,n)
    d_B   = CudaArray(B)
    d_B   = CUSOLVER.potrs!('U',d_A,d_B)
    h_B   = to_host(d_B)
    @test h_B ≈ LAPACK.potrs!('U',h_A,B)
end

##############
# test_getrf #
##############

function test_getrf!(elty)
    A          = rand(elty,m,n)
    d_A        = CudaArray(A)
    d_A,d_ipiv = CUSOLVER.getrf!(d_A)
    h_A        = to_host(d_A)
    h_ipiv     = to_host(d_ipiv)
    alu        = Base.LinAlg.LU(h_A, convert(Vector{Int},h_ipiv), zero(Int))
    @test A ≈ full(alu)
end

##############
# test_getrs #
##############
function test_getrs!(elty)
    A          = rand(elty,n,n)
    d_A        = CudaArray(A)
    d_A,d_ipiv = CUSOLVER.getrf!(d_A)
    B          = rand(elty,n,n)
    d_B        = CudaArray(B)
    d_B        = CUSOLVER.getrs!('N',d_A,d_ipiv,d_B)
    h_B        = to_host(d_B)
    @test h_B ≈ A\B
end

##############
# test_geqrf #
##############

function test_geqrf!(elty)
    A         = rand(elty,m,n)
    d_A       = CudaArray(A)
    d_A,d_tau = CUSOLVER.geqrf!(d_A)
    h_A       = to_host(d_A)
    h_tau     = to_host(d_tau)
    qra       = Base.LinAlg.QR(h_A, h_tau)
    @test A ≈ full(qra)
end

##############
# test_ormqr #
##############
function test_ormqr!(elty)
    A         = rand(elty,n,n)
    d_A       = CudaArray(A)
    d_A,d_tau = CUSOLVER.geqrf!(d_A)
    h_A       = to_host(d_A)
    h_tau     = to_host(d_tau)
    qra       = Base.LinAlg.QR(h_A, h_tau)
    B         = rand(elty,n,n)
    d_B       = CudaArray(B)
    d_B       = CUSOLVER.ormqr!('L','N',d_A,d_tau,d_B)
    h_B       = to_host(d_B)
    @test h_B ≈ qra[:Q]*B
end

##############
# test_sytrf #
##############
function test_sytrf!(elty)
    A          = rand(elty,n,n)
    A          = A + A' #symmetric
    d_A        = CudaArray(A)
    d_A,d_ipiv = CUSOLVER.sytrf!('U',d_A)
    h_A        = to_host(d_A)
    h_ipiv     = to_host(d_ipiv)
    A, ipiv    = LAPACK.sytrf!('U',A)
    @test ipiv == h_ipiv 
    @test A ≈ h_A 
end

##############
# test_gebrd #
##############
function test_gebrd!(elty)
    A                             = rand(elty,m,n)
    d_A                           = CudaArray(A)
    d_A, d_D, d_E, d_TAUQ, d_TAUP = CUSOLVER.gebrd!(d_A)
    h_A                           = to_host(d_A)
    h_D                           = to_host(d_D)
    h_E                           = to_host(d_E)
    h_TAUQ                        = to_host(d_TAUQ)
    h_TAUP                        = to_host(d_TAUP)
    A,d,e,q,p                     = LAPACK.gebrd!(A)
    #@test A ≈ h_A
    @test d ≈ h_D
    @test e ≈ h_E
    @test q ≈ h_TAUQ
    @test p ≈ h_TAUP
end

##############
# test_gesvd #
##############
function test_gesvd!(elty)
    A              = rand(elty,m,n)
    d_A            = CudaArray(A)
    d_U, d_S, d_Vt = CUSOLVER.gesvd!('A','A',d_A)
    h_S            = to_host(d_S)
    h_U            = to_host(d_U)
    h_Vt           = to_host(d_Vt)
    svda           = svdfact(A,thin=false)
    @test h_U ≈ svda[:U]
    @test h_S ≈ svdvals(A)
    @test h_Vt ≈ svda[:Vt]
end

types = [Float32, Float64, Complex64, Complex128]
for elty in types
    test_potrf!(elty)
    test_potrs!(elty)
    test_getrf!(elty)
    test_getrs!(elty)
    test_geqrf!(elty)
    test_ormqr!(elty)
    test_sytrf!(elty)
    test_gebrd!(elty)
    test_gesvd!(elty)
end
