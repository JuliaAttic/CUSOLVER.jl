using CUSOLVER
using CUDArt
using Base.Test

m = 15
n = 10
k = 1

@testset for elty in [Float32, Float64, Complex64, Complex128]
    @testset "potrf!" begin
        A    = rand(elty,n,n)
        A    = A*A' #posdef
        d_A  = CudaArray(A)
        d_A  = CUSOLVER.potrf!('U',d_A)
        h_A  = to_host(d_A)
        cA,_ = LAPACK.potrf!('U',A)
        @test h_A ≈ cA
        A    = rand(elty,m,n)
        d_A  = CudaArray(A)
        @test_throws DimensionMismatch CUSOLVER.potrf!('U',d_A)
        A    = zeros(elty,n,n)
        d_A  = CudaArray(A)
        @test_throws Base.LinAlg.SingularException CUSOLVER.potrf!('U',d_A)
    end
    @testset "potrs!" begin
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
        A    = rand(elty,m,n)
        d_A  = CudaArray(A)
        @test_throws DimensionMismatch CUSOLVER.potrs!('U',d_A,d_B)
        A    = rand(elty,n,n)
        d_A  = CudaArray(A)
        B     = rand(elty,m,m)
        d_B   = CudaArray(B)
        @test_throws DimensionMismatch CUSOLVER.potrs!('U',d_A,d_B)
    end

    @testset "getrf!" begin
        A          = rand(elty,m,n)
        d_A        = CudaArray(A)
        d_A,d_ipiv = CUSOLVER.getrf!(d_A)
        h_A        = to_host(d_A)
        h_ipiv     = to_host(d_ipiv)
        alu        = Base.LinAlg.LU(h_A, convert(Vector{Int},h_ipiv), zero(Int))
        @test A ≈ full(alu)
        A    = zeros(elty,n,n)
        d_A  = CudaArray(A)
        @test_throws Base.LinAlg.SingularException CUSOLVER.getrf!(d_A)
    end

    @testset "getrs!" begin
        A          = rand(elty,n,n)
        d_A        = CudaArray(A)
        d_A,d_ipiv = CUSOLVER.getrf!(d_A)
        B          = rand(elty,n,n)
        d_B        = CudaArray(B)
        d_B        = CUSOLVER.getrs!('N',d_A,d_ipiv,d_B)
        h_B        = to_host(d_B)
        @test h_B  ≈ A\B
        A          = rand(elty,m,n)
        d_A        = CudaArray(A)
        @test_throws DimensionMismatch CUSOLVER.getrs!('N',d_A,d_ipiv,d_B)
        A          = rand(elty,n,n)
        d_A        = CudaArray(A)
        B          = rand(elty,m,n)
        d_B        = CudaArray(B)
        @test_throws DimensionMismatch CUSOLVER.getrs!('N',d_A,d_ipiv,d_B)
    end

    @testset "geqrf!" begin
        A         = rand(elty,m,n)
        d_A       = CudaArray(A)
        d_A,d_tau = CUSOLVER.geqrf!(d_A)
        h_A       = to_host(d_A)
        h_tau     = to_host(d_tau)
        qra       = Base.LinAlg.QR(h_A, h_tau)
        @test A ≈ full(qra)
    end

    @testset "ormqr!" begin
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

    @testset "sytrf!" begin
        A          = rand(elty,n,n)
        A          = A + A' #symmetric
        d_A        = CudaArray(A)
        d_A,d_ipiv = CUSOLVER.sytrf!('U',d_A)
        h_A        = to_host(d_A)
        h_ipiv     = to_host(d_ipiv)
        A, ipiv    = LAPACK.sytrf!('U',A)
        @test ipiv == h_ipiv 
        @test A ≈ h_A 
        A    = rand(elty,m,n)
        d_A  = CudaArray(A)
        @test_throws DimensionMismatch CUSOLVER.sytrf!('U',d_A)
        A    = zeros(elty,n,n)
        d_A  = CudaArray(A)
        @test_throws Base.LinAlg.SingularException CUSOLVER.sytrf!('U',d_A)
    end

    @testset "gebrd!" begin
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

    @testset "gesvd!" begin
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
end
