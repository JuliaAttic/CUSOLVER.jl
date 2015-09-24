
#potrf 
for (bname, fname,elty) in ((:cusolverDnSpotrf_bufferSize, :cusolverDnSpotrf, :Float32),
                            (:cusolverDnDpotrf_bufferSize, :cusolverDnDpotrf, :Float64),
                            (:cusolverDnCpotrf_bufferSize, :cusolverDnCpotrf, :Complex64),
                            (:cusolverDnZpotrf_bufferSize, :cusolverDnZpotrf, :Complex128))
    @eval begin
        function potrf!(uplo::BlasChar,
                        A::CudaMatrix{$elty})
            cuuplo = cublasfill(uplo)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("Cholesky factorization is only possible for square matrices!"))
            end
            lda     = max(1,stride(A,2))
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, cublasFillMode_t, Cint,
                               Ptr{$elty}, Cint, Ptr{Cint}),
                              cusolverDnhandle[1], cuuplo, n, A, lda, bufSize))

            buffer  = CudaArray(zeros($elty,bufSize[1]))
            devinfo = CudaArray(zeros(Cint,1))
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, cublasFillMode_t, Cint,
                               Ptr{$elty}, Cint, Ptr{$elty}, Cint, Ptr{Cint}),
                              cusolverDnhandle[1], cuuplo, n, A, lda, buffer,
                              bufSize[1], devinfo))
            info = to_host(devinfo)
            if info[1] < 0
                throw(ArgumentError("The $(-info[1])th parameter is wrong"))
            elseif info[1] > 0
                throw(Base.LinAlg.SingularException(info[1]))
            end
            A
        end
    end
end

#getrf 
for (bname, fname,elty) in ((:cusolverDnSgetrf_bufferSize, :cusolverDnSgetrf, :Float32),
                            (:cusolverDnDgetrf_bufferSize, :cusolverDnDgetrf, :Float64),
                            (:cusolverDnCgetrf_bufferSize, :cusolverDnCgetrf, :Complex64),
                            (:cusolverDnZgetrf_bufferSize, :cusolverDnZgetrf, :Complex128))
    @eval begin
        function getrf!(A::CudaMatrix{$elty})
            m,n     = size(A)
            lda = max(1,stride(A,2))
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, Cint, Cint, Ptr{$elty}, Cint,
                               Ptr{Cint}), cusolverDnhandle[1], m, n, A, lda,
                              bufSize))

            buffer  = CudaArray(zeros($elty,bufSize[1]))
            devipiv = CudaArray(zeros(Cint,min(m,n)))
            devinfo = CudaArray(zeros(Cint,1))
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, Cint, Cint, Ptr{$elty},
                               Cint, Ptr{$elty}, Ptr{Cint}, Ptr{Cint}),
                              cusolverDnhandle[1], m, n, A, lda, buffer,
                              devipiv, devinfo))
            info = to_host(devinfo)
            if info[1] < 0
                throw(ArgumentError("The $(info[1])th parameter is wrong"))
            elseif info[1] > 0
                throw(Base.LinAlg.SingularException(info[1]))
            end
            A, devipiv
        end
    end
end

#geqrf 
for (bname, fname,elty) in ((:cusolverDnSgeqrf_bufferSize, :cusolverDnSgeqrf, :Float32),
                            (:cusolverDnDgeqrf_bufferSize, :cusolverDnDgeqrf, :Float64),
                            (:cusolverDnCgeqrf_bufferSize, :cusolverDnCgeqrf, :Complex64),
                            (:cusolverDnZgeqrf_bufferSize, :cusolverDnZgeqrf, :Complex128))
    @eval begin
        function geqrf!(A::CudaMatrix{$elty})
            m,n     = size(A)
            lda = max(1,stride(A,2))
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, Cint, Cint, Ptr{$elty}, Cint,
                               Ptr{Cint}), cusolverDnhandle[1], m, n, A,
                              lda, bufSize))

            buffer  = CudaArray(zeros($elty, bufSize[1]))
            devtau  = CudaArray(zeros($elty, min(m, n)))
            devinfo = CudaArray(zeros(Cint, 1))
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, Cint, Cint, Ptr{$elty},
                               Cint, Ptr{$elty}, Ptr{$elty}, Cint, Ptr{Cint}),
                              cusolverDnhandle[1], m, n, A, lda, devtau, buffer,
                              bufSize[1], devinfo))
            info = to_host(devinfo)
            if info[1] < 0
                throw(ArgumentError("The $(info[1])th parameter is wrong"))
            elseif info[1] > 0
                throw(Base.LinAlg.SingularException(info[1]))
            end
            A, devtau
        end
    end
end

#sytrf 
for (bname, fname,elty) in ((:cusolverDnSsytrf_bufferSize, :cusolverDnSsytrf, :Float32),
                            (:cusolverDnDsytrf_bufferSize, :cusolverDnDsytrf, :Float64),
                            (:cusolverDnCsytrf_bufferSize, :cusolverDnCsytrf, :Complex64),
                            (:cusolverDnZsytrf_bufferSize, :cusolverDnZsytrf, :Complex128))
    @eval begin
        function sytrf!(uplo::BlasChar,
                        A::CudaMatrix{$elty})
            cuuplo = cublasfill(uplo)
            n      = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("SymTridiagonal matrix must be square!"))
            end
            lda = max(1,stride(A,2))
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, Cint, Ptr{$elty}, Cint,
                               Ptr{Cint}), cusolverDnhandle[1], n, A, lda,
                              bufSize))

            buffer  = CudaArray(zeros($elty, bufSize[1]))
            devipiv = CudaArray(zeros(Cint, n))
            devinfo = CudaArray(zeros(Cint, 1))
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, cublasFillMode_t, Cint,
                               Ptr{$elty}, Cint, Ptr{Cint}, Ptr{$elty}, Cint,
                               Ptr{Cint}), cusolverDnhandle[1], cuuplo, n, A,
                              lda, devipiv, buffer, bufSize[1], devinfo))
            info = to_host(devinfo)
            if info[1] < 0
                throw(ArgumentError("The $(info[1])th parameter is wrong"))
            elseif info[1] > 0
                throw(Base.LinAlg.SingularException(info[1]))
            end
            A, devipiv
        end
    end
end

#potrs
for (fname,elty) in ((:cusolverDnSpotrs, :Float32),
                     (:cusolverDnDpotrs, :Float64),
                     (:cusolverDnCpotrs, :Complex64),
                     (:cusolverDnZpotrs, :Complex128))
    @eval begin
        function potrs!(uplo::BlasChar,
                        A::CudaMatrix{$elty},
                        B::CudaVecOrMat{$elty})
            cuuplo = cublasfill(uplo)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("Cholesky factorization is only possible for square matrices!"))
            end
            if size(B,1) != n
                throw(DimensionMismatch("first dimension of B, $(size(B,1)), must match second dimension of A, $n"))
            end
            nrhs = size(B,2)
            lda  = max(1,stride(A,2))
            ldb  = max(1,stride(B,2))

            devinfo = CudaArray(zeros(Cint,1))
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, cublasFillMode_t, Cint, Cint,
                               Ptr{$elty}, Cint, Ptr{$elty}, Cint, Ptr{Cint}),
                              cusolverDnhandle[1], cuuplo, n, nrhs, A, lda, B,
                              ldb, devinfo))
            info = to_host(devinfo)
            if info[1] < 0
                throw(ArgumentError("The $(info[1])th parameter is wrong"))
            end
            B
        end
    end
end

#getrs
for (fname,elty) in ((:cusolverDnSgetrs, :Float32),
                     (:cusolverDnDgetrs, :Float64),
                     (:cusolverDnCgetrs, :Complex64),
                     (:cusolverDnZgetrs, :Complex128))
    @eval begin
        function getrs!(trans::BlasChar,
                        A::CudaMatrix{$elty},
                        ipiv::CudaVector{Cint},
                        B::CudaVecOrMat{$elty})
            cutrans = cublasop(trans)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("LU factored matrix A must be square!"))
            end
            if size(B,1) != n
                throw(DimensionMismatch("first dimension of B, $(size(B,1)), must match second dimension of A, $n"))
            end
            nrhs = size(B,2)
            lda  = max(1,stride(A,2))
            ldb  = max(1,stride(B,2))

            devinfo = CudaArray(zeros(Cint,1))
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, cublasOperation_t, Cint, Cint,
                               Ptr{$elty}, Cint, Ptr{Cint}, Ptr{$elty}, Cint,
                               Ptr{Cint}), cusolverDnhandle[1], cutrans, n, nrhs,
                              A, lda, ipiv, B, ldb, devinfo))
            info = to_host(devinfo)
            if info[1] < 0
                throw(ArgumentError("The $(info[1])th parameter is wrong"))
            end
            B
        end
    end
end

#ormqr 
for (bname, fname,elty) in ((:cusolverDnSgeqrf_bufferSize, :cusolverDnSormqr, :Float32),
                            (:cusolverDnDgeqrf_bufferSize, :cusolverDnDormqr, :Float64),
                            (:cusolverDnCgeqrf_bufferSize, :cusolverDnCunmqr, :Complex64),
                            (:cusolverDnZgeqrf_bufferSize, :cusolverDnZunmqr, :Complex128))
    @eval begin
        function ormqr!(side::BlasChar,
                        trans::BlasChar,
                        A::CudaMatrix{$elty},
                        devtau::CudaVector{$elty},
                        C::CudaVecOrMat{$elty})
            cutrans = cublasop(trans)
            cuside  = cublasside(side)
            m,k     = size(A)
            lda     = max(1,stride(A,2))
            ldc     = max(1,stride(C,2))
            n       = size(C,2)
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, Cint, Cint, Ptr{$elty}, Cint,
                               Ptr{Cint}), cusolverDnhandle[1], m, k, A,
                              lda, bufSize))

            buffer  = CudaArray(zeros($elty, bufSize[1]))
            devinfo = CudaArray(zeros(Cint, 1))
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, cublasSideMode_t,
                               cublasOperation_t, Cint, Cint, Cint, Ptr{$elty},
                               Cint, Ptr{$elty}, Ptr{$elty}, Cint, Ptr{$elty},
                               Cint, Ptr{Cint}), cusolverDnhandle[1], cuside,
                              cutrans, m, k, n, A, lda, devtau, C, ldc, buffer,
                              bufSize[1], devinfo))
            info = to_host(devinfo)
            if info[1] < 0
                throw(ArgumentError("The $(info[1])th parameter is wrong"))
            elseif info[1] > 0
                throw(Base.LinAlg.SingularException(info[1]))
            end
            C
        end
    end
end

#gebrd
for (bname, fname, elty, relty) in ((:cusolverDnSgebrd_bufferSize, :cusolverDnSgebrd, :Float32, :Float32),
                                    (:cusolverDnDgebrd_bufferSize, :cusolverDnDgebrd, :Float64, :Float64),
                                    (:cusolverDnCgebrd_bufferSize, :cusolverDnCgebrd, :Complex64, :Float32),
                                    (:cusolverDnZgebrd_bufferSize, :cusolverDnZgebrd, :Complex128, :Float64))
    @eval begin
        function gebrd!(A::CudaMatrix{$elty})
            m,n     = size(A)
            lda     = max(1,stride(A,2))
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                             cusolverDnhandle[1], m, n, bufSize))

            buffer  = CudaArray(zeros($elty,bufSize[1]))
            devinfo = CudaArray(zeros(Cint,1))
            D       = CudaArray(zeros($relty,min(m,n)))
            E       = CudaArray(zeros($relty,min(m,n)))
            TAUQ    = CudaArray(zeros($elty,min(m,n)))
            TAUP    = CudaArray(zeros($elty,min(m,n)))
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, Cint, Cint, Ptr{$elty},
                               Cint, Ptr{$relty}, Ptr{$relty}, Ptr{$elty},
                               Ptr{$elty}, Ptr{$elty}, Cint, Ptr{Cint}),
                              cusolverDnhandle[1], m, n, A, lda, D, E, TAUQ,
                              TAUP, buffer, bufSize[1], devinfo))
            info = to_host(devinfo)
            if info[1] < 0
                throw(ArgumentError("The $(info[1])th parameter is wrong"))
            end
            A, D, E, TAUQ, TAUP
        end
    end
end

for (bname, fname, elty, relty) in ((:cusolverDnSgesvd_bufferSize, :cusolverDnSgesvd, :Float32, :Float32),
                                    (:cusolverDnDgesvd_bufferSize, :cusolverDnDgesvd, :Float64, :Float64),
                                    (:cusolverDnCgesvd_bufferSize, :cusolverDnCgesvd, :Complex64, :Float32),
                                    (:cusolverDnZgesvd_bufferSize, :cusolverDnZgesvd, :Complex128, :Float64))
    @eval begin
        function gesvd!(jobu::BlasChar,
                        jobvt::BlasChar,
                        A::CudaMatrix{$elty})
            m,n     = size(A)
            lda     = max(1,stride(A,2))
            bufSize = Array(Cint,1)
            statuscheck(ccall(($(string(bname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, Cint, Cint, Ptr{Cint}),
                             cusolverDnhandle[1], m, n, bufSize))

            buffer  = CudaArray(zeros($elty, bufSize[1]))
            rbuffer = CudaArray(zeros($relty, 5 * min(m, n)))
            devinfo = CudaArray(zeros(Cint, 1))
            U       = CudaArray(zeros($elty, m, m))
            ldu     = max(1,stride(U,2))
            S       = CudaArray(zeros($relty, min(m, n)))
            Vt      = CudaArray(zeros($elty, n, n))
            ldvt    = max(1,stride(Vt,2))
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverDnHandle_t, BlasChar, BlasChar, Cint,
                               Cint, Ptr{$elty}, Cint, Ptr{$relty}, Ptr{$elty},
                               Cint, Ptr{$elty}, Cint, Ptr{$elty}, Cint,
                               Ptr{$relty}, Ptr{Cint}), cusolverDnhandle[1],
                              jobu, jobvt, m, n, A, lda, S, U, ldu, Vt, ldvt,
                              buffer, bufSize[1], rbuffer, devinfo))
            info = to_host(devinfo)
            if info[1] < 0
                throw(ArgumentError("The $(info[1])th parameter is wrong"))
            end
            U, S, Vt
        end
    end
end
