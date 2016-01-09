#csrissymHost 
function issym(A::SparseMatrixCSC, inda::SparseChar='O')
    cuinda = cusparseindex(inda)
    m = size(A,1)
    if size(A,2) != m
        throw(DimensionMismatch("issym is only possible for square matrices!"))
    end
    issym = Ref{Cint}(0)
    cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
    endPtr = convert(Vector{Cint},A.colptr[2:end] - 1)
    println("\t",endPtr, " ", length(A.nzval), " ", m, " ",length(A.rowval))
    statuscheck(ccall((:cusolverSpXcsrissymHost,libcusolver), cusolverStatus_t,
                      (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                       Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                      cusolverSphandle[1], m, length(A.nzval), cudesca, convert(Vector{Cint},A.colptr),
                      endPtr, convert(Vector{Cint},A.rowval), issym))
    return issym == 1
end

#csrlsvlu 
for (fname, elty, relty) in ((:cusolverSpScsrlsvluHost, :Float32, :Float32),
                             (:cusolverSpDcsrlsvluHost, :Float64, :Float64),
                             (:cusolverSpCcsrlsvluHost, :Complex64, :Float32),
                             (:cusolverSpZcsrlsvluHost, :Complex128, Float64))
    @eval begin
        function csrlsvlu!(A::CudaSparseMatrixCSR{$elty},
                           b::CudaVector{$elty},
                           x::CudaVector{$elty},
                           tol::$relty,
                           reorder::Cint,
                           inda::SparseChar)
            cuinda = cusparseindex(inda)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("LU factorization is only possible for square matrices!"))
            end

            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            singularity = zeros(Cint,1)
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                               $relty, Cint, Ptr{$elty}, Ptr{Cint}),
                              cusolverSphandle[1], n, A.nnz, cudesca, A.nzVal,
                              A.rowPtr, A.colVal, b, tol, reorder, x, singularity))
            if singularity[1] != -1
                throw(Base.LinAlg.SingularException(singularity[1]))
            end
            x
        end
    end
end

#csrlsvqr 
for (fname, elty, relty) in ((:cusolverSpScsrlsvqr, :Float32, :Float32),
                             (:cusolverSpDcsrlsvqr, :Float64, :Float64),
                             (:cusolverSpCcsrlsvqr, :Complex64, :Float32),
                             (:cusolverSpZcsrlsvqr, :Complex128, Float64))
    @eval begin
        function csrlsvqr!(A::CudaSparseMatrixCSR{$elty},
                           b::CudaVector{$elty},
                           x::CudaVector{$elty},
                           tol::$relty,
                           reorder::Cint,
                           inda::SparseChar)
            cuinda = cusparseindex(inda)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("QR factorization is only possible for square matrices!"))
            end
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            singularity = Array(Cint,1)
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                               $relty, Cint, Ptr{$elty}, Ptr{Cint}),
                              cusolverSphandle[1], n, A.nnz, cudesca, A.nzVal,
                              A.rowPtr, A.colVal, b, tol, reorder, x, singularity))
            if singularity[1] != -1
                throw(Base.LinAlg.SingularException(singularity[1]))
            end
            x
        end
    end
end

#csrlsvchol
for (fname, elty, relty) in ((:cusolverSpScsrlsvcholHost, :Float32, :Float32),
                             (:cusolverSpDcsrlsvcholHost, :Float64, :Float64),
                             (:cusolverSpCcsrlsvcholHost, :Complex64, :Float32),
                             (:cusolverSpZcsrlsvcholHost, :Complex128, Float64))
    @eval begin
        function csrlsvchol!(A::CudaSparseMatrixCSR{$elty},
                           b::CudaVector{$elty},
                           x::CudaVector{$elty},
                           tol::$relty,
                           reorder::Cint,
                           inda::SparseChar)
            cuinda = cusparseindex(inda)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("Cholesky factorization is only possible for square matrices!"))
            end
            if length(b) != n
                throw(DimensionMismatch("dimensions of A, $n, and b, $(length(b)), must match."))
            end
            if length(x) != n
                throw(DimensionMismatch("dimensions of A, $n, and x, $(length(x)), must match."))
            end

            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            singularity = zeros(Cint,1)
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint, cusparseMatDescr_t,
                               Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                               $relty, Cint, Ptr{$elty}, Ptr{Cint}),
                              cusolverSphandle[1], n, A.nnz, cudesca, A.nzVal,
                              A.rowPtr, A.colVal, b, tol, reorder, x, singularity))
            if singularity[1] != -1
                throw(Base.LinAlg.SingularException(singularity[1]))
            end
            x
        end
    end
end

#csrlsqvqr 
for (fname, elty, relty) in ((:cusolverSpScsrlsqvqrHost, :Float32, :Float32),
                             (:cusolverSpDcsrlsqvqrHost, :Float64, :Float64),
                             (:cusolverSpCcsrlsqvqrHost, :Complex64, :Float32),
                             (:cusolverSpZcsrlsqvqrHost, :Complex128, Float64))
    @eval begin
        function csrlsqvqr!(A::CudaSparseMatrixCSR{$elty},
                            b::CudaVector{$elty},
                            x::CudaVector{$elty},
                            tol::$relty,
                            inda::SparseChar)
            cuinda = cusparseindex(inda)
            m,n = size(A)
            if m < n
                throw(ArgumentError("csrlsqvqr only works when the first dimension of A, $m, is greater than or equal to the second dimension of A, $n"))
            end
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            p = CudaArray(zeros(Cint,n))
            min_norm = zeros($relty,1)
            rankA = zeros(Cint,1)
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint, Cint,
                               cusparseMatDescr_t, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, Ptr{$elty}, $relty, Ptr{Cint},
                               Ptr{$elty}, Ptr{Cint}, Ptr{$relty}),
                              cusolverSphandle[1], m, n, A.nnz, cudesca, A.nzVal,
                              A.rowPtr, A.colVal, b, tol, rankA, x, p, min_norm))
            x, rankA[1], p, min_norm[1]
        end
    end
end

#csreigvsi 
for (fname, elty, relty) in ((:cusolverSpScsreigvsi, :Float32, :Float32),
                             (:cusolverSpDcsreigvsi, :Float64, :Float64),
                             (:cusolverSpCcsreigvsi, :Complex64, :Float32),
                             (:cusolverSpZcsreigvsi, :Complex128, Float64))
    @eval begin
        function csreigvsi(A::CudaSparseMatrixCSR{$elty},
                           μ_0::$elty,
                           x_0::CudaVector{$elty},
                           tol::$relty,
                           maxite::Cint,
                           inda::SparseChar)
            cuinda = cusparseindex(inda)
            m,n = size(A)
            if m != n
                throw(DimensionMismatch("A must be square!"))
            end
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            x = copy(x_0) 
            μ = CudaArray(zeros($elty,1)) 
            println($(string(fname)))
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint,
                               cusparseMatDescr_t, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, $elty, Ptr{$elty}, Cint,
                               $relty, Ptr{$elty}, Ptr{$elty}),
                              cusolverSphandle[1], n, A.nnz, cudesca, A.nzVal,
                              A.rowPtr, A.colVal, μ_0, x_0, maxite, tol, μ, x))
            μ[1], x
        end
    end
end

#csreigs
for (fname, elty, relty) in ((:cusolverSpScsreigsHost, :Complex64, :Float32),
                             (:cusolverSpDcsreigsHost, :Complex128, :Float64),
                             (:cusolverSpCcsreigsHost, :Complex64, :Complex64),
                             (:cusolverSpZcsreigsHost, :Complex128, :Complex128))
    @eval begin
        function csreigs(A::CudaSparseMatrixCSR{$relty},
                         lbc::$elty,
                         ruc::$elty,
                         inda::SparseChar)
            cuinda = cusparseindex(inda)
            m,n = size(A)
            if m != n
                throw(DimensionMismatch("A must be square!"))
            end
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            numeigs = zeros(Cint,1)
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint,
                               cusparseMatDescr_t, Ptr{$relty}, Ptr{Cint},
                               Ptr{Cint}, $elty, $elty, Ptr{Cint}),
                              cusolverSphandle[1], n, A.nnz, cudesca, A.nzVal,
                              A.rowPtr, A.colVal, lbc, ruc, numeigs))
            numeigs[1]
        end
    end
end
