#csrlsvlu 
for (fname, elty, relty) in ((:cusolverSpScsrlsvluHost, :Float32, :Float32),
                             (:cusolverSpDcsrlsvluHost, :Float64, :Float64),
                             (:cusolverSpCcsrlsvluHost, :Complex64, :Float32),
                             (:cusolverSpZcsrlsvluHost, :Complex128, Float64))
    @eval begin
        function csrlsvlu!(A::SparseMatrixCSC{$elty},
                           b::Vector{$elty},
                           x::Vector{$elty},
                           tol::$relty,
                           reorder::Cint,
                           inda::SparseChar)
            cuinda = cusparseindex(inda)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("LU factorization is only possible for square matrices!"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end
            Mat = transpose(A)
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            singularity = zeros(Cint,1)
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, Ptr{$elty}, $relty, Cint, Ptr{$elty},
                               Ptr{Cint}),
                              cusolverSphandle[1], n, length(A.nzval), &cudesca,
                              Mat.nzval, convert(Vector{Cint},Mat.colptr),
                              convert(Vector{Cint},Mat.rowval), b, tol, reorder,
                              x, singularity))
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
                           b::CuVector{$elty},
                           x::CuVector{$elty},
                           tol::$relty,
                           reorder::Cint,
                           inda::SparseChar)
            cuinda = cusparseindex(inda)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("QR factorization is only possible for square matrices!"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            singularity = Array{Cint}(1)
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, Ptr{$elty}, $relty, Cint, Ptr{$elty},
                               Ptr{Cint}),
                              cusolverSphandle[1], n, A.nnz, &cudesca, A.nzVal,
                              A.rowPtr, A.colVal, b, tol, reorder, x, singularity))
            if singularity[1] != -1
                throw(Base.LinAlg.SingularException(singularity[1]))
            end
            x
        end
    end
end

#csrlsvchol
for (fname, elty, relty) in ((:cusolverSpScsrlsvchol, :Float32, :Float32),
                             (:cusolverSpDcsrlsvchol, :Float64, :Float64),
                             (:cusolverSpCcsrlsvchol, :Complex64, :Float32),
                             (:cusolverSpZcsrlsvchol, :Complex128, Float64))
    @eval begin
        function csrlsvchol!(A::CudaSparseMatrixCSR{$elty},
                             b::CuVector{$elty},
                             x::CuVector{$elty},
                             tol::$relty,
                             reorder::Cint,
                             inda::SparseChar)
            cuinda = cusparseindex(inda)
            n      = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("Cholesky factorization is only possible for square matrices!"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end

            cudesca     = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            singularity = zeros(Cint,1)
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, Ptr{$elty}, $relty, Cint, Ptr{$elty},
                               Ptr{Cint}),
                              cusolverSphandle[1], n, A.nnz, &cudesca, A.nzVal,
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
        function csrlsqvqr!(A::SparseMatrixCSC{$elty},
                            b::Vector{$elty},
                            x::Vector{$elty},
                            tol::$relty,
                            inda::SparseChar)
            cuinda = cusparseindex(inda)
            m,n    = size(A)
            if m < n
                throw(DimensionMismatch("csrlsqvqr only works when the first dimension of A, $m, is greater than or equal to the second dimension of A, $n"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end
            cudesca  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            p        = zeros(Cint,n)
            min_norm = zeros($relty,1)
            rankA    = zeros(Cint,1)
            Mat      = transpose(A)
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, Ptr{$elty}, $relty, Ptr{Cint},
                               Ptr{$elty}, Ptr{Cint}, Ptr{$relty}),
                              cusolverSphandle[1], m, n, length(A.nzval),
                              &cudesca, Mat.nzval, convert(Vector{Cint},Mat.colptr),
                              convert(Vector{Cint},Mat.rowval), b,
                              tol, rankA, x, p, min_norm))
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
                           x_0::CuVector{$elty},
                           tol::$relty,
                           maxite::Cint,
                           inda::SparseChar)
            cuinda = cusparseindex(inda)
            m,n    = size(A)
            if m != n
                throw(DimensionMismatch("A must be square!"))
            end
            if n != length(x_0)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of x_0, $(length(x_0))"))
            end
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            x       = copy(x_0) 
            μ       = CuArray(zeros($elty,1)) 
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, $elty, Ptr{$elty}, Cint,
                               $relty, Ptr{$elty}, Ptr{$elty}),
                              cusolverSphandle[1], n, A.nnz, &cudesca, A.nzVal,
                              A.rowPtr, A.colVal, μ_0, x_0, maxite, tol, μ, x))
            collect(μ)[1], x
        end
    end
end

#csreigs
for (fname, elty, relty) in ((:cusolverSpScsreigsHost, :Complex64, :Float32),
                             (:cusolverSpDcsreigsHost, :Complex128, :Float64),
                             (:cusolverSpCcsreigsHost, :Complex64, :Complex64),
                             (:cusolverSpZcsreigsHost, :Complex128, :Complex128))
    @eval begin
        function csreigs(A::SparseMatrixCSC{$relty},
                         lbc::$elty,
                         ruc::$elty,
                         inda::SparseChar)
            cuinda = cusparseindex(inda)
            m,n    = size(A)
            if m != n
                throw(DimensionMismatch("A must be square!"))
            end
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            numeigs = zeros(Cint,1)
            Mat     = A.'
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$relty}, Ptr{Cint},
                               Ptr{Cint}, $elty, $elty, Ptr{Cint}),
                              cusolverSphandle[1], n, length(A.nzval), &cudesca,
                              Mat.nzval, convert(Vector{Cint},Mat.colptr),
                              convert(Vector{Cint},Mat.rowval), lbc, ruc, numeigs))
            numeigs[1]
        end
    end
end
