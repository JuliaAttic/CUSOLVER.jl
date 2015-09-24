#enum cusolverStatus_t
#error messages from CUSOLVER

import CUBLAS: BlasChar, cublasfill, cublasop, cublasside, cublasFillMode_t, cublasOperation_t, cublasSideMode_t
import CUSPARSE: cusparseMatDescr_t

typealias cusolverStatus_t UInt32
const CUSOLVER_STATUS_SUCCESS                   = 0
const CUSOLVER_STATUS_NOT_INITIALIZED           = 1
const CUSOLVER_STATUS_ALLOC_FAILED              = 2
const CUSOLVER_STATUS_INVALID_VALUE             = 3
const CUSOLVER_STATUS_ARCH_MISMATCH             = 4
const CUSOLVER_STATUS_EXECUTION_FAILED          = 5
const CUSOLVER_STATUS_INTERNAL_ERROR            = 6
const CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 7

# refactorization types

typealias cusolverRfNumericBoostReport_t UInt32
const CUSOLVER_NUMERIC_BOOST_NOT_USED           = 0
const CUSOLVER_NUMERIC_BOOST_USED               = 1

typealias cusolverRfResetValuesFastMode_t UInt32
const CUSOLVER_RESET_VALUES_FAST_MODE_OFF       = 0
const CUSOLVER_RESET_VALUES_FAST_MODE_ON        = 1

typealias cusolverRfFactorization_t UInt32
const CUSOLVER_FACTORIZATION_ALG0               = 0
const CUSOLVER_FACTORIZATION_ALG1               = 1
const CUSOLVER_FACTORIZATION_ALG2               = 2

typealias cusolverRfTriangularSolve_t UInt32
const CUSOLVER_TRIANGULAR_SOLVE_ALG0            = 0
const CUSOLVER_TRIANGULAR_SOLVE_ALG1            = 1
const CUSOLVER_TRIANGULAR_SOLVE_ALG2            = 2
const CUSOLVER_TRIANGULAR_SOLVE_ALG3            = 3

typealias cusolverRfUnitDiagonal_t UInt32
const CUSOLVER_UNIT_DIAGONAL_STORED_L           = 0
const CUSOLVER_UNIT_DIAGONAL_STORED_U           = 1
const CUSOLVER_UNIT_DIAGONAL_ASSUMED_L          = 2
const CUSOLVER_UNIT_DIAGONAL_ASSUMED_U          = 3

typealias cusolverDnContext Void
typealias cusolverDnHandle_t Ptr{cusolverDnContext}
typealias cusolverSpContext Void
typealias cusolverSpHandle_t Ptr{cusolverSpContext}
typealias cusolverRfContext Void
typealias cusolverRfHandle_t Ptr{cusolverRfContext}

#complex numbers

typealias cuComplex Complex{Float32}
typealias cuDoubleComplex Complex{Float64}

typealias CusolverFloat Union{Float64,Float32,Complex128,Complex64}
typealias CusolverReal Union{Float64,Float32}
typealias CusolverComplex Union{Complex128,Complex64}
