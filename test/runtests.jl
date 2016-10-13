using CUSOLVER
using Base.Test

@testset "dense" begin
    include("dense.jl")
end
@testset "sparse" begin
    include("sparse.jl")
end
