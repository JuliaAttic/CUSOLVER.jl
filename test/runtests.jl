using CUSOLVER
using Base.Test

@testset "CUSOLVER" begin

@testset "dense" begin
    include("dense.jl")
end

@testset "sparse" begin
    include("sparse.jl")
end

end
