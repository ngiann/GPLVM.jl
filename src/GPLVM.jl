module GPLVM

    using DifferentiationInterface
    using Distances
    using Distributions
    using JLD2
    using LinearAlgebra
    import Mooncake, Enzyme
    using Optim
    using Random

    include("gplvm.jl")
    include("loadoil.jl")

    export gplvm, loadoil

end
