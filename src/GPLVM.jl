module GPLVM

    using DifferentiationInterface
    using Distances
    using Distributions
    using JLD2
    using KernelFunctions
    using LinearAlgebra
    import Mooncake, Enzyme, FiniteDiff, FiniteDifferences
    using Optim
    using Random

    include("gplvm.jl")
    include("loadoil.jl")
    # include("gp.jl")
    include("gp2.jl")

    export gplvm, loadoil, gp2

end
