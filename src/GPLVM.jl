module GPLVM

    using LinearAlgebra
    using Distributions, Random, Statistics
    using Distances
    using Optim, PyPlot#, BenchmarkTools#Lux, Functors
    using Printf, LineSearches, Zygote, ForwardNeuralNetworks
    using ExponentialExpectations


    include("toydata.jl")
    include("covariance.jl")
    include("gplvm.jl")
    # include("gplvmvar_lux.jl")
    include("gplvmvar_pos.jl")
    include("gplvmvar.jl")
    
    include("woodbury.jl")

    include("predict.jl")
    include("myskip.jl")
    include("marginallikelihood.jl")
    include("infertestlatent.jl")


    #---- code for debugging and verifying calculations ----
    include("marginallikelihood_verify.jl")
    
    export toydata, gplvmvar_pos, gplvmvar, gplvm

end
