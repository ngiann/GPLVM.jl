module GPLVM

    using LinearAlgebra, Distributions, Random, Statistics, Distances
    using Optim, BlackBoxOptim, Zygote
    using ExponentialExpectations, FillArrays
    using ForwardNeuralNetworks
    using Printf, PyPlot
    using JLD2
    using Artifacts, LazyArtifacts
    
    # common
    include("common/covariance.jl" )
    include("common/woodbury.jl")
    include("common/rbf.jl")
    include("common/myskip.jl")
    include("common/inverterrors.jl")
    include("common/toydata.jl")
    include("common/expectation_latent_function_values.jl")
    include("common/expectation_of_sum_D_log_prior_zero_mean.jl")
    include("common/entropy.jl")
    include("common/getfg!.jl")

    export toydata


    # standard GPLVM
    include("gplvm/gplvm.jl")
    include("gplvm/predictgplvm.jl")
    include("gplvm/unpackgplvm.jl")
    include("gplvm/inferlatentgplvm.jl")

    export gplvm, inferlatentgplvm

    # variational GPLVM with auxiliary network modelling variational parameters 
    include("gplvmvar/gplvmvar.jl")
    include("gplvmvar/marginallikelihood_gplvmvar.jl")
    include("gplvmvar/marginallikelihood_VERIFY_gplvmvar.jl")
    include("gplvmvar/predictgplvmvar.jl")
    include("gplvmvar/numerically_VERIFY_gplvmvar.jl")
    include("gplvmvar/unpack.jl")
    include("gplvmvar/inferlatentgplvmvar.jl")

    export predictgplvmvar, inferlatentgplvmvar

    # warped GPLVM
    include("warpedgplvm.jl")

    export warpedgplvm
    
    # GPLVM₊
    include("gplvmplus/gplvmplus.jl")
    include("gplvmplus/predictivesampler.jl")
    include("gplvmplus/marginallikelihood.jl")
    include("gplvmplus/marginallikelihood_VERIFY.jl")
    include("gplvmplus/infertestlatent.jl")
    include("gplvmplus/inferlightcurve.jl")
    include("gplvmplus/infertestlatent_photo.jl")
    include("gplvmplus/numerically_VERIFY.jl")
    include("gplvmplus/unpack_gplvmplus.jl")
    include("gplvmplus/unpack_inferlatent_gplvmplus.jl")
    include("gplvmplus/partial_objective.jl")

    # saved model for GPLVM₊
    include("loadbossmodel.jl")

   
    export gplvmplus,  gplvmvar
    export inferlatent, infertestlatent_photo, inferlightcurve, predictivesampler, predictgplvm
    export loadbossmodel
end
