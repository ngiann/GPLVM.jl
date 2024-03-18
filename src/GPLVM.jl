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

    export toydata


    # standard GPLVM
    include("gplvm/gplvm.jl")
    include("gplvm/predictgplvm.jl")
    include("gplvm/unpackgplvm.jl")
    include("gplvm/inferlatentgplvm.jl")

    export gplvm, inferlatentgplvm

    # variational GPLVM with auxiliary network modelling variational parameters 
    include("gplvm_var_net/gplvmvarnet.jl")
    include("gplvm_var_net/marginallikelihood.jl")
    include("gplvm_var_net/marginallikelihood_VERIFY.jl")
    include("gplvm_var_net/predictgplvmvarnet.jl")
    include("gplvm_var_net/numerically_VERIFY.jl")
    include("gplvm_var_net/unpack.jl")
    include("gplvm_var_net/inferlatentgplvmvarnet.jl")

    export inferlatentgplvmvarnet

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

    # saved model for GPLVM₊
    include("loadbossmodel.jl")

   
    export gplvmplus,  gplvmvarnet
    export inferlatent, infertestlatent_photo, inferlightcurve, predictivesampler, predictgplvm, predictgplvmvarnet
    export loadbossmodel
end
