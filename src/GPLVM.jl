module GPLVM

    using LinearAlgebra, Distributions, Random, Statistics, Distances
    using Optim, BlackBoxOptim, Zygote
    using ExponentialExpectations, FillArrays
    using ForwardNeuralNetworks
    using Printf, PyPlot
    
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
    include("gplvm/unpack.jl")

    export gplvm

    # variational GPLVM with auxiliary network modelling variational parameters 
    include("gplvm_var_net/gplvmvarnet.jl")
    include("gplvm_var_net/marginallikelihood.jl")
    include("gplvm_var_net/marginallikelihood_VERIFY.jl")
    include("gplvm_var_net/predictgplvmvarnet.jl")
    include("gplvm_var_net/numerically_VERIFY.jl")
    include("gplvm_var_net/unpack.jl")

    # warped GPLVM
    include("warpedgplvm.jl")

    export warpedgplvm
    
    # GPLVM₊
    include("gplvm_pos/gplvmvar_pos.jl")
    include("gplvm_pos/predictivesampler.jl")
    include("gplvm_pos/marginallikelihood.jl")
    include("gplvm_pos/marginallikelihood_VERIFY.jl")
    include("gplvm_pos/infertestlatent.jl")
    include("gplvm_pos/infertestlatent_photo_rbf.jl")
    include("gplvm_pos/infertestlatent_photo.jl")
    include("gplvm_pos/numerically_VERIFY.jl")
    include("gplvm_pos/unpack.jl")

   
    export gplvmvar_pos,  gplvmvarnet
    export inferlatent, infertestlatent_photo, inferlatent_photo_rbf, predictivesampler, predictgplvm, predictgplvmvarnet

end
