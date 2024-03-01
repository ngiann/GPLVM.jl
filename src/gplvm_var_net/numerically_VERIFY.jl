function numerically_verify(T::Val{:gplvmvarnet}, X, Z, θ, 𝛃, μ, Λroot, JITTER, η)

    tmp1 =        marginallikelihood(T, X, Z, θ, 𝛃, μ, Λroot; JITTER = JITTER, η = η)
    
    tmp2 = marginallikelihood_VERIFY(T, X, Z, θ, 𝛃, μ, Λroot; JITTER = JITTER, η = η)
    
    @printf("Verifiying calculations\n")
    @printf("First implementation delivers  %f\n", tmp1)
    @printf("Second implementation delivers %f\n", tmp2)
    @printf("difference is %f\n", tmp1-tmp2)

    nothing

end

    