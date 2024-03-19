function numerically_verify_gplvmvar(X, Z, θ, 𝛃, μ, Λroot, w, b, JITTER, η)

    tmp1 =        marginallikelihood_gplvmvar(X, Z, θ, 𝛃, μ, Λroot, w, b; JITTER = JITTER, η = η)
    
    tmp2 = marginallikelihood_VERIFY_gplvmvar(X, Z, θ, 𝛃, μ, Λroot, w, b; JITTER = JITTER, η = η)
    
    @printf("Verifiying calculations\n")
    @printf("First implementation delivers  %f\n", tmp1)
    @printf("Second implementation delivers %f\n", tmp2)
    @printf("difference is %f\n", tmp1-tmp2)

    nothing

end

    