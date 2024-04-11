function numerically_verify_gplvmplus(X, Z, θ, β, μ, Λroot,  w, α, b, JITTER, η)

    tmp1 =        marginallikelihood_gplvmplus(X, Z, θ, β, μ, Λroot, w, α, b; JITTER = JITTER, η = η)
    
    tmp2 = marginallikelihood_VERIFY_gplvmplus(X, Z, θ, β, μ, Λroot, w, α, b; JITTER = JITTER, η = η)
    
    @printf("Verifiying calculations\n")
    @printf("First implementation delivers  %f\n", tmp1)
    @printf("Second implementation delivers %f\n", tmp2)
    @printf("difference is %f\n", tmp1-tmp2)

    nothing

end

    