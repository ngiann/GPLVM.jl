predictgplvmvarnet(xtest, R) = predictgplvmvarnet(xtest, R[:Z], R[:θ], R[:μ], R[:Λroot], R[:K]; JITTER = R[:JITTER])

function predictgplvmvarnet(xtest, X, θ, μ, Λroot, K; JITTER = JITTER)

    @assert(size(xtest, 1) == size(X, 1))

    #---------------------------------------------------------
    # Calculate covariance matrices
    #---------------------------------------------------------
    
    # Cross-covariance between training and testing
   
    D_Xx = pairwise(SqEuclidean(), X, xtest)

    K_Xx  = covariance(D_Xx, θ)

    # Covariance of test points
    
    D_xx = pairwise(SqEuclidean(), xtest)

    K_xx = Symmetric(covariance(D_xx, θ) + JITTER*I)


    #---------------------------------------------------------
    # Distribution of latent function values for test data
    #---------------------------------------------------------

    μpred = K_Xx' * (K \ μ')

    Σpred = K_xx - K_Xx' * aux_invert_K_plus_Λ⁻¹(K=K, Λroot=Λroot) * K_Xx

    return μpred, Σpred
    
end