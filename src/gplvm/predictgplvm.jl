predictgplvm(y, xtest, R) = predictgplvm(y, xtest, R[:Z], R[:θ], R[:noisy_K_chol], R[:b]; JITTER = R[:JITTER])

function predictgplvm(y, xtest, X, θ, noisy_K_chol, b; JITTER = 0.0)

    # noisy_K_chol is the lower cholesky decomposition of the kernel covariance matrix plus noise and jitter

    #---------------------------------------------------------
    # Calculate covariance matrices
    #---------------------------------------------------------

    # Cross-covariance between training and test points
    
    D_Xx = pairwise(SqEuclidean(), X, xtest)

    K_Xx  = covariance(D_Xx, θ)

    # Covariance between test points
   
    D_xx = pairwise(SqEuclidean(), xtest)

    K_xx = Symmetric(covariance(D_xx, θ) + JITTER*I)


    #---------------------------------------------------------
    # Predictive mean and covariance
    #---------------------------------------------------------

    # Note that inv(A)*B is equivalent to A\B and U'\(U\B)
    # where U = cholesky(A).L

    U = noisy_K_chol

    invK_mul_K_Xx = (U'\ (U \  K_Xx))

    μpred = K_Xx' * invK_mul_K_Xx * (y.-b) .+ b  # consult eq. 2.25 in GPML

    Σpred = K_xx - K_Xx' * invK_mul_K_Xx         # consult eq. 2.26 in GPML

    return μpred, Σpred

end