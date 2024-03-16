predictgplvm(xtest, R) = predictgplvm(xtest, R[:X], R[:Z], R[:θ], R[:K], R[:b], R[:idx]; JITTER = R[:JITTER])

function predictgplvm(xtest, Y, X, θ, K, b, idx; JITTER = 0.0)

    # matrix K includes observation noise
 
    D, N = size(Y); @assert(N == size(K,1) == size(K,2))

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
    # Predictive mean and covariance per dimension
    #---------------------------------------------------------

    # Note that inv(A)*B is equivalent to A\B and U'\(U\B)
    # where U = cholesky(A).L

    μpred = map(1:D) do d

        Kd = @views K[idx[d], idx[d]]

        Yd = @views vec(Y[d,idx[d]])
        
        K_Xx_d = @views K_Xx[idx[d],:]

        K_Xx_d'*(Kd\(Yd.-b)) .+ b # consult eq. 2.25 in GPML

    end

    Σpred = map(1:D) do d

        Kd = @views K[idx[d], idx[d]]

        K_Xx_d = @views K_Xx[idx[d],:]

        K_xx - K_Xx_d' * (Kd \ K_Xx_d)  # consult eq. 2.26 in GPML

    end
          

    return reduce(hcat, μpred), reduce(hcat, Σpred)

end