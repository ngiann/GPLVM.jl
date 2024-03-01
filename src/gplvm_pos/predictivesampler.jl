predictivesampler(xtest, R) = predictivesampler(xtest, R[:Z], R[:θ], R[:μ], R[:Σ], R[:K], R[:α], R[:b]; JITTER = R[:JITTER])

function predictivesampler(xtest, X, θ, μ, Σ, K, α, b; JITTER = 0.0) # maybe rename X to Z

    @assert(length(xtest) == size(X, 1))

    D = size(μ, 1)

    #---------------------------------------------------------
    # Cross-covariance
    #---------------------------------------------------------

    D_Xx = pairwise(SqEuclidean(), X, xtest)

    K_Xx  = covariance(D_Xx, θ)


    #---------------------------------------------------------
    # Covariance of test points
    #---------------------------------------------------------

    D_xx = pairwise(SqEuclidean(), xtest)

    K_xx = Symmetric(covariance(D_xx, θ) + JITTER*I)


    #---------------------------------------------------------
    # Posterior of latent function values
    #---------------------------------------------------------

    # use for convenience MatrixNormal(M, U, V) where 
    # M is n x p mean
    # U is n x n row covariance
    # V is p x p column covariance
    #
    # Mean μ is D × N and we have assume that the dimensions are independent.
    # Hence, we use the identity matrix for the row-covariance. 
  
    posterior = MatrixNormal(μ, Matrix(I, D, D), Σ) 


    #---------------------------------------------------------
    # Conditional variance of test points
    #---------------------------------------------------------

    V = K_xx - K_Xx' * (K \  K_Xx)

   
    #---------------------------------------------------------
    # Return function that samples from predictive
    #---------------------------------------------------------

    function drawsamplefrompredictive() 
        
        local sample = K_Xx' * (K \ rand(posterior)')  +  sqrt(V)*randn(1,D)
        #                                ↑                           ↑
        #                         mean prediction            fluctuating noise 
        #                                                    due to covariance
        exp.(α * sample .+ b)

    end
end