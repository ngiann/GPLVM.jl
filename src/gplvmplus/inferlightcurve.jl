inferlightcurve(tobs,U,B,S,R) = inferlightcurve_(tobs, U, B, S; R...)

function inferlightcurve_(tobs, U, B, S; μ = μ, Σ = Σ, K = K, η = η, Λroot = Λroot, net = net, w = w,
                             α = α, b = b, 𝛃 = 𝛃, Z = Z, θ = θ, JITTER = JITTER, rg = rg)

    # work out and verify dimensions
    D, N = size(μ); @assert(N == size(Z, 2) == size(Λroot, 1) == size(Λroot, 2))

    Q = size(Z, 1)

    J, T = size(U); @assert(size(B,1) == J); @assert(size(B, 2) == D); @assert(size(S) == size(U))

    # set up RBF network
    rbf  = GPLVM.RBF(10) ### ❗ note fixed number of basis functions in rbf network ❗
    nwts = numweights(rbf)
    ζ    = 2*((sort(tobs) .- minimum(tobs)) / (maximum(tobs) - minimum(tobs))) .- 1.0

    # precalculate

    invK_mul_μ = (K\μ')

    inv_of_K_plus_Λ⁻¹ = aux_invert_K_plus_Λ⁻¹(K=K, Λroot=Λroot)

    #--------------------------------------------------
    function unpack(p)
    #--------------------------------------------------

        local MARK = 0

        local rbfweights = reshape(p[MARK+1:MARK+nwts*Q], nwts, Q); MARK += Q*nwts
        
        local Lroot = Diagonal(p[MARK+1:MARK+T]); MARK += T

        local c = exp(p[MARK+1]); MARK += 1
        
        @assert(MARK == length(p)) # all parameters must be used up

        local Z₊ = rbf(ζ, rbfweights, 0.5)' ### ❗ note fixed width of rbf network ❗
       
        local ν = net(w, Z₊)

        return Z₊, ν, Lroot, c, rbfweights

    end

    

    #--------------------------------------------------
    function objective(Z₊, ν, Lroot, c, wrbf)
    #--------------------------------------------------

        # Calculate cross-covariance matrix between test and training inputs
        local K₊ = covariance(pairwise(SqEuclidean(), Z₊, Z), θ); @assert(size(K₊, 1) == T)

        # Calculate "self"-covariance matrix between test inputs
        local K₊₊ = Symmetric(covariance(pairwise(SqEuclidean(), Z₊), θ) + JITTER*I); @assert(size(K₊₊, 1) == T)
        
        # calculate mean of "prior" of test latent function values
        local m = (K₊*(invK_mul_μ))'
       
        @assert(size(m, 1) == D); @assert(size(m, 2) == T)

        # calculate covariance of "prior" of test latent function values
        local C = K₊₊ - K₊*inv_of_K_plus_Λ⁻¹*K₊'; @assert(size(C, 1) == T); @assert(size(C, 2) == T);


        # calculate posterior covariance of test latent function values
        local A = aux_invert_K⁻¹_plus_Λ(K=Symmetric(C+JITTER*I) , Λroot = Lroot)

        # log-prior contribution
        local ℓ = zero(eltype(Z₊))

        # local Cᵤ = cholesky(Symmetric(C)).L
        # ℓ += -0.5*D*T*log(2π) - 0.5*sum(abs2.(Cᵤ\(ν-m)')) - D*0.5*tr(C\A) - D*sum(log.(diag(Cᵤ)))

        ℓ +=  expectation_of_sum_D_log_prior_zero_mean(;K = Symmetric(C), μ = (ν-m), Σ = A)


        # log-likelihood contribution

        for t in 1:T, j in 1:J
            
            
            # local aux = 0.0
            # for d in 1:D
            #     aux += c * B[j,d] * ExponentialExpectations.E(a = α, μ = ν[d,t], σ = sqrt(A[t,t]),b = b)
            # end
            # ℓ += logpdf(Normal(aux, S[j,t]), U[j,t])
            
            # following two lines implement above commented out block - keep above for numerical verification

            Ef = exp.(α*ν[:,t] .+ α^2*A[t,t] / 2 .+  b)
            
            ℓ += logpdf(Normal(c*sum(B[j,:].*Ef), S[j,t]), U[j,t])

            # local aux_tr = 0.0
            # for d in 1:D
            #     aux_tr += c^2 * B[j,d]^2 * ExponentialExpectations.V(a = α, μ = ν[d,t], σ = sqrt(A[t,t]),b = b)
            # end
            # ℓ +=  (1 / (2*S[j,t]^2)) * aux_tr

            # following let block implements above commented out block - keep above for numerical verification
            local Vterm = let
                
                local Ef² = exp.(2*α*ν[:,t] .+ (2*α)^2*A[t,t] / 2 .+ 2b) 

                local V = Ef² .- Ef.^2 # this is V[X] = E[X²] - (E[X])² # There may be a computational gain to be had here

                c^2 * sum( B[j,:].^2 .* V)

            end
        

            ℓ +=  - (1 / (2*S[j,t]^2)) * Vterm

        end

        # entropy contribution with constants discarded
        ℓ += 0.5*D*logabsdet(A)[1] 

        # penalty on rbf weights - not in latex
        ℓ += - 0.5*η*sum(abs2.(wrbf)) - 0.5*η*sum(abs2.(Z₊)) 

        return ℓ
    end


    #-----------------------------------------------------------------
    # initialise parameters, define options, loss and gradient
    #-----------------------------------------------------------------
   
    p0 = [randn(rg, Q*nwts)*0.5; randn(rg, T); 0.0]

    opt = Optim.Options(iterations = 1000, show_trace = true, show_every = 1)

    objective(p) = -objective(unpack(p)...)

    function fg!(F, G, x)
            
        value, ∇f = Zygote.withgradient(objective,x)

        isnothing(G) || copyto!(G, ∇f[1])

        isnothing(F) || return value

        nothing

    end


    #-----------------------------------------------------------------
    # Carry out actual optimisation and obtain optimised parameters
    #-----------------------------------------------------------------

    @printf("Optimising %d number of parameters\n",length(p0))

    # p1 = let 

    #     local nopt = Optim.Options(iterations = 100, show_trace = true, show_every = 100)

    #     optimize(objective, p0, NelderMead(), nopt).minimizer

    # end


    @printf("Optimising %d number of parameters\n",length(p0))
    
    bnd = [[(-3,3) for _ in 1:Q*nwts]; [(-10,50) for _ in 1:T]; (-4,5)]
    p1 = best_candidate(bboptimize(objective, p0; Method=:generating_set_search, SearchRange = bnd, NumDimensions = length(p0), MaxFuncEvals = 200_000))

    results = optimize(objective, p1, LBFGS(), opt, autodiff=:forward).minimizer

    Zopt, νopt, Lroot = unpack(results)
   
    return Zopt

end