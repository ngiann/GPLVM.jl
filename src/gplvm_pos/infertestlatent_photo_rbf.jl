inferlatent_photo_rbf(U,B,S,R) = inferlatent_photo_rbf(U, B, S; R...)

function inferlatent_photo_rbf(U, B, S; μ = μ, Σ = Σ, K = K, η = η, Λroot = Λroot, net = net, w = w,
                             α = α, b = b, β = β, Z = Z, θ = θ, JITTER = JITTER, rg = rg)

    # work out and verify dimensions
    D, N = size(μ); @assert(N == size(Z, 2) == size(Λroot, 1) == size(Λroot, 2))

    Q = size(Z, 1)

    J, T = size(U); @assert(size(B,1) == J); @assert(size(B, 2) == D); @assert(size(S) == size(U))

    # set up RBF network
    rbf  = GPLVM.RBF(10) ### ❗ note fixed number of basis functions in rbf network ❗
    nwts = numweights(rbf)
    ζ    = collect(LinRange(-1.0, 1.0, T))

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

        return Z₊, ν, Lroot, c, w

    end

count = 0; ℓbest = -Inf; zbest = zeros(Q, T)

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

        local Cᵤ = cholesky(Symmetric(C)).L
        
        ℓ += -0.5*D*T*log(2π) - 0.5*sum(abs2.(Cᵤ\(ν-m)')) - D*0.5*tr(C\A) - D*sum(log.(diag(Cᵤ)))

        # code below implements line above - keep for numerical verification
        # let
        #     ℓ1 = 0
        #     for d in 1:D
        #         ℓ1 += logpdf(MvNormal(ν[d,:], Symmetric(C)), m[d,:]) - 0.5*tr(C\A)
        #     end
        # end

        # log-likelihood contribution

        for t in 1:T, j in 1:J
            
            # local aux = zero(eltype(ν))
            # for d in 1:D
            #     aux += c * B[j,d] * E(a = α, μ = ν[d,t], σ = sqrt(A[t,t]),b = b)
            # end
            
            # line below implements commented-out code above
            auxE = c * sum( B[j,:] .* exp.(α*ν[:,t]   .+     α^2*A[t,t] / 2 .+  b) )

            ℓ += logpdf(Normal(auxE, S[j,t]), U[j,t])

            # local aux_tr = zero(eltype(ν))
            # for d in 1:D
            #     aux_tr += c^2 * B[j,d]^2 * V(a = α, μ = ν[d,t], σ = sqrt(A[t,t]),b = b)
            # end

            # line below implements commented-out code above
            aux_V = c^2 * sum( B[j,:].^2 .* (exp.(2*α*ν[:,t] .+ (2*α)^2*A[t,t] / 2 .+ 2b) .- (exp.(α*ν[:,t]   .+     α^2*A[t,t] / 2 .+  b)).^2) )

        
            ℓ +=  (1 / (2*S[j,t]^2)) * aux_V

        end

        # entropy contribution with constants discarded
        ℓ += 0.5*D*logabsdet(A)[1] 

        # penalty on rbf weights - not in latex
        ℓ += - 0.5*η*sum(abs2.(wrbf)) - 0.5*η*sum(abs2.(Z₊)) 
        
        if ℓ > ℓbest
            ℓbest = ℓ
            zbest = copy(Z₊)
        end
        let 
            count += 1
            if mod(count, 1000) == 1
                figure(2); cla()
                plot(Z[1,:], Z[2,:], "b.")
                plot(zbest[1,:], zbest[2,:],"-ro")
                pause(0.01)
        
            end
        end

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
    results = best_candidate(bboptimize(objective, p0; SearchRange = bnd, NumDimensions = length(p0), MaxFuncEvals = 200_000))

    # results = optimize(Optim.only_fg!(fg!), p0, LBFGS(), opt).minimizer # alphaguess = InitialQuadratic(α0=1e-8)

    Zopt, νopt, Lroot = unpack(results)
   
    return Zopt

end