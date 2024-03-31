function test(;seed = 1, numsamples = 1000)

    D = 2
    T = 2
    J = 2

    rg = MersenneTwister(seed)

    b = randn(rg)
    ν = randn(rg, T, D)
    C = randn(rg, T, T); C = C'C + 0.001*I
    m = randn(rg, T, D)
    A = randn(rg, T, T); A = A'A + 0.001*I
    σ = rand(rg, T, J)
    u = randn(rg, T, J)
    B = rand(rg, J, D)

    prior = MvNormal(zeros(T),C)

    # analytical
    aux = 0.0

    # FIRST TERM #
    for t in 1:T
        for j in 1:J

            local aux_inner = 0
            for d in 1:D
                aux_inner += B[j,d]*(ν[t,d]+b)
            end 

            aux += logpdf(Normal(aux_inner, σ[t,j]), u[t,j])

        end
    end

     # SECOND TERM #
     aux_second = 0
     for t in 1:T
        for j in 1:J

            local aux_inner = 0
            for d in 1:D
                aux_inner += B[j,d]^2*A[t,t]
            end

            aux_second += (1/(2*σ[t,j]^2))*aux_inner
        end
    end
    aux += -aux_second

    # THIRD TERM #
    
    for d in 1:D
        aux += logpdf(prior, m[:,d] - ν[:,d])
    end
    aux += - (D/2)*tr(C\A)




    ######## SAMPLING ########
    r = [MvNormal(ν[:,d], A) for d in 1:D]

    function sampleterm()

        local f = [rand(rg, r[d]) for d in 1:D]
        local aux = 0.0

        # first term
        for t in 1:T
            for j in 1:J

                local aux_inner = 0.0
                for d in 1:D
                    aux_inner += B[j,d]*(f[d][t]+b)
                end
    
                aux += logpdf(Normal(aux_inner, σ[t,j]), u[t,j])
    

            end
        end

        # second term
        for d in 1:D

            aux += logpdf(MvNormal(m[:,d], C),f[d])

        end

        return aux



    end

    empiricalmean = zeros(Threads.nthreads())
    Threads.@threads for _ in 1:numsamples
        empiricalmean[Threads.threadid()] += sampleterm()
    end
  
    aux, sum(empiricalmean)/numsamples

end