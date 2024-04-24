function unpack_gplvmplus(p, D, N, net, Q)

    nwts = ForwardNeuralNetworks.numweights(net)
    
    @assert(length(p) == Q*N + 1 + 1 + nwts + N + 2 + N)

    MARK = 0

    Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

    θ = log1pexp(p[MARK+1]); MARK += 1

    β = log1pexp(p[MARK+1]) + 1; MARK += 1

    w = p[MARK+1:MARK+nwts]; MARK += nwts

    Λroot = Diagonal((p[MARK+1:MARK+N])); MARK += N

    α = log1pexp(p[MARK+1]); MARK += 1

    b = p[MARK+1]; MARK += 1

    c = log1pexp.(p[MARK+1:MARK+N]); MARK += N

    @assert(MARK == length(p))

    μ = net(w, Z)
    
    return Z, [1.0;θ], β, μ, Λroot, w, α, b, c

end