function unpack_gplvmplus(p, D, N, net, Q)

    nwts = ForwardNeuralNetworks.numweights(net)
    
    @assert(length(p) == Q*N + 2 + 1 + nwts + N + 2)

    MARK = 0

    Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

    θ = exp.(p[MARK+1:MARK+2]); MARK += 2

    β = exp(p[MARK+1]); MARK += 1

    w = p[MARK+1:MARK+nwts]; MARK += nwts

    Λroot = Diagonal(p[MARK+1:MARK+N]); MARK += N

    α = exp(p[MARK+1]); MARK += 1

    b = p[MARK+1]; MARK += 1

    @assert(MARK == length(p))

    μ = net(w, Z)
    
    return Z, θ, β, μ, Λroot, w, α, b

end