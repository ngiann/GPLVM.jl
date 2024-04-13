function unpack_gplvmvar(p, _D, N, net, Q)

    nwts = ForwardNeuralNetworks.numweights(net)

    @assert(length(p) == Q*N + 3 + nwts + N + 1)

    MARK = 0

    Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

    θ = exp.(p[MARK+1:MARK+2]); MARK += 2

    β = exp(p[MARK+1]); MARK += 1

    w = p[MARK+1:MARK+nwts]; MARK += nwts

    Λroot = Diagonal(reshape(p[MARK+1:MARK+N], N)); MARK += N

    b = p[MARK+1]; MARK += 1

    @assert(MARK == length(p))

    μ = net(w, Z)

    return Z, θ, β, μ, Λroot, w, b

end
