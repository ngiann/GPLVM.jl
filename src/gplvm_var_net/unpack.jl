function unpack(::Val{:gplvmvarnet}, p, ::Missing, D, N, net, Q)

    nwts = ForwardNeuralNetworks.numweights(net)

    @assert(length(p) == Q*N + 2 + 1 + nwts + N)

    MARK = 0

    Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

    θ = exp.(p[MARK+1:MARK+2]); MARK += 2

    β = exp(p[MARK+1]); MARK += 1

    w = p[MARK+1:MARK+nwts]; MARK += nwts

    Λroot = Diagonal(reshape(p[MARK+1:MARK+N], N)); MARK += N

    @assert(MARK == length(p))

    μ = net(w, Z)

    return Z, θ, Fill(β, D, N), μ, Λroot

end


function unpack(::Val{:gplvmvarnet}, p, 𝛃, D, N, net, Q)

    nwts = ForwardNeuralNetworks.numweights(net)

    @assert(length(p) == Q*N + 2 + nwts + N)

    MARK = 0

    Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

    θ = exp.(p[MARK+1:MARK+2]); MARK += 2

    w = p[MARK+1:MARK+nwts]; MARK += nwts

    Λroot = Diagonal(reshape(p[MARK+1:MARK+N], N)); MARK += N

    @assert(MARK == length(p))

    μ = net(w, Z)

    return Z, θ, 𝛃, μ, Λroot

end
