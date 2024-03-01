#-------------------------------------------
function unpack(::Val{:gplvmvarnet}, p, ::Missing, D, N, net, Q)
#-------------------------------------------

    nwts = ForwardNeuralNetworks.numweights(net)

    @assert(length(p) == Q*N + 2 + 1 + nwts + N)

    local MARK = 0

    local Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

    local θ = exp.(p[MARK+1:MARK+2]); MARK += 2

    local β = exp(p[MARK+1]); MARK += 1

    local w = p[MARK+1:MARK+nwts]; MARK += nwts

    local Λroot = Diagonal(reshape(p[MARK+1:MARK+N], N)); MARK += N

    @assert(MARK == length(p))

    local μ = net(w, Z)

    return Z, θ, Fill(β, D, N), μ, Λroot

end


#-------------------------------------------
function unpack(::Val{:gplvmvarnet}, p, 𝛃, D, N, net, Q)
#-------------------------------------------

    nwts = ForwardNeuralNetworks.numweights(net)

    @assert(length(p) == Q*N + 2 + nwts + N)

    local MARK = 0

    local Z = reshape(p[MARK+1:MARK+Q*N], Q, N); MARK += Q*N

    local θ = exp.(p[MARK+1:MARK+2]); MARK += 2

    local w = p[MARK+1:MARK+nwts]; MARK += nwts

    local Λroot = Diagonal(reshape(p[MARK+1:MARK+N], N)); MARK += N

    @assert(MARK == length(p))

    local μ = net(w, Z)

    return Z, θ, 𝛃, μ, Λroot

end
