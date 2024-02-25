struct RBF{T}
    centres::T
    M::Int64
end

function RBF(M)

    RBF(collect(LinRange(-1.0, 1.0, M)), M)

end


distance(rbf::RBF, ζ) = pairwise(SqEuclidean(), reshape(ζ, 1, length(ζ)), reshape(rbf.centres, 1, rbf.M))

function (rbf::RBF)(ζ::Vector{T}, r) where T

    D² = distance(rbf::RBF, ζ)

    [exp.( -D² / (2*r*r)) ones(length(ζ))]

end

function (rbf::RBF)(ζ::Vector{T}, w, r) where T

    # @assert(length(w) == rbf.M+1)

    (rbf)(ζ, r)*w

end

numweights(rbf::RBF) = rbf.M + 1 # plus 1 for bias term
