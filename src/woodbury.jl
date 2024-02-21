"""
    woodbury(;K = K, Λroot = Λroot)

Calculates (K⁻¹ + Λ)⁻¹ where Λ is diagonal and `Λroot.^2 == Λ`.
The results should be equivalent to `inv(inv(K)+Diagonal(Λroot).^2)``.
"""
function woodbury(;K = K, Λroot::Diagonal{T,Vector{T}} = Λroot) where T
    
    # sources:
    # see 145 in matrix coobook where we set A⁻¹=K, B=I, C = Λroot
    # see (A.9) in GPML 
    # The application of the woodbury identity is recommended in GPML, see (3.26) and (3.27).

    KΛroot = K*Λroot

    U = cholesky(Symmetric(I +  Λroot*KΛroot)).L

    A = U\KΛroot'

    Symmetric(K - A'A)

end