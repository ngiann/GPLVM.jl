"""
    woodbury_327(;K = K, Λroot = Λroot)

Calculates (K⁻¹ + Λ)⁻¹ where Λ is diagonal and `Λroot.^2 == Λ`.
The results should be equivalent to `inv(inv(K)+Diagonal(Λroot).^2)``.
"""
function woodbury_327(;K = K, Λroot::Diagonal{T,Vector{T}} = Λroot) where T
    
    # sources:
    # see 145 in matrix coobook where we set A⁻¹=K, B=I, C = Λroot
    # see (A.9) in GPML 
    # The application of the woodbury identity is recommended in GPML, see (3.26) and (3.27).

    KΛroot = K*Λroot

    U = cholesky(Symmetric(I +  Λroot*KΛroot)).L

    A = U\KΛroot'

    Symmetric(K - A'A)

end



"""
    woodbury_328(;K = K, Λroot = Λroot)

Calculates (K + Λ⁻¹)⁻¹ where Λ is diagonal and `Λroot.^2 == Λ`.
The results should be equivalent to `inv(K + inv(Diagonal(Λroot).^2))``.
"""
function woodbury_328(;K = K, Λroot::Diagonal{T,Vector{T}} = Λroot) where T
    
    # sources:
    # The application of the woodbury identity is recommended in GPML, see (3.28).

    KΛroot = K*Λroot

    U = cholesky(Symmetric(I +  Λroot*KΛroot)).L # inside cholesky is matrix B of equation (3.28) in GPML.

    A = U\Λroot'

    Symmetric(A'A)

end