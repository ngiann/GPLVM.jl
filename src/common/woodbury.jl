"""
Calculates (K⁻¹ + Λ) where Λ is diagonal and `Λ½.^2 == Λ`.
Serves as a better mnemonic, for the directly called `woodbury_327` method.
"""
aux_invert_K⁻¹_plus_Λ(;K = K, Λroot = Λroot) = woodbury_327(;K = K, W½ = Λroot)


"""
Calculates (K + Λ⁻¹)⁻¹ where Λ is diagonal and `Λ½.^2 == Λ`.
Serves as a better mnemonic, for the directly called `woodbury_328` method.
"""
aux_invert_K_plus_Λ⁻¹(; K = K, Λroot = Λroot) = woodbury_328(;K = K, W½ = Λroot)



"""
    woodbury_327(;K = K, W½ = W½)

Calculates (K⁻¹ + W)⁻¹ where W is diagonal and `W½.^2 == W`.
The results should be equivalent to `inv(inv(K)+Diagonal(W½).^2)``.
See equations (3.26) and (3.27) in book "Gaussian Processes for Machine Learning" by R&W.
"""
function woodbury_327(;K = K, W½::Diagonal = W½)
    
    # sources:
    # see 145 in matrix coobook where we set A⁻¹=K, B=I, C = W½
    # see (A.9) in GPML 
    # The application of the woodbury identity is recommended in GPML, see (3.26) and (3.27).

    KW½ = K*W½

    B = Symmetric(I +  W½*KW½) # equation (3.26) in GPML.

    A = Symmetric(KW½ * (B \ KW½'))

    Symmetric(K - A)

end


"""
    woodbury_328(;K = K, W½ = W½)

Calculates (K + W⁻¹)⁻¹ where W is diagonal and `W½.^2 == W`.
The results should be equivalent to `inv(K + inv(Diagonal(W½).^2))``.
See equations (3.26) and (3.28) in book "Gaussian Processes for Machine Learning" by R&W.
"""
function woodbury_328(; K = K, W½::Diagonal = W½)
    
    KW½ = K*W½

    B = Symmetric(I +  W½*KW½) # equation (3.26) in GPML

    A = W½*(B\W½) # equation (3.28) in GPML.

    Symmetric(A)

end