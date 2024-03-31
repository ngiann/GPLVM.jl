# GPLVM

[![Build Status](https://github.com/ngiann/GPLVM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ngiann/GPLVM.jl/actions/workflows/CI.yml?query=branch%3Amain)



# Recover Duck image from measurements

```
using JLD2, GPLVM, PyPlot, Statistics

X=JLD2.load("/home/nikos/Dropbox/DATA/DUCKS/Ducks_72_128x128.jld2")["X"];
Y = reduce(hcat, [vec(reshape(x,128,128)[1:5:end,1:5:end]) for x in eachcol(X)]);

let # WARMUP!
    gplvmvar(Y; iterations=10, seed=21, Q=2, H1 = 50);
end

R3 = gplvmvar(Y; iterations=10_000, seed=21, Q=2, H1 = 50);

B = randn(10,676); # sensing matrix, does 10 measurements
S = 1*ones(10,1);  # simulated standard deviation
U = B*Y[:,72] + S.*randn(10,1) # simulate noisy measurements

zinfer = inferlatentgplvmvar_photo(U',B,S',R3;iterations=3000,seed=1,repeats=30); # infer latent coordinate from measurements

mpred3 = vec(predictgplvmvar(reshape(zinfer,2,1),R3)[1]); # make prediction

pcolor(reshape(mpred3,26,26))

%%%%%%%%%%%%%%%%%%%%
R2=gplvmplus(Y; iterations=10_000, seed=54, Q=2, H1 = 50);
zinfer = infertestlatent_photo(U,B,S,R2);
mpred2 = mean([vec(predictivesampler(reshape(zinfer,2,1),R2)()') for i in 1:100]);
figure()
pcolor(reshape(mpred2,26,26))
```

# Recover spectrum from measurements

```
using Revise, GPLVM, PyPlot, Statistics, GPLVMplus_experiments, PPCASpectra

R2 = loadbossmodel_gplvmplus();
R3 = loadbossmodel_gplvmvar();

Y = let
    X = loadoriginalspectra();
    replace(X[3000:5000,1:2:end]', Inf=>missing) # unseen data
end

```