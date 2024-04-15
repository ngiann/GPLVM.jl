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

```
using GPLVM, PPCASpectra, LinearAlgebra, JLD2, Dates
BLAS.set_num_threads(10)

---------------

X = loadoriginalspectra();
Ytr = replace(Matrix(X[1:700,1:2:end]'), Inf => Inf);
R1 = gplvmvar(Ytr,seed=1,Q=3,iterations=3); 

R1 = gplvmvar(Ytr,seed=1,Q=3,iterations=15_000); JLD2.save("boss_gplvmvar.jld2","when",now(), "R",R1, "cmd","""R1 = gplvmvar(Ytr,seed=1,Q=3,iterations=15_000)""")





----------

X = loadoriginalspectra();
Ytr = replace(Matrix(X[1:700,1:2:end]'), Inf => Inf);
idx = findall(x->x<=0,Ytr);
Ytr[idx] .= Inf;
logYtr = log.(Ytr)

R2 = gplvmvar(logYtr,seed=1,Q=3,iterations=3); 

R2 = gplvmvar(logYtr,seed=1,Q=3,iterations=15_000); JLD2.save("boss_gplvmvar_log.jld2","when",now(), "R",R2, "cmd","""R2 = gplvmvar(logYtr,seed=1,Q=3,iterations=15_000)""")


----------

X = loadoriginalspectra();
Ytr = replace(Matrix(X[1:700,1:2:end]'), Inf => Inf);
R3 = gplvmplus(Ytr,seed=1,Q=3,iterations=3); 

R3 = gplvmplus(Ytr,seed=1,Q=3,iterations=15_000); JLD2.save("boss_gplvmplus.jld2","when",now(), "R",R3, "cmd","""R3 = gplvmplus(Ytr,seed=1,Q=3,iterations=15_000)""")




```