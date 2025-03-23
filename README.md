# GPLVM

## How to run
```
using Pkg
Pkg.activate(".")
using GPLVM

# load data
Y, L = loadoil()

# warmup
X = gplvm(Y; iterations = 1)

# proper run - check memory consumption
X = gplvm(Y; iterations =  500)

# plot result
using PyPlot # use your favourite plotting package here
for l in unique(L)
    idx = findall(L .== l)
    plot3D(Y[1,idx], Y[2,idx], Y[3,idx], "o")
end
```