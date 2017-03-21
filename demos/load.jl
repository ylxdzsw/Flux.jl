using Flux
using Flux: runmodel

cd(dirname(@__FILE__))

include("1-mnist.jl")
include("2-custom.jl")
