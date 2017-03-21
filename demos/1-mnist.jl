using Flux, MNIST

data = [(trainfeatures(i), onehot(trainlabel(i), 0:9)) for i = 1:60_000]
train = data[1:50_000]
test = data[50_001:60_000]

m = Chain(
  Input(784),
  Affine(128), relu,
  Affine( 64), relu,
  Affine( 10), softmax)

data[1][1]

m(data[1][1])

# Convert to MXNet
model = mxnet(m)

# An example prediction pre-training
model(data[1][1])

# Flux.train!(model, train, test, η = 1e-4)

# An example prediction post-training
model(data[1][1])

data[1][2]

# data[1][2]

using JLD
@load "mnist.jld"

@progress "train" for i = 1:10
  @progress "epoch" for i = 1:100
    sleep(0.01)
  end
end
