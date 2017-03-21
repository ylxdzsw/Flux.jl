@net type Affine2
  W
  b
  x -> x*W + b
end

# Create an Affine2 layer with parameters
a = Affine2(randn(10, 5), randn(1, 5))

Affine2(in::Int, out::Int)  = Affine2(randn(in, out), randn(1, out))

# Generate a random input
xs = rand(10)

# See the model's output
a(xs)

# Add a sigmoid
σ(a(xs))

# Run on MXNet
ta = mxnet(a)
ta(xs)

################################

@net type TLP
  first
  second
  function (x)
    l1 = σ(first(x))
    l2 = softmax(second(l1))
  end
end

# Construct a TLP (watch out for shape errors!)
t = TLP(Affine2(10, 20), Affine2(21, 5))

# Convert to MXNet
tt = mxnet(t)

# tt(xs)

@step runmodel(t, xs')
