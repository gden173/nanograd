import random
from typing import List
from nanograd.nanograd import Val


class Module:
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, in_dims: int, activation: str = "relu"):
        self._activations = ["tanh", "relu", "sigmoid"]
        assert (
            activation in self._activations
        ), f"{activation} not in {self._activations}"
        self.activation = activation
        self.in_dims = in_dims
        self.weights = [Val(random.normalvariate(0, 1)) for _ in range(self.in_dims)]
        self.bias = Val(0)

    def __call__(self, x):
        """Perform the forward pass through this layer"""
        activation = sum(i * j for i, j in zip(self.weights, x)) + self.bias

        # TODO: fix this
        if self.activation == "tanh":
            return activation.tanh()

        if self.activation == "relu":
            return activation.relu()

        if self.activation == "sigmoid":
            return activation.sigmoid()

    def parameters(self):
        return self.weights + [self.bias]

    def __repr__(self):
        act = self.activation.capitalize()
        return f"{act}(Neuron(in_dims = {self.in_dims}))"


class Layer(Module):
    def __init__(self, n_in: int, n_out: int, **kwargs):
        self.n_in = n_in
        self.n_out = n_out
        self.neurons = [Neuron(self.n_in, **kwargs) for _ in range(self.n_out)]

    def __call__(self, x):
        f = [n(x) for n in self.neurons]
        return f.pop() if self.n_out == 1 else f

    def parameters(self):
        return [params for n in self.neurons for params in n.parameters()]

    def __repr__(self):
        return f"Layer {{ {'<->'.join(str(n) for n in self.neurons)} }}"


class MLP(Module):
    def __init__(self, layers: List[Layer]):
        assert len(layers) > 0, f"Must pass in at least 1 layer"
        self.layers = layers
        self.num_in = layers[0].n_in
        self.num_out = layers[-1].n_out

        # Check that the dimension of each layer works
        if len(layers) > 1:
            for i, (l1, l2) in enumerate(zip(layers, layers[1:])):
                assert (
                    l1.n_out == l2.n_in
                ), f"Layer {i} dimensions must agree !! {l1.n_out} != {l2.n_in}"

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [params for l in self.layers for params in l.parameters()]

    def __repr__(self):
        layers: str = "\n".join(f"Layer({l.n_in}, {l.n_out})" for l in self.layers)
        return f"""
           MLP 
           {{
               {layers} 
            }}
           """
