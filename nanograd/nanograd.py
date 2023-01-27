#!/usr/bin/env python3

import math

"""
An implementation of a Nano Autograd Library 
Inspired by (and heavily influence by) by Micrograd by Andrej Karpathy
Initial implementation follows the implementation of Micrograd closely. 
"""


class Val:
    """
    Scalar value class to do Autograd things on
    """

    def __init__(self, data: float, label: str = "", _children: tuple = ()) -> None:
        self.data: float = data  # Value
        self.label: str = label  # Value from operation
        self.grad: float = 0  # Initial gradient
        self._children: set[Val] = set(_children)
        self._backward = lambda: None

    def __add__(self, y):
        """Addition"""
        y = y if isinstance(y, Val) else Val(data=y)
        x = Val(self.data + y.data, _children=(self, y), label="+")

        # Compute the accumulations of the gradient
        def _backward():
            self.grad += x.grad
            y.grad += x.grad
        x._backward = _backward
        return x

    def __radd__(self, y):
        """Reversed Addition"""
        return self + y

    def __mul__(self, y):
        y = y if isinstance(y, Val) else Val(data=y)
        x = Val(self.data * y.data, _children=(self, y), label="*")

        def _backward():
            self.grad += y.grad * x.grad
            y.grad += self.grad * x.grad
        x._backward = _backward

        return x

    def __rmul__(self, y):
        return self * y

    def __pow__(self, y):
        assert isinstance(y, (float, int)), f"{y=} must be an int or a float"
        y = y if isinstance(y, Val) else Val(data=y)
        x = Val(self.data**y.data, label=f"**{y}", _children=(self,))
        def _backward(): 
            self.grad += y * self ** (y - 1) 
        x._backward = _backward
        return x 

    def __neg__(self):
        """Negation"""
        return self * -1

    def __sub__(self, y):
        """Subtraction"""
        y = y if isinstance(y, Val) else Val(data=y)
        return self + (-y)

    def __rsub__(self, y):
        """Reversed Subtraction"""
        return y + (-self)

    def __repr__(self) -> str:
        return f"Val(data = {self.data}, grad = {self.grad})"

    def __truediv__(self, y):
        """Division"""
        return self * y**-1

    def __rtruediv__(self, y):
        """Reversed Division"""
        return y * self**-1

    def relu(self):
        """Rectified Linear Unit Activation Function"""
        x = Val(max([0, self.data]), _children=(self,), label="ReLU")

        def _backward(): 
            self.grad += (x.grad > 0) * x.grad

        x._backward = _backward
        return x 

    def exp(self): 
        """Exponential"""
        x = Val(math.exp(self.data), _children=(self,), label="Exp")
        def _backward(): 
            self.grad += math.exp(self.data)
        x._backward = _backward
        return x

    def sigmoid(self):
        """Sigmoid activation Function"""
        x = Val(self.exp()  / (self.exp() + 1), _children=(self,), label = "Sigmoid")
        def _backward(): 
            self.grad += (-self).exp() / ((-self).exp() + 1) ** 2
        x._backward = _backward
        return x 

    def tanh(self):
        """Tanh Activation Function"""
        x = Val(((2 * self).exp() - 1) / ((2 * self).exp() + 1), 
                _children=(self,),
                label="Tanh"
        )
        def _backward(): 
            self.grad += (2 / ((2 * self).exp() + 1) ** 2)
        x._backward = _backward
        return x 


    def _build_topological_graph(self):
        """Builds a Topological Graph to compute the backwards pass"""


if __name__ == "__main__":
    a = Val(-4.0)
    b = Val(2.0)
    c = a + b
    d = a * b
    print(d)
    d._backward()
    print(a)
