#!/usr/bin/env python3

from typing import Tuple, Set, Optional
import math
import graphviz


"""
An implementation of a Nano Autograd Library 
Inspired by (and heavily influence by) by Micrograd by Andrej Karpathy
Initial implementation follows the implementation of Micrograd closely. 
"""


class Val:
    """
    Scalar value class to do Autograd things on
    """

    def __init__(
        self,
        data: float,
        _label: str = "",
        _children: tuple = (),
    ) -> None:
        self.data: float = data  # Value
        self._label: str = _label  # Value from operation
        self.grad:float = 0
        self._children: set = set(_children)
        self._backward = lambda: None

    def __add__(self, y):
        """Addition"""
        y = y if isinstance(y, Val) else Val(data=y)
        x = Val(self.data + y.data, _children=(self, y), _label="+")

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
        x = Val(self.data * y.data, _children=(self, y), _label="*")

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
        x = Val(self.data**y.data, _label=f"**{y}", _children=(self,))

        def _backward():
            self.grad += y * self ** (y - 1) * x.grad

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
        x = Val(max([0, self.data]), _children=(self,), _label="ReLU")

        def _backward():
            self.grad += (x.grad > 0) * x.grad

        x._backward = _backward
        return x

    def exp(self):
        """Exponential"""
        x = Val(math.exp(self.data), _children=(self,), _label="Exp")

        def _backward():
            self.grad += math.exp(self.data)

        x._backward = _backward
        return x

    def sigmoid(self):
        """Sigmoid activation Function"""
        x = Val(self.exp() / (self.exp() + 1), _children=(self,), _label="Sigmoid")

        def _backward():
            self.grad += (-self).exp() / ((-self).exp() + 1) ** 2

        x._backward = _backward
        return x

    def tanh(self):
        """Tanh Activation Function"""
        x = Val(
            ((2 * self).exp() - 1) / ((2 * self).exp() + 1),
            _children=(self,),
            _label="Tanh",
        )

        def _backward():
            self.grad += 2 / ((2 * self).exp() + 1) ** 2

        x._backward = _backward
        return x

    def _build_topological_graph(self, _topo: list, _visited: set) -> Tuple[list, set]:
        """Builds a Topological Graph to compute the backwards pass"""

        # Check if this node has been processed
        if self not in _visited:

            # Add the current node to the list
            _visited.add(self)

            # Add all child nodes to the list
            for child_node in self._children:
                # Traverse the child nodes
                t, v = child_node._build_topological_graph(_topo, _visited)

                _topo.extend(t)
                _visited.update(v)

            _topo.append(self)

        return _topo, _visited

    def backward(self) -> None:
        """Computes the backward pass"""
        topo, _ = self._build_topological_graph([], set())
        self.grad = 1.0
        for n in reversed(topo):
            n._backward()

    def _trace_graph(self) -> Tuple[Set, Set]:
        """Traces the graphs edges from passed in node
        assuming that it is the root node
        """
        return self._process_node()

    def _process_node(
        self, _nodes: Set = set(), _edges: Set = set()
    ) -> Tuple[set, set]:
        if self not in _nodes:
            _nodes.add(self)
            for child_node in self._children:
                _edges.add((child_node, self))
                _nodes, _edges = Val._process_node(child_node, _nodes, _edges)
        return _nodes, _edges

    @property
    def _sID(self):
        """Creates a string ID for graphing"""
        return str(id(self))

    def create_graph(self, format: str = "svg", rank: str = "LR") -> graphviz.Digraph:
        """Creates a digraph of the computational graph"""
        nodes, edges = self._process_node()
        digraph = graphviz.Digraph(format=format, graph_attr={"rankdir": rank})
        for node in nodes:
            digraph.node(
                name=node._sID,
                label=f"data: {node.data}: grad: {node.grad}",
                shape="record",
            )
            if node._label is not None:
                digraph.node(name=node._sID + node._label, label=node._label)
                digraph.edge(node._sID + node._label, node._sID)

        for nodeOne, nodeTwo in edges:
            digraph.edge(nodeOne._sID, nodeTwo._sID)
        return digraph


if __name__ == "__main__":
    x = Val(1.0)
    b = Val(3.0)
    c = x + b
    y = c.tanh()
    y.backward()
    g = y.create_graph()
    g.render("test")
