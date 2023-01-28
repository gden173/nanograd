"""
An implementation of a Nano Autograd Library
Inspired by (and heavily influence by) by Micrograd by Andrej Karpathy
Initial implementation follows the implementation of Micrograd closely.
"""

from typing import Tuple, Set
import math
import graphviz


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
        self.grad: float = 0
        self._children: set = set(_children)
        self._backward = lambda: None

    def __add__(self, other_val):
        """Addition"""
        other_val = other_val if isinstance(other_val, Val) else Val(data=other_val)
        output = Val(
            self.data + other_val.data, _children=(self, other_val), _label="+"
        )

        # Compute the accumulations of the gradient
        def _backward():
            self.grad += output.grad
            other_val.grad += output.grad

        output.set_backward(_backward)
        assert not isinstance(output.data, Val), "Add"
        return output

    def __radd__(self, other_val):
        """Reversed Addition"""
        return self + other_val

    def __mul__(self, other_val):
        other_val = other_val if isinstance(other_val, Val) else Val(other_val)
        output = Val(
            self.data * other_val.data, _children=(self, other_val), _label="*"
        )

        def _backward():
            self.grad += other_val.data * output.grad
            other_val.grad += self.data * output.grad

        output.set_backward(_backward)
        return output

    def __rmul__(self, other_val):
        return self * other_val

    def item(self):
        """Returns the values data as a scalar (float)"""
        return self.data

    def set_backward(self, backward) -> None:
        """Sets the backward closure"""
        self._backward = backward

    @property
    def label(self) -> str:
        """Sets the label"""
        return self._label

    @label.setter
    def label(self, label: str) -> None:
        """Sets the label"""
        self._label = label

    @property
    def children(self) -> set:
        """gets the Children nodes"""
        return self._children

    @children.setter
    def children(self, children: set) -> None:
        """Sets the nodes children"""
        self._children = children

    def __pow__(self, other_val):
        assert isinstance(other_val, (float, int)), "only float or int powers"
        output = Val(self.data**other_val, _label=f"**{other_val}", _children=(self,))

        def _backward():
            self.grad += other_val * self.data ** (other_val - 1) * output.grad

        output.set_backward(_backward)
        return output

    def __neg__(self):
        """Negation"""
        return self * -1

    def __sub__(self, other_val):
        """Subtraction"""
        other_val = other_val if isinstance(other_val, Val) else Val(data=other_val)
        return self + (-other_val)

    def __rsub__(self, other_val):
        """Reversed Subtraction"""
        return other_val + (-self)

    def __repr__(self) -> str:
        return f"Val(data = {self.data}, grad = {self.grad})"

    def __truediv__(self, other_val):
        """Division"""
        return self * other_val ** (-1)

    def __rtruediv__(self, other_val):
        """Reversed Division"""
        return other_val * self ** (-1)

    def __len__(self):
        return 1

    def relu(self):
        """Rectified Linear Unit Activation Function"""
        output = Val(max([0, self.data]), _children=(self,), _label="ReLU")

        def _backward():
            self.grad += (output.data > 0) * output.grad

        output.set_backward(_backward)
        return output

    def exp(self):
        """Exponential"""
        if isinstance(self.data, Val):
            value = self.data.data
        else:
            value = self.data
        exponential = math.exp(value)
        output = Val(exponential, _children=(self,), _label="Exp")

        def _backward():
            self.grad += exponential * output.grad

        output.set_backward(_backward)
        return output

    def sigmoid(self):
        """Sigmoid activation Function"""
        output = self.exp() / (self.exp() + 1)
        output.label = "Sigmoid"

        def _backward():
            self.grad += (output.data * (1 - output.data)) * output.grad

        output.set_backward(_backward)
        return output

    def tanh(self):
        """Tanh Activation Function"""
        output = ((2 * self).exp() - 1) * (((2 * self).exp() + 1) ** (-1))
        xd2 = output.data**2
        output.label = "Tanh"

        def _backward():
            self.grad += (1 - xd2) * output.grad

        output.set_backward(_backward)
        return output

    def backward(self) -> None:
        """Computes the backward pass"""
        _topo = []
        _visited = set()

        def topo(node):
            if node not in _visited:
                _visited.add(node)
                for child_node in node.children:
                    topo(child_node)
                _topo.append(node)

        topo(self)

        self.grad = 1.0
        for node in reversed(_topo):
            node._backward() # pylint: disable-all

    def _trace_graph(self) -> Tuple[Set, Set]:
        """Traces the graphs edges from passed in node
        assuming that it is the root node
        """
        return self._process_node()

    def _process_node(self, _nodes=None, _edges=None) -> Tuple[set, set]:

        if _nodes is None:
            _nodes = set()
        if _edges is None:
            _edges = set()

        node_value, edge_values = set(), set()
        if self not in _nodes:
            _nodes.add(self)
            for child_node in self._children:
                _edges.add((child_node, self))
                node_value, edge_values = Val._process_node(child_node, _nodes, _edges)
        return node_value, edge_values

    @property
    def s_id(self):
        """Creates a string ID for graphing"""
        return str(id(self))

    def create_graph(
        self, output_format: str = "svg", rank: str = "LR"
    ) -> graphviz.Digraph:
        """Creates a digraph of the computational graph"""
        nodes, edges = self._process_node()
        digraph = graphviz.Digraph(format=output_format, graph_attr={"rankdir": rank})
        for node in nodes:
            digraph.node(
                name=node.s_id,
                label=f"data: {node.data}: grad: {node.grad}",
                shape="record",
            )
            if node.label is not None:
                digraph.node(name=node.s_id + node.label, label=node.label)
                digraph.edge(node.s_id + node.label, node.s_id)

        for node_one, node_two in edges:
            digraph.edge(node_one.s_id, node_two.s_id)
        return digraph


if __name__ == "__main__":
    a = Val(2, _label="a")
    b = a.sigmoid()
    print(b)
    b.backward()
    print(b)
    print(a)
