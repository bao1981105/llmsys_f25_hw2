from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """
        Accumulates the derivative (gradient) for this Variable.

        Args:
            x (Any): The gradient value to be accumulated.
        """
        pass

    @property
    def unique_id(self) -> int:
        """
        Returns:
            int: The unique identifier of this Variable.
        """
        pass

    def is_leaf(self) -> bool:
        """
        Returns whether this Variable is a leaf node in the computation graph.

        Returns:
            bool: True if this Variable is a leaf node, False otherwise.
        """
        pass

    def is_constant(self) -> bool:
        """
        Returns whether this Variable represents a constant value.

        Returns:
            bool: True if this Variable is constant, False otherwise.
        """
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """
        Returns the parent Variables of this Variable in the computation graph.

        Returns:
            Iterable[Variable]: The parent Variables of this Variable.
        """
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """
        Implements the chain rule to compute the gradient contributions of this Variable.

        Args:
            d_output (Any): The gradient of the output with respect to the Variable.

        Returns:
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple
                contains a parent Variable and the corresponding gradient contribution.
        """
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # BEGIN ASSIGN1_1
    # TODO
    # parents of variable means the inputs needed to calculate this variable result
    visited = set()
    res = []

    def dfs(variable: Variable) -> None:
        if variable.is_constant() or variable.unique_id in visited:
            return
        visited.add(variable.unique_id)
        for input in variable.parents:
            dfs(input)
        res.append(variable)
    dfs(variable)
    # res vector stores the variables(tensors) from input to output, forward pass topological order
    # res[::-1] is backward propagation order, we process output first
    return res[::-1]
    # END ASSIGN1_1


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """

    # Example: Visualization
    # a(leaf) - -exp --> c \
    #                    * --> e
    # b(leaf) - -inv --> d /

    # # Example Forward pass
    # a = Tensor.make([2.0], (1,), backend=backend)
    # b = Tensor.make([3.0], (1,), backend=backend)
    # c = a.exp()  # c = exp(a)
    # d = 1 / b  # d = inv(b)
    # e = c * d  # e = c * d
    # # Example Backward pass
    # e.backward()  # triggers backpropagate(e, Tensor([1.0]))
    # # During backpropagation:
    # # e.chain_rule(Tensor([1.0])) -> [(c, Tensor([1.0]) * d), (d, Tensor([1.0]) * c)]
    # # c.chain_rule(Tensor([1.0]) * d) -> [(a, (Tensor([1.0]) * d) * exp(a))]
    # # d.chain_rule(Tensor([1.0]) * c) -> [(b, (Tensor([1.0]) * c) * (-1 / b^2))]


    # BEGIN ASSIGN1_1
    backward_topo_order = topological_sort(variable)
    grad_map = {variable.unique_id: deriv}
    for v in backward_topo_order:
        if v.is_constant():
            continue
        grad = grad_map.get(v.unique_id, None)
        if grad is None:
            continue
        # is_leaf: it was created directly by the user and not as the result of a differentiable operation
        # most operations are differentiable in our discussion scope
        if v.is_leaf():
            v.accumulate_derivative(grad)
        else:
            # v.chain_rule returns (input1 of v, ∂loss/∂(input1 of v)), the argument to v.chain_rule grad = ∂loss/∂v
            # basically, v.chain_rule appends ∂v/∂(input1 of v) to grad(=∂loss/∂v)
            for parent, parent_grad in v.chain_rule(grad):
                if parent.unique_id in grad_map:
                    grad_map[parent.unique_id] += parent_grad
                else:
                    grad_map[parent.unique_id] = parent_grad

    # END ASSIGN1_1


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
