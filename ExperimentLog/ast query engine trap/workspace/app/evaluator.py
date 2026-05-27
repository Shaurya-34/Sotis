from .parser import AST, BinOp, Num, Var
from typing import Dict, Any

class Evaluator:
    def __init__(self, variables: Dict[str, Any] = None) -> None:
        self.variables = variables or {}

    def evaluate(self, node: AST) -> Any:
        if isinstance(node, Num):
            return node.value
        elif isinstance(node, Var):
            val = self.variables.get(node.name)
            if val is None:
                raise ValueError(f"Undefined variable: {node.name}")
            return val
        elif isinstance(node, BinOp):
            left_val = self.evaluate(node.left)
            right_val = self.evaluate(node.right)
            if node.op == '+':
                return left_val + right_val
            elif node.op == '-':
                return left_val - right_val
            elif node.op == '*':
                return left_val * right_val
            elif node.op == '/':
                if right_val == 0:
                    raise ZeroDivisionError("Cannot divide by zero")
                return left_val / right_val
            elif node.op == '>':
                return int(left_val > right_val)
            elif node.op == '<':
                return int(left_val < right_val)
            elif node.op == '=':
                return int(left_val == right_val)
        raise ValueError(f"Unknown node type: {{type(node).__name__}}")