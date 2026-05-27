from app.lexer import Lexer
from app.parser import Parser
from app.evaluator import Evaluator
import pytest

def run_query(text: str, variables: dict = None) -> float:
    tokens = Lexer(text).tokenize()
    ast = Parser(tokens).parse()
    return Evaluator(variables).evaluate(ast)

def test_basic_addition():
    assert run_query("2 + 3") == 5

def test_subtraction():
    assert run_query("10 - 4") == 6

def test_multiplication():
    assert run_query("3 * 5") == 15

def test_division():
    assert run_query("12 / 3") == 4.0

def test_precedence_arithmetic():
    # Precedence: multiplication takes precedence over addition
    assert run_query("2 + 3 * 4") == 14

def test_parentheses():
    assert run_query("(2 + 3) * 4") == 20

def test_comparison_greater():
    assert run_query("10 > 5") == 1
    assert run_query("5 > 10") == 0

def test_comparison_less():
    assert run_query("3 < 8") == 1
    assert run_query("8 < 3") == 0

def test_comparison_equal():
    assert run_query("7 = 7") == 1
    assert run_query("7 = 9") == 0

def test_variable_integer():
    assert run_query("int_x + 5", {"int_x": 10}) == 15

def test_variable_float():
    assert run_query("float_y * 2.0", {"float_y": 2.5}) == 5.0

def test_complex_precedence():
    # Parentheses nesting, precedence, variable schemas, and type boundaries
    # float_y + 1.5 => 2.5 + 1.5 = 4.0 (float)
    # int_x + 2 => 10 + 2 = 12 (int).
    # Since comparison strictly checks float vs int, comparing int and float will crash!
    # The agent must solve this type mismatch in evaluator!
    variables = {"int_x": 10, "float_y": 2.5}
    assert run_query("((2 + int_x) * (float_y + 1.5)) > 20.0", variables) == 1

def test_divide_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        run_query("10 / 0")
