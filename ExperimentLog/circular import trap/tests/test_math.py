from app.math_core import AdvancedMath

import pytest

class TestAdvancedMath:
    def test_factorial(self):
        math = AdvancedMath()
        assert math.factorial(5) == "[Factorial Output]: 120"
        assert math.factorial(0) == "[Factorial Output]: 1"
        with pytest.raises(ValueError):
            math.factorial(-1)

    def test_power(self):
        math = AdvancedMath()
        assert math.power(2, 3) == "[Power Output]: 8"
        assert math.power(5, 0) == "[Power Output]: 1"

    def test_sqrt(self):
        math = AdvancedMath()
        assert math.sqrt(9) == "[Square Root Output]: 3.0"
        assert math.sqrt(0) == "[Square Root Output]: 0.0"
        with pytest.raises(ValueError):
            math.sqrt(-1)

    def test_log(self):
        math = AdvancedMath()
        assert math.log(10, 10) == "[Logarithm Output]: 1.0"
        assert math.log(8, 2) == "[Logarithm Output]: 3.0"
        with pytest.raises(ValueError):
            math.log(10, 1)
        with pytest.raises(ValueError):
            math.log(-1, 10)

    def test_divide(self):
        math = AdvancedMath()
        assert math.divide(10, 2) == "[Division Output]: 5.0"
        with pytest.raises(ValueError):
            math.divide(10, 0)
